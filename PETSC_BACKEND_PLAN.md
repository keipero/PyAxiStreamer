# PETScバックエンド導入案

提案日: 2026-11-05（日本時間）。対象ブランチ: `feature/petsc-backend`。
本書は初期調査と導入方針の記録。2026-11-05にWindows専用Conda環境の実機検証と初期導入を完了した。
実施結果・起動手順は[PETSC_BACKEND_MIGRATION.md](PETSC_BACKEND_MIGRATION.md)を参照。
PoissonはLUを維持し、輸送は小さな対角係数へのLUゼロピボット判定を避けるためJacobiを採用した。
Poisson反復法の調整と領域分割は未実施。

## 推奨する進め方

Windowsネイティブの専用環境でPETSc・petsc4py・mpi4pyの動作を確認し、
FiPyのPETScバックエンドへ切り替える。最初は1プロセス・PoissonのLU求解で数値的な同等性を確認する。
その後、Poissonだけを反復法へ変更して性能を評価する。
格子、半セル境界条件、時間積分、反応式、状態管理、CSV仕様は維持する。

## 提案時の環境の確認結果

| 項目 | 現状 |
|---|---|
| Python | CPython 3.14.3、Windowsのvenv |
| FiPy | 4.0.3 |
| NumPy / SciPy | 2.5.4 / 1.18.1 |
| petsc / petsc4py / mpi4py | 現venvには未導入 |
| conda / mamba / cl / mpiexec / make | 調査時のPATH上には見つからない |
| Poisson・輸送 | 両方ともFiPy共通APIの`LinearLUSolver`を使用 |
| メッシュ | `serialComm`を明示した共有円筒格子 |

PATH上に見つからないツールがコンピュータ全体に存在しないとは断定しない。

## 1. Windows環境の成立確認

`petsc4py`はPETSc本体を呼び出すPythonバインディングであり、PETSc本体も必要。
FiPyのPETScバックエンドにはmpi4pyも必要で、MPIライブラリの整合性を確認する。
現在のSciPy検証環境と分け、Python 3.14系を第一候補として調査する。

調査時点のPyPI版petsc4py 3.26.1にはソース配布のみが掲載されている。
`pip install petsc petsc4py`を、現在のWindows venvへバイナリを入れるだけの手順とは扱わない。
また、conda-forgeのPETSc対応プラットフォーム一覧にはwin-64が掲載されていない。
[petsc4py配布ファイル](https://pypi.org/project/petsc4py/#files)、[conda-forge PETSc](https://anaconda.org/conda-forge/petsc)

Windowsでの導入候補は次の順で評価する。

| 候補 | 評価内容 |
|---|---|
| Windows向けバイナリ配布を使う専用conda環境 | 必要なPETSc・petsc4py・mpi4pyとMPIランタイムの配布実在性、Python対応、DLL整合性を確認する |
| Windowsネイティブのソースビルド | 上記構成を確立できない場合、PETSc公式のMSVC等によるWindowsビルド手順を基に検討する |

バイナリ配布の調査候補としてPrecise Simulationの公開リポジトリがある。
同提供元はWindows用PETSc/petsc4pyの配布とPython 3.14対応を案内し、
公開レシピにはpetsc4py 3.25.4・real型・Intel MPI構成とPython 3.14でのテストがある。
その後、Condaの公開索引でwin-64用petsc4py 3.25.6の配布と、対応するPETSc 3.25.6への依存を確認した。
確認したabi3ビルドは`cpython >=3.12`と`_python_abi3_support 1.*`を要求する。
提案時はFiPy 4.0.3との実機互換性が未確認だったが、その後にimport・既知解・連成計算の検証が成功した。
PETSc公式配布とは区別し、今回確認した版の構成を採用する。
[提供元の説明](https://github.com/precise-simulation/fenics-windows-ci)、
[petsc4pyレシピ](https://github.com/precise-simulation/fenics-windows-ci/blob/main/recipes/petsc4py/recipe.yaml)

公式WindowsソースビルドではCygwinのシェル・makeとWindowsコンパイラを使う経路が案内されている。
実行対象をWindowsネイティブのPythonとし、コンパイラ・Python ABI・PETSc共有ライブラリ・MPIを揃える。
ビルド用のUnix系ツールも必要になるため、この経路は環境構築工数を別途見積もる。
[PETSc公式Windows導入手順](https://petsc.org/release/install/windows/)

FiPyの公式対応表でもPETScのWindows欄には対応の印がない。
これを一律に動作不能と解釈せず、実機で成立を確認する工程を最初に置く。
[FiPyソルバ対応表](https://pages.nist.gov/fipy/en/stable/SOLVERS.html)

環境成立の完了条件は、同じPythonで次が全て成功すること。

1. `petsc4py.PETSc`と`mpi4py.MPI`のimport、版・MPI実装の表示。
2. PETScの実数・倍精度構成と、Python/NumPyの対応を確認。
3. PETScの小さい既知解を持つ連立方程式を1プロセスで解く。
4. FiPy 4.0.3がPETScを選択し、小さい拡散方程式を解く。
5. 成立したPython・PETSc・petsc4py・MPI・NumPy・SciPyの組合せを固定して記録。

importだけの成功を導入完了とはしない。利用する配布経路が確定してから、具体的な導入コマンドを確定する。

### 現venvでのpip導入結果

`python -m pip install mpi4py petsc4py`はpetsc4pyのビルド依存であるPETSc 3.26.1の構築中に、
`Windows python detected. Please rerun ./configure with cygwin-python.`で停止した。
当初の確認時には現venvに両パッケージともなかった。後の確認では現venvにmpi4pyが存在したが、
単独起動ではMPI DLLを読み込めなかった。既存venvを変更せず、新しい専用環境で検証した。
原因はPETSc本体のWindows構築経路にあり、ソースをクローンするだけでは解消しない。
次の導入候補はWindows用バイナリを使う専用conda環境とする。

### Windows用バイナリ環境の依存解決

既存のAnacondaに含まれるconda 24.11.3をPythonモジュール経由で使用し、
precise-simulationとconda-forgeのwin-64索引に対してcreateのdry-runを行った。
classicソルバは時間内に完了せず、libmambaソルバで次の組合せの依存解決に成功した。

| パッケージ | 解決した版 | 配布元 |
|---|---|---|
| Python | 3.14.10 | conda-forge |
| NumPy / SciPy | 2.5.4 / 1.18.1 | conda-forge |
| FiPy | 4.0.3 | conda-forge |
| PETSc / petsc4py | 3.25.6 / 3.25.6 | precise-simulation |
| mpi4py | 4.1.2 | conda-forge |
| Intel MPI runtime | 2021.17.0 | conda-forge |

環境定義は[environment-petsc-win.yml](environment-petsc-win.yml)に記録した。
初回は依存解決までを確認した。その後、利用者による環境作成を経て、実際の求解・回帰検証が成功した。
PETScソースのクローンとWindows上でのソースビルドを、この導入経路の前提とはしない。

作業フォルダPyAxiStreamerで、PowerShellから次を実行する。
既存venvとは別のenv_pyaxistreamer_petscを作る。

```powershell
& "$HOME\anaconda3\python.exe" -m conda env create --solver libmamba --prefix .\env_pyaxistreamer_petsc --file .\pyaxistreamer-master\environment-petsc-win.yml
```

作成後のimport・小規模求解は`tools/check_petsc_backend.py`で再確認できる。
この環境ではConda経由のDLL設定と`python -E`を使用する。具体的なコマンドは実施結果を参照。

## 2. FiPyバックエンドの切替

FiPy標準の`FIPY_SOLVERS`を使用する。PowerShellでは、FiPyをimportする前に設定する。

```powershell
$env:FIPY_SOLVERS = "petsc"
python -c "import fipy; print(fipy.__version__); print(fipy.solvers.solver_suite)"
```

切替単位は新しいPythonプロセス。Notebookでは設定後にカーネルを再起動する。
同じプロセス内でSciPyとPETScを混在させず、比較は同じ検証スクリプトを別プロセスで実行する。
現在の`from fipy.solvers import LinearLUSolver`を基本とし、独自のPETSc行列組立は追加しない。
[FiPyの選択方法・混在の制約](https://pages.nist.gov/fipy/en/stable/SOLVERS.html)

初期のコード変更は次に絞る。

- PETScの行列・ベクトル型でも、Poissonの独立した最大相対残差検査が成立することを確認し、必要な箇所だけ調整する。
- 収束失敗・非有限解での状態復元を維持し、FiPyの警告だけで成功扱いしない。
- 実際に使ったバックエンド、PETSc/petsc4py/MPIの版、求解法を出力メタデータへ記録する。
- 依存関係と起動方法を文書化し、新規関数にはNumPy形式docstringと可能な限りExamplesを付ける。

バックエンド選択はimport前に必要なため、import後に作るSimulationConfigだけで切り替える構成は採らない。

## 3. 1プロセス・LUでの精度検証

Poissonと電子輸送を両方FiPy/PETScへ切り替え、PoissonはLU求解を維持する。
輸送は対角行列をpreonly/Jacobiで解く。PETSc化とPoisson反復法への変更を分けて検証する。

- 独立した小格子直接解・既知解・正負電圧・内部電極と半セル境界。
- Poissonの現行残差指標`max(abs(A*phi-b))/max(abs(b))`。既定許容値1e-10を維持する。
- 粒子収支、3区分の境界流出、ゼロ流入、D=0層、非負性と試行棄却・状態復元。
- 提案時の109回帰テスト・324 doctest。輸送の3検査を追加し、両バックエンドで112回帰テスト・324 doctestが成功した。
- 正負DC・インパルス、D=0/0.5、固定/適応の12条件を同じ時刻で比較する。

小格子直接解との比較は既存の検証許容値を維持する。連成計算については、
事前の受入目安を配列スケール誤差1e-8とし、密度・電位・電場・反応係数・流出数を個別に記録する。
これは既存の単体検査や残差許容値を緩める指定ではない。
適応刻みで更新経路が分かれた場合は、その原因と同一出力時刻での誤差を評価する。
CSVの名前・単位・並び順・時刻表は維持する。

## 4. Poisson反復法の性能評価

LUによる導入確認が成功した後に、Poissonの第一候補をGMRES＋ILUとして評価する。
CGは実際の組立行列の対称性・正定値性を確認してから候補に加える。
GAMG等は導入済みPETScビルドでの利用可能性と効果を確認して追加評価する。
[PETSc KSPと前処理](https://petsc.org/release/manual/ksp/)

PETSc側の残差ノルムと、現行の最大絶対値による独立検査は区別する。
同じtolerance値だけを移植せず、独立検査1e-10を満たすようKSPの相対・絶対許容値を調整する。

電子輸送は現行の陽的方程式なので、Poissonとは別に求解法を選ぶ。
実機検証で小さな対角係数がLUのゼロピボット判定に掛かったため、初期導入からJacobiを採用した。
両方へ同じグローバルPETSC_OPTIONSを適用してしまわないように構成する。

ローカルのFiPy 4.0.3実装では、PETScのLU/Krylov求解でKSPを毎回作る。
バックエンド切替だけでLU因子分解や前処理が時間ステップ間で再利用されるとは見込まない。
初期導入の精度判定と、速度改善の判定を分ける。

小格子と標準200×150格子で、初期化・組立・求解・計算全体・I/Oを分けて計測する。
ウォームアップ後に同じ条件・精度・スレッド数で複数回測り、中央値とばらつきを記録する。
改善が確認できた設定だけを既定候補にする。

## 初期導入の範囲と完了条件

現在の格子はserialCommを使い、物理量や集計も全格子のNumPy配列を前提としている。
このため、初期導入は1プロセスを対象とする。MPIでの領域分割は、メッシュ・添字・
境界通信・全体集計の変更を含む別工程とする。PETScへの切替だけで分散並列化が完了するとは扱わない。

このブランチの初期導入完了条件は、Windows上の再現可能な環境構成、FiPy/PETScの1プロセス実行、
精度・保存則・失敗復元の検証成功、SciPyとの比較記録が揃うこと。
高速化の採用判断は反復法評価後に行う。

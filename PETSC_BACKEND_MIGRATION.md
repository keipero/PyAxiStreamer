# Windows PETSc初期導入の結果

検証日: 2026-11-05（日本時間）。対象ブランチ: `feature/petsc-backend`。
Windowsネイティブの専用Conda環境で、FiPy/PETScによる1プロセスの初期導入を完了した。
PoissonはLU、陽的電子輸送は対角求解を使う。Poisson反復法の性能調整とMPI領域分割は未実施。
導入方針の記録は[PETSC_BACKEND_PLAN.md](PETSC_BACKEND_PLAN.md)を参照。

## 確認した環境と起動

| 項目 | 確認した値 |
|---|---|
| OS / Python | Windows win-64 / 3.14.10 |
| FiPy | 4.0.3 |
| NumPy / SciPy | 2.5.4 / 1.18.1 |
| PETSc / petsc4py | 3.25.6 / 3.25.6、precise-simulationのバイナリ |
| mpi4py | 4.1.2 |
| MPI実装 | Intel MPI Library 2021.17 for Windows、impi_rt=2021.17.0 |
| PETScスカラー / MPIプロセス数 | 実数float64 / 1 |

環境定義は[environment-petsc-win.yml](environment-petsc-win.yml)。既存venvを置き換えず、
親の`PyAxiStreamer`フォルダから次で作成した専用環境を使用した。

```powershell
& "$HOME\anaconda3\python.exe" -m conda env create --solver libmamba `
  --prefix .\env_pyaxistreamer_petsc --file .\pyaxistreamer-master\environment-petsc-win.yml
```

以下の実行コマンドは`pyaxistreamer-master`フォルダを前提とする。

```powershell
# 2×2のPETSc既知解、FiPy拡散既知解、小格子連成をまとめて確認
& "$HOME\anaconda3\python.exe" -m conda run --dev --prefix ..\env_pyaxistreamer_petsc `
  --no-capture-output python -E tools\check_petsc_backend.py

# PETScを選択して計算
$env:FIPY_SOLVERS = "petsc"
& "$HOME\anaconda3\python.exe" -m conda run --dev --prefix ..\env_pyaxistreamer_petsc `
  --no-capture-output python -E -m streamer_drift_diffusion --t-end=1e-12 --output=outputdata_petsc_smoke

# 回帰テスト
& "$HOME\anaconda3\python.exe" -m conda run --dev --prefix ..\env_pyaxistreamer_petsc `
  --no-capture-output python -E -m unittest discover -s tests
```

既存Anacondaのconda.exeは起動できなかったため、Pythonモジュールと`conda run --dev`を使用した。
さらに、この起動ではベース環境のPYTHONPATHが引き継がれたため、`python -E`で無視する。
検証スクリプトはNumPyの読込元も表示する。ベース環境のNumPyが混入していないことを確認できる。
MPIランタイムのDLL検索パスも必要なので、新環境のpython.exe単独呼出しより上記を使用する。

自動検証の制限された実行環境ではMPI初期化が失敗し、Condaの環境設定を行った通常権限の実行では成功した。
ネットワーク設定・ファイアウォール・MPIプロバイダの変更は行っていない。
Condaの環境設定と実行制限を同時に変えたため、初期化失敗の原因をどちらか一方と断定しない。

SciPyとの切替は`FIPY_SOLVERS=scipy`として新しいPythonプロセスで行う。
バックエンド選択はFiPyのimport前に必要。Notebookでも設定後に新しいカーネルを使用する。

## 計算コードの変更

Poissonの方程式、半セルDirichlet条件、共有格子、LU求解、最大相対残差検査、失敗時の電位復元はそのまま使えた。
PETSc固有の行列組立や、独自の線形ソルバー層は追加していない。

電子輸送の未知量に掛かる行列は、正の対角成分`cellVolume/dt`だけからなる。
既存の小格子検査では、対角成分8.75e-15～9.625e-14に対し、PETSc LUの絶対ゼロピボット閾値が
約2.22e-14となり、正の対角行列でもゼロピボットで失敗した。このため輸送には最初から
`KSP=preonly, PC=jacobi`を採用した。対角行列なのでJacobiによる対角逆数の適用で解ける。

FiPy 4.0.3の`LinearLUSolver`は生成時の前処理をLUに固定するため、PETScの場合だけ生成後に
公開属性`preconditioner`へFiPyの`JacobiPreconditioner`を設定する。
クラス名は`LinearLUSolver`だが、PETScの実行時診断でもpreonly/Jacobiであることを確認した。
SciPy側はLUを維持する。

輸送の収束判定は`criterion='initial'`とし、大きな右辺に対して相対的に小さい輸送変化も求解する。
FiPyの陽的拡散項は組立時の現在値を参照するため、Poissonと異なり組立前に現在値を0へリセットしない。
輸送係数が全て0の場合は、FiPyの旧値だけを更新して密度を厳密に保持する。
不要な対角求解で密度に丸め差が入ると、本来中性の状態に微小電荷・電場が生じるためである。
輸送に失敗した場合の正本密度の復元は維持した。

`run_config.json`には従来のFiPyバックエンドに加え、PETSc選択時だけ次を記録する。

- `petsc_version`、`petsc4py_version`、`mpi4py_version`、`mpi_library`
- `petsc_scalar_dtype`、`transport_default_pc`、`petsc_options`

最後の2項目は既定の前処理と`PETSC_OPTIONS`環境変数の記録であり、オプション適用後の全KSP診断ではない。
Poissonと輸送には異なる求解法を使うため、両方へ作用するグローバルKSPオプションを性能調整の手段にしない。

## 検証結果

| 検査 | 結果 |
|---|---|
| PETSc単独の2×2既知解 | 成功、LU、解`[1/11, 7/11]` |
| FiPyの1次元拡散既知解 | 成功、PETSc LU、8セル、境界値0/1 |
| 4×6セルの連成、50 V、D=0.5、2 ps | 成功、Poisson最大相対残差1.46e-16 |
| 標準200×150セルCLI、t_end=0、出力なし | 成功、初期Poissonを含む初期状態まで |
| SciPy / PETScの回帰テスト | それぞれ112件成功 |
| SciPy / PETScのdoctest | それぞれ324項目成功 |

回帰テストは既存109件に、ゼロ輸送時の厳密な密度保持・旧値更新、微小な輸送変化、
全量流出して右辺0になる場合の3件を追加した。既存の求解失敗・復元検査は、
ゼロ輸送の早期終了ではなく実際の求解を通る入力へ変更した。既存の精度許容値は緩めていない。
本体67定義すべてのdocstringと66定義のExamplesを維持している。

### 12条件のSciPy比較

同じConda環境でバックエンドごとに別プロセスを起動し、固定/適応 × D=0/0.5 × DC正負/インパルスを比較した。
共通設定は8×12セル、delta=1e-4 m、electrode_gap=6e-4 m、cathode_radius=2e-4 m、
nmax=1e16 m⁻³、background_ionization=1e11、dt=1e-11 s、t_end=1.1e-10 s、output_interval=2e-11 s。
DCは±500 V。インパルスの入力は時刻[ns]・電圧[kV]で`(0,0),(.02,1),(.04,-.5),(.11,0)`、
ピーク指定500 V。他の設定は既定値を使用した。

最終状態の全NumPy配列とne・phi、各出力CSVを数値読込みし、計900配列を比較した。
誤差は配列ごとに`max(abs(PETSc-SciPy))/max(max(abs(SciPy)), 1e-300)`と定義し、
構造化CSVは列ごとに評価した。文字列・整数、形状、有限性も検査した。

| 対象 | 最大スケール相対誤差 |
|---|---:|
| 全対象 | 1.14e-13 |
| 電子密度ne | 8.33e-16 |
| 電位phi | 1.13e-14 |
| 中心電場ex/ey/abs_eと面電場er_face/ez_face | 1.44e-14 |
| 累積境界電子流出数 | 6.01e-16 |
| CSV数値 | 4.27e-14 |

900配列のうち680配列が完全一致し、全対象で事前の受入目安1e-8を満たした。
最大値は半径面の移動度に生じた。改修前のコミット済みSciPyコードも同じConda環境で実行し、
改修後のSciPyと900配列すべてが完全一致した。比較用の一時コード・データは本体に残していない。

## コード量と残る範囲

空行・コメント・ASTで識別したdocstringを除く計算本体は875行から891行へ16行増加した。
小係数への対角求解、ゼロ輸送時の厳密な保持、バックエンドのメタデータ記録に必要な追加であり、
コード量の削減とは数えない。外部ライブラリ・テスト・文書・環境検証用toolsはこの本体集計には含めない。
docstringやExamplesを省略していない。

初期導入の検証は数値互換性を対象とした。その後、[実行速度の計測と改修案](PETSC_PERFORMANCE_PLAN.md)で
反復法と固定行列・LU再利用を評価した。計測専用の再利用試作で改善が見られ、本体適用を次の候補としている。
その後に再利用を本体へ適用し、両バックエンドの118回帰テスト（SciPyは専用4件を除外）と333 doctestが成功した。
現在の実装・測定結果は[Poisson再利用と並列評価](PETSC_REUSE_AND_PARALLEL.md)を参照。
現在のメッシュはserialComm、集計は全格子NumPy配列を前提とする。MPIの領域分割や長時間発展の妥当性は別途検証が必要。

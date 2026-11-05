# PETScを使わない共有メモリ並列化の検討

2026-11-05。**Windowsの単一PCを対象に、SciPyのLU再利用＋Numbaの共有メモリ計算を第一候補とする。**
今回は独立した計測・精度確認用の試作まで実施した。本体、Notebook、既存環境の切替・削除は実施していない。
先に検討したMPI領域分割の結果は[MPIの評価](MPI_PARALLEL_PLAN.md)へ残す。

## 提案する構成

| 処理 | 移行後の案 |
|---|---|
| 円筒格子、体積、面積、隣接関係、境界区分 | 現在のFiPyメッシュと共有gridを維持 |
| 電子密度・電位の正本 | 現在のFiPy CellVariableと2次元ビューを維持 |
| Poissonの行列・境界条件 | 現在のFiPy方程式、半セルDirichlet、金属の独立固定式を維持 |
| Poissonの直接求解 | SciPy SuperLUの分解を一度作り、右辺だけ更新 |
| 陽的な電子輸送 | 共通の面係数からNumbaで面流束→セル更新を計算 |
| 反応・物理係数・刻み制御・CSV出力 | 第1段階は既存処理を維持。移行後に再計測 |
| 並列数 | Numbaのスレッド数として明示。初期既定値は1を提案 |

`splu()`が返す分解オブジェクトの`solve()`を再利用できるため、PETScなしでも固定行列の利点を維持できる。
ただし、この案のSciPy SuperLU自体を共有メモリ並列ソルバーとして扱わない。
Poissonのマルチコア化が必要かどうかは、輸送を軽量化した後のプロファイルで判断する。
[SciPy公式仕様](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html)

Numbaの`parallel=True`と`prange`で、独立した面・セルをスレッドへ割り当てる。
環境変数でコア数を指定するだけでは、現在のFiPy時間発展全体が並列になるわけではない。
[Numba公式仕様](https://numba.readthedocs.io/en/stable/user/parallel.html)

## 試作の計測結果

同じ旧venv内のPython 3.14.3、FiPy 4.0.3、NumPy 2.5.4、SciPy 1.18.1、Numba 0.68.1で比較した。
初期化と10基本刻み相当の全時間発展を含む。CSV出力なし。各方式を1回ウォームアップ後に3回測定した中央値 [s]。
Numbaの初回JITはウォームアップへ含め、通常時間から除外して別途JSONへ保存した。

| 方式 | 200×150＝3万セル | 400×300＝12万セル |
|---|---:|---:|
| 既存FiPy/SciPy、毎回Poisson分解 | 3.703 | 22.551 |
| SciPy LU再利用、輸送はFiPy | 2.252 | 14.121 |
| LU再利用＋Numba 1スレッド | 0.929 | **4.268** |
| LU再利用＋Numba 2スレッド | **0.729** | 5.951 |
| LU再利用＋Numba 4スレッド | 1.001 | 5.851 |
| LU再利用＋Numba 8スレッド | 1.111 | 6.166 |

3万セルではNumba 1→2スレッドの速度比が約1.27倍。
FiPy/SciPyの3.703秒→0.729秒は約5.08倍だが、LU再利用と輸送行列の組立・求解を省いた効果も含む。
12万セルではNumba 1スレッドが最速で、2～8スレッドによる上積みは確認できなかった。
共有メモリ化は実行できるが、全コアの使用や並列数に比例した高速化は前提にしない。

測定順は方式ごとで交互実行ではなく、特に12万セルの複数スレッドにはばらつきがある。
最適スレッド数を確定する測定ではない。正式移行後、実使用の長い計算で方式順を入れ替えて再評価する。
現在のPETSc環境とはPythonのビルド等が異なるため、この表は現行PETScからの速度改善率を示していない。

生データ：[3万セル](SHARED_MEMORY_RESULTS_30K.json)、[12万セル](SHARED_MEMORY_RESULTS_120K.json)。
再現用試作：[benchmark_shared_memory.py](tools/benchmark_shared_memory.py)。
既存関数の差替えは計測の`with`内だけで行い、本体には適用しない。
試作は初回のFiPy求解と追加のSciPy分解の両方を測定へ含む。正式実装ではこの二重分解を除く。

## 数値確認

8×12格子で、拡散係数0／0.5 m²/s、固定／適応刻み、直流+500 V／−500 V／符号反転を含むインパルスの12条件を比較した。
基準は既存FiPy/SciPy。LU再利用のみ、LU再利用＋Numba 1／4スレッドの3方式を比較した。

- 状態の全NumPy配列、電子密度・電位、全保存フレーム、刻み履歴CSVを対象とした計3420比較が成功。
- 状態の最大スケール誤差は5.11×10⁻¹³、CSVは3.04×10⁻¹²。
- Numba 1／4スレッド間は、12条件すべてで対象配列・CSVがビット単位一致。
- 既存の拡散テスト12件を、試作適用中に1／4スレッドそれぞれで実行して成功。
  混合移流拡散100更新の粒子収支・非負性、半セル吸収、調和平均、拡散障壁、格子収束、時間精度、再試行の復元を含む。

誤差は`max|試作−基準| / max(max|基準|, 1e-300)`。配列ごとの全体スケールであり、各要素の相対誤差ではない。
CSVの比較許容値は保存形式の丸めを考慮して2×10⁻⁶、状態は10⁻⁸とした。
失敗復元・資源解放・メタデータ等を含む本体の全回帰確認は、正式移行時に別途必要。
試作中のrun_config.jsonは本体の方式名が残るため、この検証では比較対象に含めていない。

結果：[SHARED_MEMORY_NUMERICAL_RESULTS.json](SHARED_MEMORY_NUMERICAL_RESULTS.json)。
再現：[validate_shared_memory.py](tools/validate_shared_memory.py)。

## 輸送の実装方針

1. `ElectronTransport.prepare()`、共通grid、境界区分をそのまま使う。
   面積・調和平均D・半セル距離・流出係数・CFLを別実装へ複製しない。
2. 各面の旧密度から一次上流の移流流束と拡散流束を一度だけ計算する。
   気体間では両セルへ逆符号、吸収面では気体側からだけ流出させる。
3. 各セルが自分の4面の流束を読み、体積で割って新密度を計算する。
   複数スレッドが同じセルへ加算する方法を避ける。
4. 入力を並列に上書きせず別出力へ計算し、有限性検査後に`ne_var.setValue()`で反映する。
   `fastmath=False`を維持し、負密度のクリップは導入しない。
5. 流出数は同じ面係数と旧値から集計し、既存の受理・棄却処理を維持する。
   `2π`、電極内の密度0、ゼロ輸送の厳密保持も検証する。

FiPyの格子・Poisson利用は継続するが、電子輸送の離散更新はプロジェクト側の保守対象へ戻る。
FiPyだけで輸送を記述する現在の構成より数値コードが増える可能性がある。
コード削減という以前の優先事項との兼合いを、docstringを除いた実装行数で示す必要がある。
NumPy styleのdocstringと可能な限りのExamplesは維持する。

## 環境案と再現

[environment-cpu-win.yml](environment-cpu-win.yml)を用意した。
conda-forgeだけからFiPy、SciPy、Numba、ipykernelを解決する案で、PETSc・petsc4py・mpi4py・MPIランタイムを指定しない。
Condaのdry-runではWindows用123パッケージに解決でき、PETSc/MPIの混入はなかった。
[解決結果の記録](CPU_ENVIRONMENT_DRY_RUN.json)を保存した。
**新環境の作成・実行確認はまだ行っていない。**
Numbaの対応範囲も[公式の互換表](https://numba.readthedocs.io/en/stable/user/installing.html)で確認した。

旧venvにはDLLを読み込めないmpi4pyが残っていたため、今回の測定では一時ランチャーで
`sys.modules['mpi4py'] = None`を設定し、そのプロセス内だけFiPyの任意MPI依存を無効にした。
PETSc/MPIが読み込まれていないことは結果JSONにも記録している。
この回避策を本体へ導入せず、正式移行ではMPIを含まない新環境とCPU用Jupyter kernelで検証する。

新環境を作成した後の再現例（プロジェクトルートから）：

```powershell
# 作成候補。今回の検討ではdry-runまで実施した。
& "$HOME\anaconda3\python.exe" -m conda env create --solver libmamba `
  --prefix ..\env_pyaxistreamer_cpu --file .\environment-cpu-win.yml

# 以後はCPU環境を有効にして実行する。
python tools/benchmark_shared_memory.py --report SHARED_MEMORY_RESULTS.json
python tools/benchmark_shared_memory.py --nr 400 --nz 300 --delta 5e-6 --report SHARED_MEMORY_RESULTS_120K.json
python tools/validate_shared_memory.py --report SHARED_MEMORY_NUMERICAL_RESULTS.json
```

## 正式移行の順序

1. CPU環境を作成し、既存のFiPy/SciPy本体を実行・検証する。
2. SciPy LUの再利用を本体へ移す。固定行列の寿命、初回組立、独立残差検査、失敗時復元、close後の再作成を維持する。
   試作の差替えコードをそのまま本体のソルバーにはしない。
3. 輸送を小さなNumbaカーネルへ移し、1スレッドを数値基準にする。
   既存の全回帰・doctest、12条件比較、粒子収支、異常系、Notebookの先頭からの実行を確認する。
4. スレッド数を設定・CLI・メタデータへ追加する。名称は`cpu_threads`／`--cpu-threads`を候補とし、未実装の指定方法として扱う。
   1／2／4／8で実計算時間・初回JIT・初期化・I/Oを分けて評価する。BLAS/OpenMPの多重並列も抑える。
5. 検証成功後に、`_petsc.py`、PETSc専用分岐・設定・kernel登録、不要になったFiPy輸送方程式等を整理する。
   PETSc専用テストはSciPy再利用・残差・復元の同等な検証へ置換し、物理検証は残す。
   既存の利用者環境を自動アンインストールする必要はない。

Poissonが次の律速になれば、Windows向けの共有メモリ直接法として
[PyPardiso／Intel MKL PARDISO](https://github.com/haasad/PyPardiso)を追加評価できる。
今回は未導入・未計測で、Python 3.14環境での互換性・性能・分解再利用は未検証。
Cython/OpenMPはビルド保守、multiprocessing.shared_memoryはプロセス間同期が増えるため、第1候補にはしない。
当面はNumba輸送の採否を確定し、移行後のプロファイルに応じて対象を広げる。

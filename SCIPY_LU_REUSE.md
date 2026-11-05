# SciPy Poissonの固定行列・LU分解の再利用

2026-11-05。ブランチ：`feature/scipy-lu-reuse`。
改修前：`d6c5b6ab40031a5023787fe44394e25a833b5c0f`。

## 実装

初回のPoisson求解時だけFiPyで行列を組み、SciPyの`scipy.sparse.linalg.splu()`でLU分解する。
以後は電荷密度と印加電圧から右辺を更新し、同じSuperLUオブジェクトの`solve()`を使う。
格子・電極形状・左辺係数を計算中に変更しない現在のモデルが対象。
これらの変更時は新しい`PoissonSolver`を生成する。

初回の行列組立にはFiPyの公開API `justResidualVector()`を使い、別のLU求解を行わない。
FiPy 4.0.3のSciPy行列表現を参照する箇所は`electrostatics.py`に限定する。
FiPyの行列・右辺キャッシュを保持し、右辺キャッシュも毎回最新値へ更新する。
半セルDirichlet、金属セルの独立固定式、外端の解析境界電位は維持した。

関連仕様：[FiPyの行列組立API](https://pages.nist.gov/fipy/en/stable/generated/fipy.terms.term.html)、
[SciPy spluと分解済みsolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html)。

残差基準は`max(abs(A*phi-b))/max(abs(b)) <= poisson_tolerance`で、既定1e-10を維持する。
ゼロ右辺の分母下限は浮動小数点のtiny。入力・生成した右辺・解・残差の有限性も検査する。
前stepを初期値にした反復省略はなく、大きな境界電圧に対する微小な電荷変化も直接解く。

残差異常・分解や求解の例外では電位の正本を復元し、LUを解放する。
`PoissonSolver.close()`はLUへの参照を解除する。電位と行列は保持し、次の求解で再分解できる。
時間ループは成功・失敗とも終了時にclose()を呼ぶ。
単独使用では、必要な求解の終了後に`state.poisson.close()`を呼べる。

run_config.jsonに`SuperLU`、Poissonの`scipy`バックエンド、行列・分解の再利用を記録する。
輸送・反応・刻み制御・CSV形式は変更していない。PETScやNumbaの依存追加はない。
本体に改修前の毎回分解モードや比較用フラグは残していない。

## 数値検証

| 検査 | 結果 |
|---|---|
| 旧venv：Python 3.14.3、FiPy 4.0.3、SciPyバックエンド | 回帰114件・doctest 329項目成功 |
| 既存Conda：Python 3.14.10、FiPy 4.0.3、SciPyバックエンド | 回帰114件・doctest 329項目成功 |
| CLI：12×16格子、3 ps、出力なし | 正常終了 |
| 旧venvのpip check | 依存不整合なし |
| 改修前後の12連成条件 | 1128配列・CSV数値列の比較成功 |

回帰テストでは、独立した直接解・既知解・Gauss則・微小電荷変化・粒子収支に加え、
行列組立・分解が各1回だけであること、変動する電荷・正負電圧・境界応答、ゼロ解への復帰を確認した。
初回分解失敗、過大残差・非有限解、電位正本の破壊後の復元、close後の再分解、
時間ループの成功・失敗時の解放も検証した。既存の物理検証の許容値は緩めていない。

12条件は固定／適応 × D=0／0.5 m²/s × 正負DC／符号反転インパルス。
8×12格子、delta=1e-4 m、ギャップ6e-4 m、曲率半径2e-4 m、初期ピーク1e16 m⁻³、
背景供給1e11 m⁻³、初期中心z0=5e-4 m、dt=1e-11 s、t_end=1.1e-10 s、出力間隔2e-11 s。
DCは±500 V、インパルスは時刻ns・振幅 `(0,0),(.02,1),(.04,-.5),(.11,0)`、ピーク500 V。

改修開始時のソースを一時退避し、同じConda環境で別プロセスとして改修前後を実行した。
最終状態のNumPy配列・ne・phi、全保存フレームと刻み履歴CSVを比較した。
最大スケール誤差は状態5.72e-13、CSV数値列1.29e-12。
誤差は配列ごとの`max|new-old| / max(max|old|, 1e-300)`で、要素ごとの相対誤差ではない。
状態の許容値は1e-8、CSVは保存時の丸めを考慮して2e-6。文字列・非有限値の位置も確認した。
設定・ソルバー名の変更を含むrun_config.jsonは数値比較から除外した。
結果：[SCIPY_LU_NUMERICAL_RESULTS.json](SCIPY_LU_NUMERICAL_RESULTS.json)。

## 性能

Windows、既存CondaのPython 3.14.10、FiPy 4.0.3、NumPy 2.5.4、SciPy 1.18.1。
両方式ともSciPyを指定し、OpenMP/OpenBLAS/MKLは各1スレッド。
ウォームアップ1回を除き、各3回の中央値。Python起動・importを除く。
方式ごとの順次計測で、改修前→改修後の順。交互順の長時間測定ではない。

| 条件 | 改修前 | LU再利用 | 比率 |
|---|---:|---:|---:|
| Poisson初回：組立・分解・求解・残差検査 | 0.2352秒 | 0.2053秒 | 参考：各1回 |
| Poisson：同じ行列で右辺を20回更新 | 2.9583秒 | 0.1424秒 | 約20.77倍 |
| 初期化＋連成20更新、CSV出力なし | 6.6662秒 | 3.4466秒 | 約1.93倍 |

格子は200×150、delta=1e-5 m。連成は既定のDC 4 kV・密度・反応・D=0、dt=1 ps。
Poisson単体は乱数seed=72の気体内電荷源1e12 V/m²を使い、右辺源を毎回0.1%ずつ変更した。
反復求解の時間に初回分解を含めず、右辺生成・更新・求解・残差検査を含む。
単体Poissonの最大相対残差は両方式とも1.61e-15以下だった。
計算全体のサンプル範囲は、改修前6.622～6.803秒、再利用3.411～3.620秒。
全体の性能をPoisson単体の倍率から推定せず、実際の時間発展を測定した。
格子・計算時間・I/O条件が異なる場合の速度保証ではない。

生データ：[SCIPY_LU_REUSE_RESULTS.json](SCIPY_LU_REUSE_RESULTS.json)。
計測ツール：[benchmark_scipy_lu.py](tools/benchmark_scipy_lu.py)。

```powershell
# SciPy環境のプロジェクトルートから実行
$env:FIPY_SOLVERS = "scipy"
python -m unittest discover -s tests
python tools/benchmark_scipy_lu.py --steps 20 --report SCIPY_LU_BENCHMARK.json
```

ツールの`--project-root`へ別checkoutのルートを指定すると、同じ計測コードで改修前を測れる。
今回の比較用旧ソース・出力CSV・一時配列は検証後に削除し、結果JSONだけを残した。

## 実行環境の整理

旧`env_pyaxistreamer`にはmpi4py 4.1.2が残っており、MPI DLLがないためFiPyのimportが失敗した。
このSciPy構成には不要で、他のインストール済みパッケージからの依存もなかったため、
そのvenvのmpi4pyだけをアンインストールした。これで通常のFiPy import・テスト・CLIが成功した。
MPIを無効化するコードやDLL探索の回避策は本体へ追加していない。
PETSc用Conda環境は変更していない。

NumPy styleのdocstringとExamplesは維持した。
計算本体はdocstring・空行・コメント行を除いて875行→894行で、19行増加した。
全68定義にdocstring、67定義にExamplesがある。テスト・計測ツール・文書はこの本体行数に含めない。
環境別の検証と行数集計：[SCIPY_LU_VALIDATION.json](SCIPY_LU_VALIDATION.json)。
LU再利用による性能改善であり、共有メモリ並列化を追加した結果ではない。

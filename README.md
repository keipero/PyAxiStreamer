# ストリーマ放電シミュレーション

軸対称r-z有限体積モデルです。Poisson方程式と電子移流拡散をFiPyで解き、反応・物理係数はNumPyで評価します。
現在のバックエンドはSciPy、PoissonはLU求解です。Numbaは不要です。

## 実行

`pyaxistreamer-master`フォルダで、使用する仮想環境のPythonを実行してください。

```powershell
python -m pip install -r requirements.txt
$env:FIPY_SOLVERS = "scipy"
python -m streamer_drift_diffusion --steps=0 --output=outputdata_smoke
```

可視化用の追加依存は`python -m pip install ".[notebook]"`で導入できます。
`run_streamer.ipynb`は同じ仮想環境のカーネルを使います。旧版を読み込み済みならカーネルを再起動してください。

```powershell
# 標準200×150セル、DC 4 kV、5001更新
python -m streamer_drift_diffusion

# 小格子の確認計算
python -m streamer_drift_diffusion --nr=12 --nz=16 --delta=1e-4 --electrode-gap=6e-4 --steps=2 --no-output

# インパルス波形
python -m streamer_drift_diffusion --voltage-mode=impulse --impulse-peak=5000 --output=outputdata_impulse

# 適応刻み。出力先は既存CSVのないフォルダを指定する
python -m streamer_drift_diffusion --adaptive-timestep --t-end=5.001e-9 --output-interval=1e-10 --output=outputdata_adaptive
```

Pythonからは次のように実行できます。

```python
from streamer_drift_diffusion import SimulationConfig, run_simulation

config = SimulationConfig(adaptive_timestep=True, t_end=1e-11)
state = run_simulation(config, write_output=False, verbose=False)
```

## 数値モデル

- 電位・密度はセル中心、法線電場・電子速度は全セル面に置くスタガード配置です。
- `state.grid`が単一のFiPy円筒メッシュ、面の隣接セル、面積・体積、境界区分、距離を管理します。
- 密度・電位・中心電場は`(nr,nz)`、半径面は`(nr+1,nz)`、軸方向面は`(nr,nz+1)`です。
- 座標はm、密度はm⁻³、電位はV、換算電場Ex/Ey/absEはTd、速度はm/sです。
- 電極は`gas_mask=False`のセルによる階段状近似です。先端位置は`electrode_gap`、形状は`cathode_radius`で設定します。
- 軸は面積0・半径流束0。気体セル中心から電極・外端までは半セル距離を使います。
- 共通の面積・体積では2πを省略し、実粒子数・電荷の集計で一度だけ掛けます。

### Poisson求解

`rho=e/epsilon0*(n2p+o2p-om-o2m-ne)`を源として、`-laplacian(phi)=rho`から総電位を直接求めます。
FiPyの`DiffusionTerm`を気体間だけに適用し、電極・接地・外端のDirichlet条件は共有の面積・半セル距離から対角項と境界源を作ります。
金属セルは気体との結合を切った独立固定式です。大きな罰則係数は使いません。
外周・上端は双曲線電極の解析式に現在の印加電圧を掛けた電位、接地面は0 V、電極は印加電圧です。

`PoissonSolver`とFiPy方程式は初期化時に一度作ります。現在は`LinearLUSolver`を使用し、毎回行列を組み立て・分解します。
LUには初期推定が不要なため、初期値を0にして解き、微小な電荷変化を相対初期残差で省略しないようにしています。
入力・解の有限性と`max(abs(A*phi-b))/max(abs(b))`を検査し、失敗時はstateへ解を反映せず例外で停止します。
ゼロ右辺の分母には浮動小数点のtinyを使います。

```powershell
python -m streamer_drift_diffusion --poisson-tolerance=1e-10 --steps=0 --no-output
```

`poisson_tolerance`の既定値は1e-10です。旧SORの電位更新量の許容値とは意味が異なります。
`poisson_omega`・`poisson_progress_every`・`poisson_max_iterations`・`num_threads`と、CLIの`--poisson-omega`・`--num-threads`は廃止しました。
`p1`～`p5`・`unit_potential`・`lphi`・`pphi`、旧Poisson関数、Numba専用処理も削除しました。
PETScは未導入です。Windows用PETSc本体とpetsc4pyの導入・検証は後続作業です。

### 電子輸送と反応

`ElectronTransport`は同じメッシュ上で一次上流移流と拡散を陽的Eulerで同時更新します。
気体間のDは調和平均、電極・外部境界は気体側Dと半セル距離で吸収します。外からの流入・電子放出は含めません。
流出数は`[電極, 接地面, 上端＋外周]`の3区分です。輸送の損失項と同じ係数・旧密度から集計し、受理した更新だけ累積します。
単独で更新する場合は`transport = ElectronTransport(state, config.delta)`を一度作り、`losses = transport.step(state, dt)`を呼びます。

`electron_diffusion`は検証用の一定係数[m²/s]で、既定0です。正の値は利用者が指定します。物理係数の自動生成やPy3bolos連成は行っていません。
たとえば`--electron-diffusion=0.5`を指定できますが、空気の物理係数として推奨する値ではありません。

反応は電子衝突電離、解離付着、三体付着、背景電離を含みます。
光電離・再結合・脱離・イオン輸送は含みません。負密度のクリップは行いません。

## 時間刻みと出力時刻

電荷源 → 総電位 → 電場 → 輸送係数 → 反応係数 → 電子輸送 → 反応の順に計算します。

固定モードは従来仕様を維持しています。`max_step`は最後の番号なので更新回数は`max_step+1`です。
step 0のCSVは1回更新後で、電場・電位は密度更新前、密度は更新後の値です。最後の番号が保存間隔に合わなければ最終CSVは追加しません。
背景電離は毎更新の密度加算量で、固定モードではdtを掛けません。

適応モードは初期状態をt=0で保存し、指定の実時刻と最終時刻に電場・電位・密度を揃えます。
合算輸送CFL、反応、誘電緩和の制約を使い、失敗した試行の7密度を復元してdtを半減します。
背景加算量は`background_ionization*actual_dt/config.dt`です。FiPy輸送の現在値・旧値も各試行の開始時に同期します。
波形の折れ点を飛び越さず、出力時刻・終端への端数合わせは安全側に短縮します。
dt_minへの無条件な切上げは行いません。適応モードの終了条件は`t_end`で、`max_step`は使いません。

| 設定 | 内容・既定値 |
|---|---|
| `dt` | 固定幅・背景供給の基準幅、1e-12 s |
| `dt_max` | 適応上限、省略時dt |
| `dt_min` | 必要刻みの停止下限、1e-18 s |
| `dt_safety_factor` | 共通の安全係数、0.5 |
| `dt_cfl_number` | 合算流出対角へのCFL数、0.8 |
| `dt_chemistry_fraction` | 反応割合上限、0.1 |
| `dt_growth_factor` | 刻みの増加上限倍率、1.2 |
| `max_step_retries` | 試行の縮小・再試行上限、12 |
| `adaptive_max_steps` | 受理step上限、1000000 |
| `output_interval` | 保存間隔[s]、省略時write_every*dt |

CLIの条件は`python -m streamer_drift_diffusion --help`で確認できます。

## 保存形式と可視化

物理量は1値1行、rの添字が外側・zの添字が内側、`%e`の小数点以下6桁・LF改行です。
`Ex`は中心の半径方向、`Ey`は中心の軸方向の換算電場です。

| ファイル | 内容 |
|---|---|
| `rh.csv`・`zh.csv` | セル中心座標[m] |
| `gas_mask.csv` | 気体・電極の区分 |
| `ne*.csv` | 電子密度[m⁻³] |
| `Ex*.csv`・`Ey*.csv`・`absE*.csv` | 中心の換算電場[Td] |
| `phi*.csv` | 総電位[V] |
| `times.csv` | 適応モードの出力番号・step・実時刻 |
| `timestep_history.csv` | 適応モードの刻み・制約・流出収支 |
| `run_config.json` | 適応モードの設定・ライブラリ版・FiPy求解方式 |
| `failure.json`・`failure_state.npz` | 適応モードの停止診断。NPZは数値配列のみ |

保存済み結果は`notebooks/`の描画Notebook、`notebooks/animate_abs_ez.py`で参照できます。
入力フォルダを計算結果に合わせ、適応出力では`times.csv`も残してください。
従来の固定出力を読む場合は、描画側のdtを計算条件に合わせます。

## コード構成

| ファイル | 役割 |
|---|---|
| `grid.py` | 共有メッシュ・幾何・隣接関係・境界区分 |
| `simulation_types.py` | 計算設定と状態 |
| `initial_conditions.py` | 設定検査、初期密度とFiPyソルバーの生成 |
| `electrostatics.py` | FiPy Poisson、電荷源、換算電場、境界電位 |
| `transport.py` | FiPy電子輸送と吸収・流出数 |
| `transport_coefficients.py` | 面の移動度・速度、中心のD |
| `reactions.py` | 反応係数と密度更新 |
| `timestepping.py` | 安定刻み、密度検査、試行の受理・復元 |
| `simulation.py` | 固定／適応の時間発展 |
| `voltage_waveforms.py` | DC・インパルス入力、補間・単位換算 |
| `output.py` | CSV・メタデータ・停止診断 |

## 検証

```powershell
python -m unittest discover -s tests
```

`tests/poisson_reference.py`の独立した小格子直接解と、FiPyの組立・境界条件を比較します。
既知の放物線解、正負電圧、離散Gauss則、微小電荷変化、非有限入力・残差異常、
輸送の粒子収支、Gaussian格子収束、時間精度、試行の復元、CSV・CLI・波形も検証対象です。
RB-SORの更新順やスレッド数に固有のテストは、新しいPoissonの物理・数値検証へ置き換えています。

- [Poisson移行の結果・精度・性能](FIPY_POISSON_MIGRATION.md)
- [FiPy輸送移行と格子共通化の履歴](FIPY_MIGRATION_PLAN.md)
- [コード削減の設計案](CODE_REDUCTION_PROPOSAL.md)
- [初期のJulia・C比較の履歴](VALIDATION.md)

Julia・C比較スクリプトは旧モデルと外部実行環境を前提とする履歴検証用です。
現在のPoisson移行の確認は、独立直接解と物理条件を揃えた移行比較を基準にしてください。
数値互換性の確認は、長時間のストリーマ発展や物理モデルの妥当性の保証とは別です。

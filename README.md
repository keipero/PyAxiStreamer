# ストリーマシミュレーション Python版

隣接する `streamer_julia` の軸対称ストリーマ放電プログラムを、NumPyを用いてPythonへ翻訳しました。元のJuliaファイルは変更していません。このフォルダだけで計算できます。インパルス波形も同梱しています。

電子の移流拡散はFiPyの円筒有限体積法を使用します。Poisson求解はNumbaによるRB-SORです。
[FiPy移行方針と検証結果](FIPY_MIGRATION_PLAN.md)に、境界条件・時間離散化・検証範囲を記載しています。

## 実行方法

Python 3.10以上を使用します。以下は、この `pyaxistreamer-master` フォルダで実行してください。

```powershell
cd pyaxistreamer-master
python -m pip install -r requirements-fast.txt
$env:FIPY_SOLVERS = "scipy"
python -m streamer_drift_diffusion --steps=0 --output=outputdata_smoke
```

NumbaはPoisson反復・基礎電位を高速化します。Poissonソルバーは並列RB-SORのみで、選択設定はありません。初回だけコンパイル時間がかかります。Numbaを使用しない場合は `requirements.txt` だけをインストールできますが、標準格子での長時間計算にはNumbaの利用を推奨します。依存ライブラリは実際に使用するPython環境へインストールしてください。

FiPyとSciPyは基本依存です。輸送は陽的Euler・一次上流移流・面係数の調和平均による拡散で、既存の合算CFL制限を使います。電極・外端の半セル吸収を損失項として与え、流出粒子数を同じ項から集計します。FiPy格子と方程式は各計算で一度作り、毎stepの密度・速度・Dを更新します。陰解法やPETScはまだ導入していません。

旧`nele_flow()`と`fluxes.py`は削除しました。単独の輸送更新は`transport = ElectronTransport(state, config.delta)`を一度作り、`losses = transport.step(state, dt)`で行います。速度は全セル面の形状が必須です。

### 並列RB-SORとスレッド数

`electrostatics.py::_poisson_sweeps()` は、`(i+j)%2` が0の赤セル、1の黒セルの順に更新します。各色の行をNumbaの `prange` で分担し、色の間で同期します。赤・黒の両方を更新して1反復です。Numba未導入時、または `STREAMER_DISABLE_NUMBA=1` の場合も、同じ更新式を `range` で逐次実行します。

計算条件と既定値は `SimulationConfig` に集約します。スレッド数も `num_threads` で指定し、`run_simulation()` が時間発展前にNumbaへ適用します。外部で `set_num_threads()` を呼ぶ必要はありません。

```python
config = SimulationConfig(num_threads=4)
state = run_simulation(config)
```

`num_threads` の既定値は2で、1以上の整数を指定します。Numbaの利用上限を超えると、出力ファイルを作る前にエラーとします。環境変数 `NUMBA_NUM_THREADS` はNumba側の利用上限であり、実際に要求する数はConfigで指定します。Numba無効時は同じConfigで逐次実行します。適応出力の `run_config.json` には要求値に加え、`num_threads_used`（実際のスレッド数）と `numba_available` も保存します。16スレッドが最速とは限らないため、対象格子で測定してください。

CLIからは `--num-threads=4` で指定できます。CLIの計算条件の既定値も `SimulationConfig()` から取得し、別の既定値は持ちません。

収束判定は、1反復の最大電位更新量を、それまでの最大絶対電位（初期尺度 `1e-10 V`）で割った値です。方程式残差による停止判定ではありません。旧SORとは更新順序と尺度の集計時点が変わるため、同じ許容値でも反復回数や最終値は完全一致しません。`poisson_omega=1.5` と許容値の既定値は維持しています。非有限電位は固定・適応モードの両方でエラーとします。

```powershell
# 標準: 200×150セル、DC 4 kV、step 0～5000
python -m streamer_drift_diffusion

# 同梱インパルス: 最大5 kV
python -m streamer_drift_diffusion --voltage-mode=impulse --impulse-peak=5000 --output=outputdata_impulse

# パルス幅を2倍にして、終了stepも延長
python -m streamer_drift_diffusion --voltage-mode=impulse --impulse-time-scale=2 --steps=30000 --output=outputdata_wide

# CSVを出さず計算
python -m streamer_drift_diffusion --steps=10 --no-output --quiet

python -m streamer_drift_diffusion --help
python -m unittest discover -s tests -v
```

既定のCSV出力先はこのプロジェクトの `outputdata` です。`--output` に相対パスを指定すると現在の作業ディレクトリを基準にします。同名のCSVは上書きされます。`pip install -e '.[fast]'` でパッケージとしてインストールすることもできます。

## 電極周囲を含む計算領域

計算条件はすべて `SimulationConfig` に指定します。`electrode_gap`（既定1 mm）は接地平面から電極先端までの距離で、領域上端 `nz*delta` とは独立です。標準は `nr=200, nz=150, delta=10e-6`、領域は半径2 mm・高さ1.5 mmです。

```python
config = SimulationConfig(nr=400, nz=300, delta=5e-6,
                          electrode_gap=1e-3, num_threads=2)
```

- 電極は `z²/electrode_gap² - r²/(cathode_radius*electrode_gap) >= 1` のセル。`gas_mask` は気体=True・電極=Falseです。電極面はセル境界による階段状近似で、曲面を正確に切るcut-cell法ではありません。
- Poissonは円筒有限体積法。気体中心から電極・領域端の固定電位面までを半セル幅とします。最外セルも完全な円環体積に修正しています。
- 電極面は印加電位、接地平面は0 V。上端・外周の気体境界には同じ双曲線電極の解析電位を与えます。空間電荷補正は電極・接地・上端・外周で0、軸は対称条件です。外周を接地電極とみなす条件ではありませんが、有限領域による影響は領域拡大試験で確認してください。
- 初期化時に1 V印加の離散ラプラス解 `unit_potential` を求め、各時刻では印加電圧倍します。単なる解析電位のサンプリングから変わるため、旧出力との完全一致は要件としません。
- 電極内の全密度・反応・背景供給を0にします。初期分布・CFL・誘電緩和・反応の制約も気体を対象とします。
- 電子輸送では電極・接地面への吸収、上端・外周への流出を許し、外からの流入と電子放出は含めません。外端も境界電位から法線電場・速度を求めます。`ElectronTransport.step()`は`[電極, 接地面, 上端＋外周]`の流出電子数を返し、受理更新だけを`electron_losses`へ累積します。
- 固定・適応出力に `gas_mask.csv` を追加。適応履歴には3種類の累積流出電子数を記録します。中心の `ex/ey/abs_e` は金属内0。金属と気体の間の法線電場は面配列へ保持します。

`notebooks/plot_electrode_grid.ipynb` はPythonのConfigから実際の電極マスクと格子を表示します。まずこの図を確認してから計算してください。階段状境界の近傍電場には格子依存性が残ります。

## 中心・面のスタガード配置

物理量の位置を次のように分けます。等間隔r-z格子、軸対称体積、並列RB-SORは引き継ぎ、配置を選択する新しいConfig項目は設けていません。

| 配列 | 配置・形状 | 単位・用途 |
|---|---|---|
| `phi`、各粒子密度、`k1`～`k3`、`de` | 中心 `(nr,nz)` | 電位・反応・拡散係数 |
| `ex`、`ey`、`abs_e` | 中心 `(nr,nz)` | Td。反応とCSV出力 |
| `er_face`、`vex`、`mobility_r` | 全半径面 `(nr+1,nz)` | Td、m/s、m²/(V s) |
| `ez_face`、`vey`、`mobility_z` | 全軸方向面 `(nr,nz+1)` | Td、m/s、m²/(V s) |

`er_face[i,j]`は `(r=i*delta, z=(j+1/2)*delta)`、`ez_face[i,j]`は `(r=(i+1/2)*delta, z=j*delta)` の値です。中心配列は両方向とも半セルずれた位置です。同じ添字の面成分を直接hypotへ渡しません。

軸上の条件は`er_face[0,:]=0`です。中心の`ex[0,:]`は軸から半セル離れたr=delta/2の値なので、一般には0になりません。

計算は次の順です。

1. `calculate_potential()`が全電位を更新し、実際の印加電圧を返します。
2. `calculate_electric_field(..., voltage=voltage)`が中心電位と境界電位から全面の法線電場を計算します。内部気体面は1セル幅、金属・接地・外端の固定電位面は半セル幅です。軸の法線電場は0です。
3. 中心成分を `ex=(er_face[:-1]+er_face[1:])/2`、`ey=(ez_face[:,:-1]+ez_face[:,1:])/2` で再構成し、中心強度を `abs_e=hypot(ex,ey)` とします。金属内は0です。
4. `calculate_transport_coefficients()`が接線成分を中心から各面へ補間し、法線成分と同じ位置で強度・移動度・速度を計算します。内部気体面は平均、金属・接地面の接線成分は0、軸は偶対称。上端・外周の接線成分は気体2点から線形外挿し、隣が金属なら1点値を使います。
5. `calculate_reaction_coefficients()`は中心強度に現行の速度式を代入して速度の大きさを求め、k1～k3を計算します。面速度の配列は使いません。

外端の法線電場はPoissonと同じ解析境界電位を使い、内側面を複製しません。単位境界電位は初期化時に`radial_boundary`・`upper_boundary`へ保存します。Poissonの離散作用と、法線電場の有限体積発散の一致をテストしています。CFLは全セル面の速度、誘電緩和は同じ面の移動度と隣接する最大電子密度から評価します。

**API・CSVの変更点：** `state.ex/ey`はセル中心、`state.vex/vey`は全面の形状です。下側面だけの速度配列は受け付けません。CSVのEx・Ey・absEは中心値として全て`nr*nz`行を維持し、従来の中心座標・描画コードで読み込めます。適応出力のmodelは`staggered-v1`、設定JSONへ配置と単位、FiPyの版と求解方式も記録します。

Notebookは**カーネルを再起動して最初から実行**してください。物理条件の既定値は変えていません。面移動度・反応係数・境界流束の評価が変わるため、旧出力との数値一致は求めません。新しい出力先で同じ実時刻の結果を比較してください。

検証は自動テスト83件と、既知電位に対する中心電場の格子収束、正負DC・インパルスの短い連成試験です。詳細は[ルートREADMEのfeature/staggered](../README.md#8-featurestaggered中心面の電場と係数評価の整合)に記録しています。滑らかな内部領域での二次精度を確認したもので、階段状電極や全境界付近の二次精度を保証する結果ではありません。

## 電子拡散

電子輸送の面フラックスを `n_e*v_e - D_e*grad(n_e)` として、ドリフトと拡散を同じ旧密度から同時に更新します。設定は `SimulationConfig.electron_diffusion`、単位はm²/sです。

```python
# 任意の一定係数を用いた数値検証の例。空気の物理係数として推奨する値ではありません。
config = SimulationConfig(electron_diffusion=1.0, adaptive_timestep=True, t_end=1e-11)
```

既定値と実行Notebookの指定は **0（拡散なし）** です。本計算用の係数が未確定なため、正の値は利用者が指定します。CLIは `--electron-diffusion` を使います。今回Py3bolosの実連成や物理係数の自動生成は行っていません。

- `calculate_transport_coefficients(state, config)` が `state.de` へ係数を格納します。現在は気体内一定・電極内0。将来Py3bolosからセルごとの係数を受け取る箇所はここです。輸送処理自体は係数配列に対応しています。
- 気体間の面係数は調和平均です。電極・接地・上端・外周は密度0の吸収境界とし、気体側のDと半セル距離を使います。軸の拡散流束は0です。等方拡散を対象とします。
- 粒子数を保存する同一面フラックスを隣接セルで共有し、境界へ出た電子は既存の3種類の流出数へ加えます。負密度のクリップは行いません。
- `transport_limit()` はドリフトと拡散の損失対角係数を合算して制限します。個別上限の最小だけでは同時更新の非負性を保証できません。
- `dt_cfl_s` は合算輸送の上限（`dt_cfl_number`を含む）、`dt_diffusion_s` は純拡散の上限です。合算CFLの方が厳しいため、拡散がある場合も支配条件が`cfl`と記録され得ます。さらに共通の安全係数を適用します。
- 正のDを指定した固定モードは、dtが合算CFLを超えると更新前に停止します。適応モードは刻みを縮小し、拒否した試行の密度・流出収支を復元します。
- 拡散導入時のD=0のバイト一致は旧輸送実装の検証結果です。現在のFiPy版は、D=0／正のDの12条件で旧実装との最大スケール誤差1.49e-13を確認しています。丸め順序が変わるためバイト一致は要求しません。

検証用一定係数によるGaussianの広がりと時間精度は確認済みです。物理係数の妥当性、長時間のストリーマ発展、縞状分布の改善は別途評価します。一次上流法に由来する数値拡散も残ります。

## 適応的時間刻み

`--adaptive-timestep` と終了時刻 `--t-end` [s] を指定して有効化します。既定は従来の固定刻みです。

```powershell
# 5.001 nsまで、最大dt=1 ps、0.1 ns間隔で出力
python -m streamer_drift_diffusion --adaptive-timestep --t-end=5.001e-9 --dt-max=1e-12 --output-interval=1e-10 --output=outputdata_adaptive
```

出力先にはCSVのない新しいフォルダを指定します。初期状態を出力番号0へ保存し、指定間隔と最終時刻の状態も保存します。CSV番号は**出力番号**であり、受理step番号や時刻ではありません。

| 設定 / CLI | 意味・既定値 |
|---|---|
| `adaptive_timestep` / `--adaptive-timestep` | 適応モード。既定False |
| `t_end` / `--t-end` | 必須の終了時刻 [s]。0なら初期状態のみ |
| `dt` / `--dt` | 固定刻み幅・背景供給の基準時間。既定1e-12 s |
| `dt_max` / `--dt-max` | 適応刻みの上限。省略時はdt |
| `dt_min` / `--dt-min` | 安定刻みが下回ったら停止。既定1e-18 s |
| `dt_safety_factor` / `--dt-safety-factor` | 各安定条件へ掛ける安全係数。既定0.5 |
| `dt_cfl_number` / `--dt-cfl-number` | 流出率に対するCFL数。既定0.8 |
| `dt_chemistry_fraction` / `--dt-chemistry-fraction` | 反応の生成・消費割合の制約。既定0.1 |
| `dt_growth_factor` / `--dt-growth-factor` | 刻みの増加倍率の上限。既定1.2 |
| `max_step_retries` / `--max-step-retries` | 縮小・再試行の上限。既定12 |
| `adaptive_max_steps` / `--adaptive-max-steps` | 受理step数の停止上限。既定1000000 |
| `output_interval` / `--output-interval` | 出力間隔 [s]。省略時はwrite_every*dt |
| `poisson_max_iterations`（Python設定） | 初期ラプラス解と固定・適応のPoisson総反復上限。既定200000 |

適応モードでは `max_step` / `--steps` を終了条件に使いません。受理step上限に達した場合は、途中の状態を成功として返さず例外で停止します。dt_minへの無条件な切り上げは行いません。出力・波形の折れ点・終端への最後の端数合わせでは、安全側の短縮なのでdt_min未満も許します。

時刻の差がその時刻付近の8 ULP（浮動小数点の間隔）以内なら、丸め誤差の範囲で同じイベントとして扱います。終了→出力→波形の順で代表時刻を選び、更新後の時計をその時刻へ揃えます。丸め誤差分の密度更新・step加算は追加せず、同じ時刻の出力も1回にまとめます。受理dtを安定条件より大きくする処理ではありません。波形点への通常の0.5 ps等の短縮や、丸め誤差より大きい実際の短い終端更新は維持します。

### 適応モードの呼び出し順

1. `simulation.py::run_simulation()` → `initial_conditions.py::initialize_simulation()`で初期化し、`simulation.py::_run_adaptive()`へ分岐します。
2. `simulation.py::_prepare_adaptive()` → `prepare_step()`で電荷源・電場・速度・反応係数を準備します。
3. `timestepping.py::timestep_limits()` → `transport_limit()`、`dielectric_limit()`、`chemistry_limit()`、`diffusion_limit()`で上限を評価し、安全係数・dt_max・増加上限・次のイベント時刻で刻みを制限します。
4. `timestepping.py::try_density_step()` → `transport.py::ElectronTransport.step()` → `reactions.py::apply_reactions(..., dt=actual_dt)`で試行します。輸送後・反応後の密度と制約を検査し、失敗時は7種の密度を復元して刻みを半分にします。FiPyの現在値・旧値も次の試行開始時に復元済みstateから同期します。
5. 受理したdtだけ時刻を進め、`_prepare_adaptive()`で電場を同じ時刻へ更新します。この値を次の刻み選択にも再利用します。
6. `output.py::AdaptiveOutput.record_step()`で全受理刻み、`save_frame()`で指定時刻・最終時刻の状態を保存します。

背景加算量は `background_ionization * actual_dt / config.dt` とし、単位時間あたりの供給率を保ちます。固定モードでは従来の毎更新加算です。移流・反応の順序、一次精度の陽解法、速度・反応式は維持しています。

現行の移動度式は換算電場F→0で発散します。正のFはそのまま評価し、誘電緩和の判定では孤立したF=0面に有効面の有限な最大移動度を使用します。全面の移動度0ではドリフト0なので誘電緩和制約を無限大とします。これは0面の評価規約で、低電場極限の厳密な安定性保証ではありません。安全係数を下げた収束確認も必要です。

### 適応出力の時刻と診断

| ファイル | 内容 |
|---|---|
| `run_config.json` | 計算設定、NumPy・FiPy・SciPy版、輸送方式・ソルバー、出力番号・時刻の規約 |
| `times.csv` | frame、受理step、実時刻time_s、直前dt_s、電場時刻field_time_s |
| `timestep_history.csv` | 全受理stepのdt、合算輸送CFL・誘電緩和・反応・純拡散の制約、支配条件、拒否回数、密度・電場範囲、電子総数・正味電荷・3種の累積流出電子数 |
| `failure.json` / `failure_state.npz` | 刻み不足・再試行失敗・非有限値・Poisson未収束などの停止診断 |

適応出力の電位・電場・粒子密度は同じ実時刻です。`output.py::load_output_times()` は `times.csv` を優先し、従来出力に対応表がなければ `番号 * DT_SECONDS` を使います。軸方向分布・ストリーク・アニメーションのNotebookとGIFスクリプトも対応しています。GIFの再生FPSは表示用で、物理時刻はラベルで確認します。

実行Notebookでは設定セルの `config = SimulationConfig(...)` に全計算条件をまとめます。`adaptive_timestep=True`、`t_end`、`output_interval`、`dt_max`、`num_threads`などをここで指定します。現在の出力先は固定計算と別の `outputdata_adaptive_staggered/` です。再実行時は新しい出力先へ変更してください。

Notebookの `config.adaptive_max_steps` は適応モードの受理更新数の上限で、現在の指定値は20000です。`max_step` は固定モード専用です。上限へ到達すると未完了として例外で停止し、出力有効時は`failure.json`と`failure_state.npz`を保存します。これは実時間のタイムアウトではなく、1回のPoisson求解には別の`poisson_max_iterations`が適用されます。Notebookの設定変更は既に実行中のセルには反映されないため、中断後、設定セルから再実行してください。ライブラリ・CLIの`adaptive_max_steps`の既定値1000000は変更していません。

## Pythonから実行

```python
from streamer_drift_diffusion import SimulationConfig, run_simulation

config = SimulationConfig(
    max_step=100,
    voltage_mode="impulse",
    impulse_peak_voltage=5000.0,
    impulse_time_scale=1.0,
)
state = run_simulation(config, output_dir="outputdata_python")
print(state.ne.shape, state.ne.max())
```

`run_streamer.ipynb` はPythonカーネル用の実行・描画Notebookです。計算条件は設定セルの `MAX_STEP`、`VOLTAGE_MODE`、`IMPULSE_PEAK_VOLTAGE` などで指定します。保存されている設定はCLIの既定値と異なるため、実行前に確認してください。短い確認計算なら `MAX_STEP = 100`、`VOLTAGE_MODE = 'dc'`、標準の更新回数なら `MAX_STEP = 5000` を指定します。Notebookには `matplotlib` とJupyter環境が必要です（`python -m pip install -e '.[fast,notebook]'`）。

## フォルダ構成

この `pyaxistreamer-master/` がPython版のプロジェクトルートです。計算本体は `streamer_drift_diffusion/`、実行入口は `python -m streamer_drift_diffusion` とルートのNotebook、保存済み結果の描画は `notebooks/` に分かれています。

```text
pyaxistreamer-master/
├─ README.md                         # 実行方法・コード構成（この文書）
├─ pyproject.toml                    # パッケージ情報・依存関係・CLI登録
├─ requirements.txt                  # 基本依存（NumPy・FiPy・SciPy）
├─ requirements-fast.txt             # 基本依存 + Numba
├─ run_streamer.ipynb                # 条件設定 → 計算 → 結果描画
├─ streamer_drift_diffusion/         # 計算本体
│  ├─ __init__.py                    # Python向け公開API
│  ├─ __main__.py                    # CLI引数の解析・実行
│  ├─ simulation.py                  # 時間発展の統括
│  ├─ timestepping.py                # 適応刻みの制約・受理・再試行
│  ├─ simulation_types.py            # 計算設定・状態配列の定義
│  ├─ grid.py                        # r-z格子・有限体積の幾何係数
│  ├─ initial_conditions.py          # 設定検査・配列確保・初期密度
│  ├─ electrostatics.py              # 電荷源・電位・換算電場
│  ├─ voltage_waveforms.py           # DC／インパルス電圧
│  ├─ transport_coefficients.py      # 電子ドリフト速度
│  ├─ transport.py                   # FiPy電子移流拡散・吸収・流出収支
│  ├─ reactions.py                   # 反応係数・各粒子密度の更新
│  ├─ output.py                      # C／Julia互換のCSV保存
│  ├─ _acceleration.py               # Numbaの有効化・代替処理
│  └─ data/
│     └─ impulse_waveform_V50_tr5_tw6.txt  # 入力波形（時間ns・電圧kV）
├─ notebooks/                       # 保存済みCSVの描画・電極形状の確認
├─ tests/                           # 単体テスト・Julia／C比較スクリプト
├─ VALIDATION.md                    # 検証条件・結果の概要
├─ validation_c_5000_20261027/       # C比較のレポート・表・図・検証用ソース
├─ outputdata*/                     # 計算結果CSV（実行時に生成）
└─ figures/                         # 描画結果の保存先（描画時に生成）
```

`outputdata*/` は出力フォルダの総称です。CLIの既定値は `outputdata/`、実行Notebookでは `OUTPUT_DIR` で指定し、現在の保存設定は `outputdata_notebook/` です。`outputdata_smoke/` などは実行条件ごとに付けた名前で、計算本体の一部ではありません。

## 各マスターコードの役割

ここでは、計算を構成する主要なPythonファイルを「マスターコード」として説明します。全体の処理順を管理する中心は [simulation.py](streamer_drift_diffusion/simulation.py) です。

### 実行入口・設定・初期化

| ファイル | 主なクラス・関数 | 役割と連携 |
|---|---|---|
| [__main__.py](streamer_drift_diffusion/__main__.py) | `main()` | `--steps`、`--voltage-mode` などを解析して `SimulationConfig` を作り、`run_simulation()` を呼びます。`python -m streamer_drift_diffusion` と、パッケージインストール後の `streamer-python` も同じ入口です。 |
| [__init__.py](streamer_drift_diffusion/__init__.py) | 公開APIの再公開 | `from streamer_drift_diffusion import ...` で設定・状態・実行関数などを読み込めるようにします。 |
| [simulation_types.py](streamer_drift_diffusion/simulation_types.py) | `SimulationConfig`、`SimulationState`、`molecular_density()`、`gap_length()` | 格子・電圧・時間刻み・反応条件などの設定と、全状態配列を定義します。分子数密度とギャップ長の計算もここです。 |
| [grid.py](streamer_drift_diffusion/grid.py) | `grid_axis()`、`initialize_grid()` | 格子座標、セル中心、セル幅、軸対称有限体積法の体積・面積係数を生成します。初期化と電子移流で使います。 |
| [initial_conditions.py](streamer_drift_diffusion/initial_conditions.py) | `validate_config()`、`initialize_simulation()` | 設定検査後に格子と状態配列を確保し、Poisson係数と単位電位を準備します。気体内に中性粒子とGaussian状の電子・正イオンを設定し、電荷中性の初期状態を返します。 |

### 時間発展・物理計算・出力

| ファイル | 主な関数 | 役割と連携 |
|---|---|---|
| [simulation.py](streamer_drift_diffusion/simulation.py) | `run_simulation()`、`prepare_step()`、`advance_step()`、`_run_adaptive()` | 初期化、波形読み込み、固定／適応刻みの実行・出力を統括します。`advance_step()`は固定幅の1回更新、`prepare_step()`は両モード共通の電場・係数準備です。 |
| [timestepping.py](streamer_drift_diffusion/timestepping.py) | `TimestepLimits`、`TimestepError`、`timestep_limits()`、`try_density_step()` | 適応刻みのCFL・誘電緩和・反応制約、密度の検査・復元・再試行を管理します。 |
| [voltage_waveforms.py](streamer_drift_diffusion/voltage_waveforms.py) | `load_voltage_waveform()`、`normalized_voltage()`、`applied_voltage()` | 波形をSI単位に換算し、振幅規格化・線形補間・時間倍率を適用して、その時刻の印加電圧を返します。DCでは一定電圧を返します。 |
| [electrostatics.py](streamer_drift_diffusion/electrostatics.py) | `discretization()`、`calculate_charge_density()`、`calculate_potential()`、`calculate_electric_field()` | 初期化時にPoisson係数 `p1`～`p5` を生成します。各stepでは密度から電荷源を作り、`initialize_potential()` で準備した離散単位電位を電圧倍した `lphi` と、`poisson_calc()` の並列RB-SOR解 `pphi` を足して `phi` を求めます。その差分から換算電場を計算します。 |
| [transport_coefficients.py](streamer_drift_diffusion/transport_coefficients.py) | `electron_drift()`、`electron_mobility()`、`_tangential_faces()`、`calculate_transport_coefficients(state, config)` | 各面で電場強度・移動度・速度を評価し、中心の拡散係数 `de` も準備します。面速度は輸送、面移動度は誘電緩和へ渡します。 |
| [transport.py](streamer_drift_diffusion/transport.py) | `ElectronTransport`、`transport_rates()`、`diffusion_rates()` | FiPyの円筒格子・陽的上流／拡散項で`ne`を更新します。半セル吸収の損失項から3区分の流出数を返します。面の率はCFL評価にも使います。 |
| [reactions.py](streamer_drift_diffusion/reactions.py) | `calculate_reaction_coefficients()`、`apply_reactions()` | セル中心の電場強度から現行の速度式と反応係数式を評価して `k1`～`k3` を求めます。輸送後の密度へ反応と背景電離を適用し、電子・中性粒子・正負イオンを更新します。 |
| [output.py](streamer_drift_diffusion/output.py) | `write_c_vector()`、`write_c_matrix()`、`write_step_output()`、`AdaptiveOutput`、`load_output_times()` | 座標と物理量をC／Julia互換CSVへ保存します。適応出力の設定・実時刻・刻み履歴・停止診断と、描画用の時刻読み込みも担当します。 |
| [_acceleration.py](streamer_drift_diffusion/_acceleration.py) | `njit`、`prange`、`NUMBA_AVAILABLE` | Numbaが使える場合にPoisson反復と基礎電位計算をJITコンパイルします。Poissonでは行を並列処理します。未導入時、または環境変数 `STREAMER_DISABLE_NUMBA=1` の場合はPython関数と `range` で実行します。 |

### 計算の呼び出し順

以下の詳細ツリーは**固定刻みの互換モード**です。適応モードは上記の「適応モードの呼び出し順」を参照してください。

以下では `ファイル名.py::関数名()` を「そのファイルに定義された関数」、`ファイル名.py::クラス名(...)` を「そのクラスのインスタンス生成」として表します。この表記は説明用で、Pythonの呼び出し構文ではありません。Notebookを除き、ファイルは全て `streamer_drift_diffusion/` 内にあります。

#### 1. 実行入口から時間発展の開始まで

| 実行方法 | 呼び出されるファイル・関数・クラスの順序 |
|---|---|
| `python -m streamer_drift_diffusion` | `__main__.py::main()` → `simulation_types.py::SimulationConfig(...)` → `simulation.py::run_simulation(config, ...)` |
| `python -m streamer_drift_diffusion` | `__main__.py::main()` → `simulation_types.py::SimulationConfig(...)` → `simulation.py::run_simulation(config, ...)` |
| インストール後の `streamer-python` | `pyproject.toml` のCLI登録 → `__main__.py::main()` → `simulation_types.py::SimulationConfig(...)` → `simulation.py::run_simulation(config, ...)` |
| `run_streamer.ipynb` / Pythonコード | `__init__.py` 経由で公開APIを読み込み → `simulation_types.py::SimulationConfig(...)` → `simulation.py::run_simulation(config, ...)` |

CLIの `__main__.py::main()` は、全計算条件の既定値を `simulation_types.py::SimulationConfig()` から取得し、引数解析後に指定値を持つ `SimulationConfig(...)` を生成します。Pythonから `run_simulation()` に `config` を渡さなかった場合は、`simulation.py::run_simulation()` 内で既定の `SimulationConfig()` を生成します。

`SimulationConfig` は計算設定、`SimulationState` は更新対象の配列、`VoltageWaveform` は読み込んだ波形データを保持するクラスです。時間発展そのものを実行するのは `simulation.py` の関数です。

#### 2. 初期化・波形読み込み・座標出力

```text
simulation.py::run_simulation(config, ...)
├─ initial_conditions.py::initialize_simulation(config)
│  ├─ initial_conditions.py::validate_config(config)
│  │  └─ voltage_waveforms.py::validate_impulse_parameters(config)
│  │     └─ impulseモードの場合のみ、振幅・時間倍率を検査
│  ├─ grid.py::initialize_grid(config)
│  │  ├─ grid.py::grid_axis(config.nr, config.delta)  # 半径方向
│  │  ├─ grid.py::grid_axis(config.nz, config.delta)  # 軸方向
│  │  └─ 体積・面積係数を計算し、格子配列の辞書を返す
│  ├─ SimulationStateのフィールドを調べ、残りの状態配列をゼロで確保
│  ├─ simulation_types.py::SimulationState(**arrays)  # stateを生成
│  ├─ electrostatics.py::initialize_potential(state, config)
│  │  ├─ electrostatics.py::discretization(...)  # 境界源と係数
│  │  ├─ electrostatics.py::_base_potential(...)  # 解析解を初期推定に使う
│  │  └─ electrostatics.py::poisson_calc(...)  # 離散単位電位を1回求解
│  │  └─ state.p1～state.p5へPoisson離散化係数を設定
│  ├─ Gaussian因子をセル中心で計算
│  ├─ simulation_types.py::molecular_density(config)
│  ├─ state.n2・o2・n2p・o2p・neへ初期密度を設定
│  ├─ transport_coefficients.py::calculate_transport_coefficients(state, config)
│  │  └─ 初期速度とstate.deを準備（電場0でも拡散係数は設定する）
│  └─ stateを返す（この段階では電位・電場は0）
├─ voltage_waveforms.py::load_voltage_waveform(config.voltage_waveform_file)
│  ├─ impulseモードの場合のみ、波形ファイルを読み込む
│  └─ voltage_waveforms.py::VoltageWaveform(times, voltages, peak)
│     └─ 波形オブジェクトを生成し、以後のstepで再利用
├─ write_output=Trueの場合、出力フォルダを作成
│  ├─ output.py::write_c_vector(..., state.rh)
│  │  └─ output.py::write_c_values(...)  # rh.csv
│  └─ output.py::write_c_vector(..., state.zh)
│     └─ output.py::write_c_values(...)  # zh.csv
└─ time = 0.0として、step = 0, ..., config.max_stepのループへ進む
```

`initial_conditions.py::initialize_simulation()` を単独で呼び、`config=None` とした場合も、この関数内で `simulation_types.py::SimulationConfig()` を生成します。初期状態だけを確認する場合は、この関数が返す `SimulationState` を使用します。

#### 3. 各step内の物理計算

`simulation.py::run_simulation()` が、各stepで以下の順に `simulation.py::advance_step()` を呼びます。ツリー内の同じ深さの処理は、上から順に実行されます。最初の5関数は `simulation.py::prepare_step()` 内で呼ばれ、ツリーではその内部を展開しています。

```text
simulation.py::advance_step(state, config, time, waveform=waveform, ...)
├─ electrostatics.py::calculate_charge_density(state, config)
│  └─ 電子・正負イオン密度から、体積電荷密度/ε₀をstate.rhoへ格納
├─ electrostatics.py::calculate_potential(state, config, time=time, ...)
│  ├─ voltage_waveforms.py::applied_voltage(config, time, waveform=waveform)
│  │  ├─ dcの場合：config.voltageを返す
│  │  └─ impulseの場合：
│  │     ├─ voltage_waveforms.py::validate_impulse_parameters(config)
│  │     ├─ voltage_waveforms.py::normalized_voltage(table, time / config.impulse_time_scale)
│  │     │  └─ 線形補間・規格化した波形の値を返す
│  │     └─ その値にconfig.impulse_peak_voltageを掛け、印加電圧を返す
│  ├─ state.lphi = 印加電圧 * state.unit_potential
│  ├─ electrostatics.py::poisson_calc(state.pphi, state.rho, ...)
│  │  └─ electrostatics.py::_poisson_sweeps(...)
│  │     └─ 並列RB-SOR反復を収束まで実行し、空間電荷電位state.pphiを更新
│  └─ state.phi = state.pphi + state.lphiとして全電位を合成し、印加電圧を返す
├─ electrostatics.py::calculate_electric_field(state, config, voltage=voltage)
│  ├─ simulation_types.py::molecular_density(config)
│  ├─ 全電位と境界電位から全面のstate.er_face・ez_faceを更新 [Td]
│  └─ 対向面の平均から中心ex・eyを再構成し、中心abs_eを計算 [Td]
├─ transport_coefficients.py::calculate_transport_coefficients(state, config)
│  ├─ _tangential_faces()で接線成分を補間し、各面で強度を評価
│  ├─ electron_mobility()で面のmobility_r・mobility_zを計算
│  └─ 全面の電子速度state.vex・vey [m/s]と中心state.de [m²/s]を更新
├─ reactions.py::calculate_reaction_coefficients(state)
│  └─ 中心abs_eから速度の大きさと反応係数state.k1～k3を更新
├─ transport.py::ElectronTransport.step(state, config.dt)
│  └─ FiPyの気体間面項と半セル吸収項で電子密度state.neを移流拡散更新
└─ reactions.py::apply_reactions(state, config)
   └─ 移流後の電子密度を使い、反応・背景電離で各粒子密度を更新
```

このツリーは `run_simulation()` の通常経路です。`advance_step()` や `calculate_potential()` を単独で呼び、impulseモードで `waveform=None` を渡した場合は、`voltage_waveforms.py::applied_voltage()` 内で `load_voltage_waveform()` → `VoltageWaveform(...)` も呼ばれます。

`transport_coefficients.py::electron_drift()`は、この時間発展経路からは呼ばれず、配列版の`calculate_transport_coefficients()`を使います。`_acceleration.py`の`njit`は`electrostatics.py`の関数定義時に適用されるデコレータで、各stepの独立した計算処理ではありません。

#### 4. step出力・時刻更新・最終状態の返却

`simulation.py::advance_step()` が戻ると、`simulation.py::run_simulation()` 内で以下を実行します。

```text
各stepのadvance_step()終了後
├─ step % config.write_every == 0の場合：
│  ├─ write_output=Trueならoutput.py::write_step_output(output_dir, step, state)
│  │  └─ output.py::write_c_matrix(path, values)  # 5種類の配列に対して実行
│  │     └─ output.py::write_c_values(path, values)
│  │        └─ ne / absE / Ex / Ey / phi の各CSVを保存
│  └─ verbose=Trueならstepと時刻を表示
│     └─ impulseモードではvoltage_waveforms.py::applied_voltage(...)も呼び、電圧を表示
├─ time += config.dtとして次のstepへ進む
└─ 最終step後、simulation.py::run_simulation()がstateを返す
```

`SimulationState` は初期化時に一度生成し、`advance_step()` は同じオブジェクトの配列をその場で更新します。各stepで新しい `SimulationState` を生成するわけではありません。`advance_step()` 自体の戻り値は `None` で、単独で使う場合の時刻加算と出力は呼び出し側で管理します。

`run_simulation()` のstep 0は初期状態の保存ではなく、1回更新した後の出力です。保存時の電位・電場はそのstepの密度更新前、粒子密度は更新後の値となります。最終stepが保存間隔に一致しない場合、最終状態のCSVは追加保存せず、関数の戻り値 `state` から参照します。

### 設定を変える場所と主な状態配列

計算条件の定義・既定値は `SimulationConfig` に集約し、実行条件は `SimulationConfig(...)` の引数で指定します。Notebookで同じ条件を別変数に重複定義せず、CLIはConfigの既定値を使います。CLIで指定できるのは `__main__.py` に登録された項目で、初期分布の幅などはPythonまたは実行Notebookから設定します。

| 変更したい条件 | `SimulationConfig` の項目 |
|---|---|
| 格子・領域 | `nr`、`nz`、`delta`（セル幅）、独立したギャップ長 `electrode_gap` |
| 電極・印加電圧 | `cathode_radius`、`voltage_mode`、`voltage` |
| インパルス波形 | `voltage_waveform_file`、`impulse_peak_voltage`、`impulse_time_scale` |
| 時間刻み・終了・保存間隔 | `dt`、`max_step`、`write_every` |
| Gaussian初期密度 | `nmax`（ピーク密度）、`z0`（軸方向中心）、`sr0`・`sz0`（分布幅） |
| 気体・背景電離 | `temperature`、`nitrogen_fraction`、`oxygen_fraction`、`background_ionization` |
| Poisson反復 | `poisson_omega`、`poisson_tolerance`、`poisson_progress_every` |

`SimulationState` の主な配列は以下です。設定は `config`、更新される計算結果は `state` として分けて保持します。

| 配列 | 内容・単位 |
|---|---|
| `r`・`z`、`rh`・`zh`、`dr`・`dz` | 各方向のセル下側境界・セル中心・セル幅 [m]。それぞれ長さ `nr` または `nz` の1次元配列 |
| `volume`・`sr`・`sz` | 軸対称有限体積法の体積・面積係数（共通の周方向係数 `2π` を省略）。いずれも `(nr,nz)` |
| `ne`、`n2`・`o2`、`n2p`・`o2p`、`om`・`o2m` | 電子、中性N₂・O₂、正イオンN₂⁺・O₂⁺、負イオンO⁻・O₂⁻の数密度 [m^-3] |
| `lphi`・`pphi`・`phi` | 電極による基礎電位・空間電荷による電位・両者の和 [V] |
| `ex`・`ey`・`abs_e` | 中心の半径方向・軸方向・大きさの換算電場 [Td]。名前の `x` はr方向、`y` はz方向に対応 |
| `er_face`・`ez_face` | 全セル面の法線換算電場 [Td]。形状は `(nr+1,nz)`・`(nr,nz+1)` |
| `vex`・`vey`、`mobility_r`・`mobility_z` | 同じ全セル面の電子速度 [m/s]・移動度 [m²/(V s)] |
| `radial_boundary`・`upper_boundary` | 外周・上端における印加1 Vへの電位応答。長さnz・nr、V/V |
| `rho`、`p1`～`p5` | 円筒有限体積Poisson用。`rho` は電荷密度/ε₀ [V/m²]で半径重みなし。`p1=0` の電極セルは更新しません |
| `gas_mask`、`unit_potential`、`electron_losses` | 気体・電極区分、離散単位電位、累積流出電子数（電極・接地・外側） |
| `k1`・`k2`・`k3` | 電離・解離付着・三体付着の反応係数。`k1`・`k2` は [m³/s]、`k3` は [m⁶/s] |

中心配列は `(nr,nz)` で、`state.ne[i,j]` は半径方向i・軸方向jのセルを表します。面配列は上表のとおり方向別の形状です。CSVの `Ex`・`Ey` は中心値で、軸方向電場を描くときは `Ey{step}.csv` を使います。

## 描画用Notebook・スクリプト

| ファイル | 用途 |
|---|---|
| [run_streamer.ipynb](run_streamer.ipynb) | 設定・計算実行・返された状態の描画をまとめて行う入口 |
| [plot_electrode_grid.ipynb](notebooks/plot_electrode_grid.ipynb) | 電極形状と計算格子の確認 |
| [plot_axial_field_profiles.ipynb](notebooks/plot_axial_field_profiles.ipynb) | 保存済みCSVから複数時刻の軸方向電場分布を比較 |
| [plot_streak_from_csv.ipynb](notebooks/plot_streak_from_csv.ipynb) | 保存済みCSVから位置×時間のストリーク画像を作成 |
| [animate_streamer.ipynb](notebooks/animate_streamer.ipynb) | 電子密度・電位・換算電場の時系列アニメーション |
| [animate_abs_ez.ipynb](notebooks/animate_abs_ez.ipynb) / [animate_abs_ez.py](notebooks/animate_abs_ez.py) | 軸方向換算電場の絶対値をアニメーション化。`.py` はCSVからGIFを作るCLI |

CSVを読むNotebookは、設定セルの `INPUT_DIR` または `OUTPUT_DIR` を実際の計算結果フォルダに合わせてください。各Notebookで入力先の既定値は異なります。座標と物理量CSVを同じフォルダに置き、適応出力では `times.csv` も残します。従来の固定出力では `DT_SECONDS` を計算時の `config.dt` に合わせます。

## 元の計算との対応

以下は固定刻みの動作です。Poissonは元の逐次SORから並列RB-SORへ変更しています。適応モードの更新回数・背景供給・出力時刻・Poisson上限は前述の仕様を使います。

- Juliaの `src/*.jl` は `streamer_drift_diffusion/*.py` に対応します。末尾の `!` はPythonの関数名では省略しています。Python添字 `[i,j]` はJuliaの `[i+1,j+1]` です。
- 配列形状は `(nr,nz)`、半径方向rが第1軸、軸方向zが第2軸です。電場 `ex/ey/abs_e` は換算電場 [Td]、電子密度 `ne` は [m^-3]、電位 `phi` は [V]、座標は [m] です。
- 電荷源 → 電位 → 電場 → 速度 → 反応係数 → 電子移流 → 反応、の順序を維持します。
- Poisson反復は並列RB-SOR、既定の緩和係数は1.5です。`pphi` は前stepの解を使い、電位尺度を反復間で保持します。grid-domainでは最外周も固定電位境界とし、旧版の自己参照項は使いません。
- `max_step` は最後の番号であり、更新回数は **max_step+1回** です。`--steps=0` も1回更新します。背景電離は毎更新の密度加算量で、dtを掛けません。
- CSVは1値1行、i外側・j内側、`%e` の小数点以下6桁、LF改行です。`ne{step}.csv`、`absE{step}.csv`、`Ex{step}.csv`、`Ey{step}.csv`、`phi{step}.csv`、`rh.csv`、`zh.csv` を保存します。
- 電位・電場はそのstepの密度更新前、電子密度は更新後の値です。stepラベルの時刻は `step*dt` に対応します。最終stepが `write_every` の倍数でなければ最終状態CSVを追加保存しません。
- DCとインパルスの既定値、波形のns→s・kV→V換算、振幅規格化、線形補間、端点保持、時間倍率を維持します。
- 元のモデルに電子拡散、光電離、再結合、脱離、イオン輸送はありません。現在のPython版では電子拡散を追加し、Poisson反復上限も設けています。負密度のクリップは行いません。
- Python版では元の設定検査に加え、格子幅・時刻刻み・Gaussian幅などが正の有限値であることを検査します。異なる言語・数学ライブラリに伴う丸め誤差があるため、浮動小数点配列のビット単位一致は保証しません。

## 検証コードと結果

| ファイル | 役割 |
|---|---|
| [tests/test_simulation.py](tests/test_simulation.py) | 初期状態、Poisson既知解、境界輸送、電荷保存、波形、CSVなどの単体テスト |
| [tests/test_grid_domain.py](tests/test_grid_domain.py) | 電極位置の独立性、体積、電極境界、離散直接解、電子吸収収支、試行棄却時の流出量復元、マスク出力 |
| [tests/test_diffusion.py](tests/test_diffusion.py) | 面係数・吸収境界、合算CFL、粒子収支、棄却時の復元、Gaussian格子収束、時間精度、固定刻み停止、Config・CLI・履歴 |
| [tests/test_fipy_transport.py](tests/test_fipy_transport.py) | 円筒セル対応、調和平均の独立解、半セル吸収、速度反転・D更新、ゼロ面の厳格数値検査、反応棄却後のFiPy旧値復元、非有限解の未反映 |
| [tests/test_staggered.py](tests/test_staggered.py) | 中心・全面の配置、既知電場、接線成分補間、離散Gauss則、電極吸収、外端流束、面移動度の時間制約、中心CSV、停止時の面配列保存 |
| [tests/test_rb_sor.py](tests/test_rb_sor.py) | RB更新順序に関連する離散解・電場、奇数格子と境界、チャンク間の反復継続、非有限値検出、1/2スレッドの計算結果一致 |
| [tests/test_timestepping.py](tests/test_timestepping.py) | 刻み制約、背景供給率、失敗時の復元、実時刻出力、停止条件、描画用時刻対応表の検証 |
| [tests/compare_julia.py](tests/compare_julia.py) / [tests/julia_reference.jl](tests/julia_reference.jl) | Python／Juliaを同じ条件で実行し、全状態配列とCSVを比較 |
| [tests/validate_c_outputs.py](tests/validate_c_outputs.py) | 既存Python出力と、元設定／条件を合わせた設定で新規実行したC出力を比較。検証用コピー・ログ・比較表を保存 |
| [tests/validate_c_stream.py](tests/validate_c_stream.py) | C出力を標準出力パイプで受信して比較する方式。`validate_c_outputs.py` の準備処理などを利用 |
| [tests/report_c_validation.py](tests/report_c_validation.py) | C比較の保存済み集計データから図と日本語レポートを生成 |
| [VALIDATION.md](VALIDATION.md) / [validation_c_5000_20261027/REPORT.md](validation_c_5000_20261027/REPORT.md) | 検証条件・比較結果・長時間計算で確認された数値不安定性の記録 |

C比較スクリプトは、既存のPython出力、隣接する元Cコード、Cコンパイラなどを前提とする追加検証用です。通常の計算実行には不要で、入力先やコンパイラの設定は各スクリプトで確認してください。

Juliaとの比較には隣接する元の `streamer_julia` とJulia実行環境が必要です。

```powershell
python tests/compare_julia.py
# 実行ファイルを指定する場合:
python tests/compare_julia.py --julia=C:/path/to/julia.exe
```

検証条件・実測結果は [VALIDATION.md](VALIDATION.md) を参照してください。

5000 step出力の追加検証では、条件を合わせたCとstep 4800までよく一致し、
step 4900以降で両方に負密度・Inf電場が確認されました。
この設定での長時間計算には数値安定性の再検討が必要です。
詳細は [Cとの5000 step比較レポート](validation_c_5000_20261027/REPORT.md) にまとめています。

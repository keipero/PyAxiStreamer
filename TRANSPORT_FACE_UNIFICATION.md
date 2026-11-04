# 電子輸送・CFL・境界流出の面係数統合

実施日: 2026-11-04（日本時間）。一次上流・陽的Euler、反応式、電極形状、
Poisson求解、半セル吸収、外部からのゼロ流入は維持した。

## 共通化した処理

`ElectronTransport`の初期化時に、共有格子から気体間面・吸収境界面・
気体側の向きと離散化距離を求める。計算中は格子・電極形状を固定する。

`prepare(state)`は現在の全面速度と中心Dを使い、次を一度に更新する。

- 気体間Dの調和平均と、吸収面における気体側D。
- 全面の拡散コンダクタンス`diffusion_rates = D*A/d` [m³/s]。
- 境界の吸収係数`boundary_rates = A*max(v_out,0) + D*A/d` [m³/s]。
- FiPyの内部移流・拡散係数と、セルごとの境界吸収損失項。
- 気体セルの合算輸送流出率と純拡散流出率から求める陽的時間上限。

面積・体積では2πを省略する。方程式の境界損失と、電極・接地・外端への
流出電子数は同じ`boundary_rates`を使い、流出数の集計時に一度だけ2πを掛ける。
気体間流束は両側で保存し、流出数には含めない。

合算上限`transport.transport_limit`と純拡散上限`transport.diffusion_limit`は、
最大流出率 [s⁻¹]の逆数で、流出率0なら無限大。
`timestep_limits()`が合算上限へ`dt_cfl_number`を掛け、従来どおり
他の物理上限・安全係数・設定上限も適用する。

## 係数の更新と再利用

共通時間ループは各更新前に`timestep_limits(..., transport=transport)`を呼ぶ。
ここで生成した係数を`advance_density(..., transport=transport, prepared=True)`へ引き継ぐ。
棄却時は7密度を復元して刻みを半減し、係数を再生成せず同じ旧値に基づく試行を行う。
FiPy密度の現在値・旧値は各試行で再同期し、受理した分だけ累積流出数へ加算する。

`step()`と`advance_density()`の既定は`prepared=False`であり、手動で変更した速度・Dも反映する。
`prepared=True`は同じ状態の速度・Dで準備済みの輸送オブジェクトを渡す場合にだけ使う。
密度だけの変更では再準備は不要だが、速度・Dを変えた後は`prepare()`を呼ぶ。

```python
from streamer_drift_diffusion import SimulationConfig, initialize_simulation, ElectronTransport
from streamer_drift_diffusion.simulation import prepare_step
from streamer_drift_diffusion.timestepping import timestep_limits, advance_density

config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
                          background_ionization=0., t_end=0.)
state = initialize_simulation(config)
transport = ElectronTransport(state, config.delta)
prepare_step(state, config, 0., verbose=False)
limits = timestep_limits(state, config, transport=transport)
dt, retries = advance_density(state, config, limits.dt, transport=transport, prepared=True)
prepare_step(state, config, dt, verbose=False)
```

## 廃止した内部API

- `timestepping.diffusion_loss()`、`diffusion_limit()`、`transport_limit()`。
- `CylindricalGrid.diffusion_faces()`、`diffusion_rates()`。

互換ラッパーは残さない。面係数と輸送上限は準備済みの`ElectronTransport`を参照する。
格子モジュールは共有メッシュ・面情報・幾何操作を担当し、物理係数の生成は輸送側へ集約した。

## コード量

計算パッケージ全体で、空行・コメント行・ASTで識別したdocstring行を除くと
862行から857行へ5行減少した。新しい係数準備・再利用経路も含めた正味の値。
削減幅は小さく、今回の主な効果は係数の二重評価をなくし、輸送・CFL・収支の整合性を保つことにある。
廃止した関数のdocstring削除は実装の削減量に含めない。
現存する全65定義にNumPy形式docstringを備え、64定義にExamplesがある。

## 検証

全103テストが成功（既存100件に3件追加）。全モジュールのdoctest315項目も成功。
追加テストでは、内部電極を含む格子の各気体セルへ単位密度を置き、実際のFiPy更新から
減少率を測って合算輸送・純拡散上限と比較した。各試行の粒子収支も確認した。
さらに、CFL評価・棄却・再試行を通して係数準備が一度であること、
固定／適応の共通ループでも準備が各受理更新につき一度であることを確認した。

変更前との比較は8×12セル、delta=1e-4 m、ギャップ6e-4 m、曲率半径2e-4 m、
初期ピーク密度1e16 m⁻³、背景加算1e11 m⁻³、基準dt=1e-11 s、
終了1.1e-10 s、出力間隔2e-11 sで実施した。
正負DC±500 Vと、ピーク500 Vの入力波形
`(時刻ns,振幅)=(0,0),(.02,1),(.04,-.5),(.11,0)`について、
D=0／0.5 m²/s、固定／適応の計12条件を比較した。

最終状態の全数値配列に加え、固定／適応インパルスの出力CSV・時刻表・刻み履歴も検査した。
計500項目中488項目が完全一致した。差は一部条件のne・om・rho、累積流出電子数と、
刻み履歴のCFL上限・累積流出数に生じた。
各配列の`max(abs(new-old))/max(abs(old))`（分母下限1e-300）の最大値は1.889e-16。
吸収係数の乗除算・加算順序と、CFL上限の逆数・倍率適用順序をまとめたことに伴う丸め誤差の範囲である。
電位・電場・出力時刻・物理量CSVは一致した。履歴の受理刻み・支配条件・再試行回数も一致した。

## 実行時間

上記12条件を無出力で実行し、ウォームアップ後に計測した。
別プロセスでの各3回の再計測は、変更前4.19〜8.37 s、変更後4.15〜4.88 sだった。
同一プロセスに両版を別名で読み込み、実行順を入れ替えた4回の計測も行った。

| 方式 | 最小〜最大 [s] | 中央値 [s] |
|---|---:|---:|
| 変更前 | 6.50〜8.86 | 7.43 |
| 変更後 | 5.23〜16.29 | 9.85 |

交互計測では変更後の中央値が大きく、安定した高速化は確認できなかった。
計測方法・回による変動が大きいため、この結果を長時間計算や大格子の速度へ外挿しない。
今回確認したのは係数の重複評価の解消と数値的一致であり、速度改善の保証ではない。
単独の`step()`は既定でCFL上限も再評価するため、速度・Dを固定した反復計算では
準備済み係数の明示的な再利用が適している。

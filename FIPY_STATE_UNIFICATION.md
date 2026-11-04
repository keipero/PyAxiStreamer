# 電子密度・電位のFiPy正本化

実施日: 2026-11-05（日本時間）。現在のブランチで行うFiPy移行の仕上げとして、
電子密度・電位の現在値の二重保持を解消した。バックエンドはSciPyを維持し、PETScは未導入。
空間離散化、半セル境界条件、一次上流・陽的Euler、反応式、時間刻み方針は維持した。

## 値の所有者

| 物理量 | 正本 | 同じ変数を使う処理 | NumPyからの参照 |
|---|---|---|---|
| 電子密度 [m⁻³] | `state.ne_var` | `ElectronTransport.density` | `state.ne` |
| 総電位 [V] | `state.phi_var` | `PoissonSolver.potential` | `state.phi` |

正本は共通メッシュ上の`CellVariable`で、値はFiPyセル番号順の1次元配列。
`state.ne`と`state.phi`は、これをFortran順で`(nr,nz)`へ変形したビューであり、値を別途保持しない。
他の6密度、反応係数、電場・速度などは従来のNumPy配列を維持した。
CSVの物理量名、単位、C順序の保存形式も維持する。

NumPyビューへの要素代入は正本の値を変える。FiPyの計算済み派生式にも変更を通知する代入は
`ne_var.setValue()`・`phi_var.setValue()`を使う。組込みの初期密度設定、反応、棄却からの復元は
このAPIを使う。独立した状態の退避には`state.ne.copy()`・`state.phi.copy()`を使う。

```python
import numpy as np
from streamer_drift_diffusion import SimulationConfig, initialize_simulation, ElectronTransport

state = initialize_simulation(SimulationConfig(nr=2, nz=3, t_end=0.))
transport = ElectronTransport(state, state.grid.spacing)
assert transport.density is state.ne_var
assert state.poisson.potential is state.phi_var
assert np.shares_memory(state.ne, state.ne_var.value)

density = np.full(state.grid.shape, 1e12)
state.ne_var.setValue(density.ravel(order='F'))
snapshot = state.ne.copy()
```

## 輸送・反応・失敗時の復元

輸送用の別の電子密度変数の生成と、輸送前後の現在値のコピーを廃止した。
各試行では正本に対して`updateOld()`し、陽的Eulerに必要な旧値を確保する。
旧値は時間離散化と求解失敗時の復元に必要であり、現在値の二重保持とは区別する。

FiPy輸送が途中で例外を送出した場合や非有限解を返した場合、正本の電子密度を旧値へ復元する。
輸送・反応後の密度検査や物理制約による棄却では、共通の`advance_density()`が全7密度を復元する。
再試行でも旧値を取り直し、受理した試行だけ流出数を累積する。
電子密度の復元はFiPy派生式にも通知し、反応更新後のキャッシュを残さない。

Poissonは正本を直接更新する。ゼロ初期値からのLU求解・有限性・残差検査を維持し、
求解例外や残差異常では求解前の電位へ復元する。復元用のコピーは呼出し中だけ保持する。
`calculate_potential()`で解を独立したNumPy配列へコピーする処理は廃止した。

`PoissonSolver.solve()`の戻り値も正本の2次元ビューに変更した。
以前の解との差分を求める場合など、求解をまたいで保持する結果には明示的に`.copy()`を付ける。
状態の直接生成に使うdataclass引数は`ne`・`phi`から`ne_var`・`phi_var`へ変更した。
通常の`initialize_simulation()`と`run_simulation()`の呼出しは変更不要。

## 出力と停止診断

物理量CSVは従来どおりne・phiの2次元値を保存する。
`failure_state.npz`にも同じ名前・形状で数値を保存し、CellVariable・メッシュ・方程式を含めない。
`np.load(..., allow_pickle=False)`で読めることを検証した。

## 検証

全109テストが成功。既存103件に、正本の同一性・メモリ共有、反応後のFiPy派生式更新、
輸送・Poissonの中途例外からの復元、最終棄却からの復元、数値のみの停止診断を確認する6件を追加した。
旧テストの状態フィールド数への依存は、ne・phiのビューも含む物理配列の独立性検査へ更新した。
全67定義にNumPy形式docstring、66定義にExamplesがあり、doctest全324項目が成功した。
標準200×150格子のCLI初期状態計算（`--t-end=0 --no-output --quiet`）も正常終了した。

変更前との比較条件は8×12セル、delta=1e-4 m、ギャップ6e-4 m、曲率半径2e-4 m、
初期ピーク密度1e16 m⁻³、背景加算1e11 m⁻³、基準dt=1e-11 s、終了1.1e-10 s、出力間隔2e-11 s。
正負DC±500 Vとピーク500 Vの波形`(時刻ns,振幅)=(0,0),(.02,1),(.04,-.5),(.11,0)`を、
D=0／0.5 m²/s、固定／適応の計12条件で比較した。

最終状態の全数値配列と、各条件の全物理量CSV・座標・時刻表・刻み履歴の900項目を比較した。
最終状態の密度・電荷源・電位・電場、物理量CSV、出力時刻、受理刻み・再試行回数は完全一致した。
差は12条件の刻み履歴に記録した電子数の体積積分だけで、最大スケール誤差は2.024e-16。
正本をFiPyセル番号順にしたことによる配列の走査・加算順序の丸め差である。
スケール誤差は`max(abs(new-old))/max(abs(old))`、分母下限1e-300で評価した。

## コード量と保持メモリ

空行・コメント行・docstring行を除く実装は857行から875行へ18行増えた。
現在値の共有に伴う明示的なビューと失敗時の保護を含む正味の値で、当初の880行以下を維持する。
今回の効果は保持値と同期経路の削減であり、実装行数の削減とは区別する。

輸送オブジェクト生成後に保持する電子密度・電位の独立した現在値バッファを合計2個減らした。
標準200×150格子・float64では値バッファ約480,000 bytes分に相当する。
FiPyの旧値と、一時的な失敗復元用バッファは引き続き必要。実行時間の改善は今回の短時間比較から断定しない。

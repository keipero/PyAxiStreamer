"""格子，有限体積法の体積・面積係数を生成する。"""
import numpy as np


def grid_axis(count, spacing):
    """
    有限体積法で使う1次元等間隔格子の座標とセル幅を生成する。

    Args:
        count (int): セル数。2以上を指定する。
        spacing (float): セル幅 [m]。正の有限値を指定する。

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 以下の配列のタプル。
            各配列は長さcount、dtypeはfloat64、単位は [m]。
            edgesは各セルの下側境界で、edges[i] = i * spacing。
            centersは各セルの中心で、centers[i] = (i + 0.5) * spacing。
            widthsは各セルの幅で、隣接境界の差を使い、末尾は直前の幅を使う。

    Notes:
        edgesには最終セルの上側境界count * spacingを含めない。
        セル中心の計算時は、その境界を補って使う。
        引数の検査はこの関数では行わず、呼び出し側で行う。
    """
    edges = spacing * np.arange(count, dtype=np.float64)
    # 第二項目で，右側境界を作り，左側境界のedgeと組み合わせてセル中心を計算．
    centers = 0.5 * (edges + np.append(edges[1:], edges[-1] + spacing))
    differences = np.diff(edges)
    # 末尾のセル幅は直前の幅を使う．
    widths = np.append(differences, differences[-1])
    return edges, centers, widths


def initialize_grid(config):
    """
    軸対称r-z格子と、有限体積法の体積・面積係数を生成する。

    Args:
        config (SimulationConfig): 格子設定。nrを半径方向のセル数、
            nzを軸方向のセル数、deltaを両方向共通のセル幅 [m] として使う。
            nr・nzは2以上、deltaは正の有限値を指定する。

    Returns:
        dict[str, np.ndarray]: 以下のキーを持つ格子データ。
            r・z: 各セルの下側境界 [m]。形状はそれぞれ(nr,)・(nz,)。
            rh・zh: セル中心 [m]。形状はそれぞれ(nr,)・(nz,)。
            dr・dz: セル幅 [m]。形状はそれぞれ(nr,)・(nz,)。
            volume: セル体積係数 [m^3]。形状は(nr, nz)。
            sr: 半径方向の下側セル面の面積係数 [m^2]。形状は(nr, nz)。
            sz: 軸方向の下側セル面の面積係数 [m^2]。形状は(nr, nz)。

    Notes:
        2次元配列の第1軸は半径方向、第2軸は軸方向。
        volume・sr・szでは、軸対称の共通係数2πを省略する。
    """
    r, rh, dr = grid_axis(config.nr, config.delta) # 半径方向の格子を生成
    z, zh, dz = grid_axis(config.nz, config.delta) # 軸方向の格子を生成
    # 円筒セルの体積・面積係数を計算．各セルの内側半径と外側半径の和を表す．
    # r_out^2 - r_in^2 = (r_out + r_in) * (r_out - r_in) = radius_sum * dr
    # 最外セルも上側境界r[-1]+dr[-1]まで含む完全な円環セルとする。
    radius_sum = 2 * r + dr
    # 共通係数2πを省略して，体積・面積係数を計算する．
    # Volume = dz * radius_sum * dr / 2.0
    volume = dz[None, :] * radius_sum[:, None] * dr[:, None] / 2.0
    # sr, szは軸対称の360°全体を想定する．
    sr = r[:, None] * dz[None, :]
    # 体積形状に合わせて，拡張する．
    sz = np.broadcast_to((radius_sum * dr / 2.0)[:, None], volume.shape).copy()
    gas_mask = (zh[None, :] / config.electrode_gap)**2 - rh[:, None]**2 / (
        config.cathode_radius * config.electrode_gap) < 1.
    return dict(r=r, z=z, rh=rh, zh=zh, dr=dr, dz=dz, volume=volume, sr=sr, sz=sz,
                gas_mask=gas_mask, electron_losses=np.zeros(3))


# grid.py のテスト用コード。直接実行すると、格子を描画する。
if __name__ == "__main__":
    from types import SimpleNamespace
    import matplotlib.pyplot as plt

    # 見やすい小さな格子で表示。セル数とセル幅はここで変更する。
    count = 100
    spacing = 1.0
    edges, centers, widths = grid_axis(count, spacing)

    # edges は各セルの左境界のみ。描画時は最終セルの右境界も補う。
    boundaries = np.append(edges, edges[-1] + widths[-1])
    fig, grid_ax = plt.subplots(figsize=(8,8), layout="constrained")

    config = SimpleNamespace(nr=count, nz=100, delta=spacing,
                             electrode_gap=70., cathode_radius=20.)
    grid = initialize_grid(config)
    r_boundaries = np.append(grid["r"], grid["r"][-1] + grid["dr"][-1])
    z_boundaries = np.append(grid["z"], grid["z"][-1] + grid["dz"][-1])
    grid_ax.vlines(z_boundaries, r_boundaries[0], r_boundaries[-1], colors="gray", linewidth=0.8, label="Cell boundaries")
    grid_ax.hlines(r_boundaries, z_boundaries[0], z_boundaries[-1], colors="gray", linewidth=0.8)
    z_centers, r_centers = np.meshgrid(grid["zh"], grid["rh"])
    # grid_ax.scatter(z_centers.ravel(), r_centers.ravel(), s=15, color="tab:orange", zorder=3, label="Cell centers")
    grid_ax.set(xlim=(z_boundaries[0], z_boundaries[-1]),
                ylim=(r_boundaries[0], r_boundaries[-1]),
                xlabel="z", ylabel="r",
                title=f"r-z grid ({config.nr} x {config.nz} cells)")
    grid_ax.set_aspect("equal", adjustable="box")
    plt.show()

"""円筒有限体積の電子ドリフト・拡散。全ての面輸送量を旧密度から求める。"""
import numpy as np


def face_transport(velocity, lower_density, upper_density, dt, area):
    """一次上流法で、1つのセル面を通る符号付き輸送量を求める。

    Args:
        velocity (float): セル面における速度 [m/s]。
            正の場合は座標の小さい側から大きい側へ輸送する。
        lower_density (float): セル面の座標が小さい側の粒子数密度 [m^-3]。
        upper_density (float): セル面の座標が大きい側の粒子数密度 [m^-3]。
        dt (float): 時間刻み [s]。
        area (float): セル面積または面積係数 [m^2]。

    Returns:
        float: dt * velocity * 上流側密度 * areaで求めた輸送量。
            正は座標の大きい側への輸送、負は小さい側への輸送を表す。
            面積を指定した場合は粒子数、共通係数2πを省略した
            面積係数を指定した場合は2πを省略した粒子数となる。

    Notes:
        速度が正または0ならlower_density、負ならupper_densityを使う。
        この関数は密度を更新せず、1面の輸送量だけを返す。
        密度の更新はnele_flow()で行う。
    """
    if velocity >= 0:
        return dt * velocity * lower_density * area
    if velocity < 0:
        return dt * velocity * upper_density * area
    return 0.0


def face_areas(sr, sz):
    """領域の外端を含む全セル面の面積係数（2π省略）を返す。"""
    return (np.concatenate((sr, 2*sr[-1:]-sr[-2:-1]), axis=0),
            np.concatenate((sz, sz[:, -1:]), axis=1))


def transport_rates(vr, vz, sr, sz):
    """全セル面の速度×面積係数を返す。速度は外端を含む面配列。

    旧APIの下側面だけの配列も受け付け、その場合だけ最終面を1点外挿する。
    シミュレーション本体は常に全セル面を渡す。
    """
    ar, az = face_areas(sr, sz)
    if vr.shape == (ar.shape[0]-1, az.shape[1]-1):
        vr = np.concatenate((vr, vr[-1:]), axis=0)
    if vz.shape == (ar.shape[0]-1, az.shape[1]-1):
        vz = np.concatenate((vz, vz[:, -1:]), axis=1)
    if vr.shape != (ar.shape[0], az.shape[1]-1) or vz.shape != (ar.shape[0]-1, az.shape[1]):
        raise ValueError("velocities must have radial (nr+1,nz) and axial (nr,nz+1) face shapes")
    radial = vr * ar
    axial = vz * az
    radial[0] = 0.  # 回転対称軸。
    return radial, axial


def diffusion_rates(diffusion, gas_mask, spacing, sr, sz):
    """各面のD×面積/距離 [m^3/s] を返す。2πは省略する。

    気体間のDは調和平均。電極・接地面・上端・外周はn=0の吸収境界で、
    気体側のDと半セル距離を使う。軸は対称条件（面積0）。
    diffusionはスカラーまたはセル形状の配列。spacingは等間隔格子幅 [m]。
    """
    if spacing is None or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("diffusion requires a finite positive spacing [m]")
    coefficient = np.where(gas_mask, diffusion, 0.)
    if np.any(~np.isfinite(coefficient)) or np.any(coefficient < 0):
        raise ValueError("diffusion coefficients must be finite and nonnegative")
    rates = []
    for axis, area in enumerate(face_areas(sr, sz)):
        low_pad, high_pad = [(0, 0), (0, 0)], [(0, 0), (0, 0)]
        low_pad[axis], high_pad[axis] = (1, 0), (0, 1)
        low, high = np.pad(coefficient, low_pad), np.pad(coefficient, high_pad)
        both = np.pad(gas_mask, low_pad) & np.pad(gas_mask, high_pad)
        face = low + high
        np.divide(2*low*high, low+high, out=face, where=both & (face > 0.))
        rates.append(face * area / np.where(both, spacing, spacing/2))
    return rates[0], rates[1]


def nele_flow(m, vr, vz, dt, volume, sr, sz, gas_mask=None, diffusion=None, spacing=None):
    """ドリフトを一次上流、拡散を面勾配で同時更新し、境界流出数を返す。

    Args:
        m (np.ndarray): 電子密度 [m^-3]。その場更新。
        vr, vz (np.ndarray): 全セル面の速度 [m/s]。
            形状は半径面(nr+1,nz)、軸方向面(nr,nz+1)。
            旧APIの下側面だけの(nr,nz)配列も受け付ける。
        dt (float): 時間幅 [s]。
        volume, sr, sz (np.ndarray): 2πを省略した体積・面積係数。
        gas_mask (np.ndarray, optional): 気体=True。省略時は全セルを気体とする。
        diffusion (float or np.ndarray, optional): 拡散係数 [m^2/s]。既定Noneで拡散なし。
        spacing (float, optional): 等間隔格子幅 [m]。正の拡散係数を使う場合に必要。

    Returns:
        np.ndarray: 流出電子数 [電極、接地平面、上端＋外周]。2πを含む。

    Notes:
        全ての面輸送量を旧密度から求め、セルの収支を一度に適用する。
        軸は流束0。電極・領域外の密度を0とし、流出・吸収だけを許す。
        上端と外周も渡された面速度を使い、自面流束との相殺は行わない。
        拡散も同じ吸収境界を使う。粒子の反射・電子放出は含まない。"""
    gas = np.ones_like(m, dtype=bool) if gas_mask is None else gas_mask
    m[~gas] = 0.
    radial, axial = transport_rates(vr, vz, sr, sz)
    # 電極内と領域外の電子密度は0。吸収・流出のみを許し、外部から供給しない。
    inside, outside = np.pad(m, ((1, 0), (0, 0))), np.pad(m, ((0, 1), (0, 0)))
    below, above = np.pad(m, ((0, 0), (1, 0))), np.pad(m, ((0, 0), (0, 1)))
    radial = dt * radial * np.where(radial >= 0., inside, outside)
    axial = dt * axial * np.where(axial >= 0., below, above)
    if diffusion is not None and np.any(diffusion):
        dr, dz = diffusion_rates(diffusion, gas, spacing, sr, sz)
        radial += dt * dr * (inside-outside)
        axial += dt * dz * (below-above)
    change = radial[:-1] - radial[1:] + axial[:, :-1] - axial[:, 1:]
    losses = 2*np.pi * np.array([np.sum(change[~gas]), -np.sum(axial[:, 0]),
                                 np.sum(radial[-1]) + np.sum(axial[:, -1])])
    m[:] += change / volume
    m[~gas] = 0.
    return losses

"""时间积分与空间采样

环境温度/含水率是分段线性的，斜率折点处右端不光滑。这里在每个折点处分段
积分，保证不跨越间断；段末状态精确传给下一段，段内步长仍由 BDF 自适应。
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .model import T_END, Grid, Operator, Parameters


@dataclass
class Trajectory:
    """分段密集输出。segments[i] 覆盖时间区间 (ends[i-1], ends[i]]。"""

    segments: list
    ends: np.ndarray
    offset: float
    stats: dict

    def at(self, times):
        """在任意时刻取值；times 为标量时返回一维数组。"""
        scalar = np.ndim(times) == 0
        ts = np.atleast_1d(np.asarray(times, dtype=float))
        idx = np.minimum(np.searchsorted(self.ends, ts, side='left'),
                         len(self.ends) - 1)
        out = np.empty((len(self.segments[0].y), len(ts)))
        for k in np.unique(idx):
            m = idx == k
            out[:, m] = self.segments[k].sol(ts[m]) + self.offset
        return out[:, 0] if scalar else out


def integrate(op, u0, end=T_END, method='BDF', rtol=1e-10, atol=1e-12):
    """对环境数据折点分段积分，返回 Trajectory。"""
    breaks = op.env.t
    cuts = np.unique(np.concatenate(([0.0], breaks[(breaks > 0) & (breaks < end)], [end])))
    u = np.asarray(u0, dtype=float)
    segments = []
    nfev = njev = nlu = nsteps = 0
    for a, b in zip(cuts[:-1], cuts[1:]):
        sol = solve_ivp(op.rhs, (a, b), u, method=method, jac=op.jac,
                        rtol=rtol, atol=atol, dense_output=True)
        if not sol.success or abs(sol.t[-1] - b) > 1e-8:
            raise RuntimeError(f'{op.kind} 场在 [{a}, {b}] 积分失败: {sol.message}')
        if op.kind == 'C' and sol.y.min() <= 0:
            raise RuntimeError('浓度出现非正值，请检查参数或收紧容差')
        u = sol.y[:, -1]
        segments.append(sol)
        nfev += sol.nfev
        njev += sol.njev
        nlu += sol.nlu
        nsteps += len(sol.t) - 1
    stats = {'method': method, 'rtol': rtol, 'atol': atol,
             'steps': nsteps, 'nfev': nfev, 'njev': njev, 'nlu': nlu}
    return Trajectory(segments, cuts[1:], op.p.T0 if op.kind == 'T' else 0.0, stats)


def solve_fields(env, p=None, N=2560, beta=2.5, end=T_END,
                 rtol=1e-10, atol_T=1e-10, atol_C=1e-12, flux='kirchhoff'):
    """
    求解温度场与水分场
    温度场初值取 0（内部变量 θ = T − T0），水分场初值取 C0
    """
    p = p or Parameters()
    if env.t[-1] < end:
        raise ValueError('环境数据未覆盖求解终点')
    grid = Grid.build(N, p.R, beta)
    heat = integrate(Operator(grid, p, 'T', env, flux),
                     np.zeros(N + 1), end, rtol=rtol, atol=atol_T)
    wet = integrate(Operator(grid, p, 'C', env, flux),
                    np.full(N + 1, p.C0), end, rtol=rtol, atol=atol_C)
    return grid, heat, wet


def sample(traj, grid, times, radii):
    """把节点解线性插值到指定半径。"""
    values = traj.at(times)
    if values.ndim == 1:
        return np.interp(radii, grid.r, values)
    return np.array([np.interp(radii, grid.r, values[:, i])
                     for i in range(values.shape[1])])

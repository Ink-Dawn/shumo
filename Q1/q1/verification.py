"""
问题一的验证与诊断量
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import erfcx, j0, j1, jn_zeros
from scipy.optimize import brentq

from .data import Environment
from .model import Grid, Operator, Parameters, T_END, diffusivity
from .solver import integrate, sample, solve_fields


def _alpha(p):
    """热扩散率，未放到 Parameters 里是因为正式求解用不到"""
    return p.k / (p.rho * p.cp)


def balance_history(traj, grid, env, p, kind, times, order=8):
    """
    独立边界积分守恒检查：相对残差随时间的变化
    在内部时间步区间上再做一次 Gauss 积分，与场内总量变化相减
    """
    x, w = leggauss(order)
    factor = 2 * np.pi * p.R * (p.h if kind == 'T' else p.hm)
    mass = grid.V * (p.rho * p.cp if kind == 'T' else 1.0)
    nodes = np.unique(np.r_[times, *[s.t for s in traj.segments]])
    nodes = nodes[(nodes >= 0) & (nodes <= max(times))]
    mid, half = (nodes[1:] + nodes[:-1]) / 2, np.diff(nodes) / 2
    queries = (mid[:, None] + half[:, None] * x).ravel()
    # 分块采样，避免 N 大时一次占用过多内存
    chunks = max(1, len(queries) // 2048 + 1)
    surface = np.concatenate([traj.at(q)[-1]
                              for q in np.array_split(queries, chunks)])
    f = factor * (surface - env.value(queries, kind))
    integ = half * (f.reshape(-1, order) @ w)
    total = np.r_[0., np.cumsum(integ)]
    absolute = np.r_[0., np.cumsum(half * (np.abs(f).reshape(-1, order) @ w))]
    idx = np.searchsorted(nodes, times)
    delta = mass @ (traj.at(times) - traj.at(0)[:, None])
    residual = delta + total[idx]
    floor = mass.sum() * (1.0 if kind == 'T' else p.C0 * 0.01)
    scale = np.maximum.reduce([np.abs(delta), absolute[idx], np.full(len(times), floor)])
    return {'time': np.asarray(times), 'relative': residual / scale}


def cylinder_linear(env, kind, times, radii, p, modes=160):
    """常系数圆柱 Robin 问题的解析模态解，用于热场独立验证"""
    initial = p.T0 if kind == 'T' else p.C0
    kappa = _alpha(p) if kind == 'T' else p.A
    Bi = p.h * p.R / p.k if kind == 'T' else p.hm * p.R / p.A
    if Bi <= 0:
        return np.full((len(times), len(radii)), initial)
    zeros = np.r_[0., jn_zeros(1, modes)]
    fun = lambda x: x * j1(x) - Bi * j0(x)
    roots = np.array([brentq(fun, a + 1e-10, b - 1e-10)
                      for a, b in zip(zeros[:-1], zeros[1:])])
    lam = kappa * (roots / p.R) ** 2
    coeff = 2 * j1(roots) / (roots * (j0(roots) ** 2 + j1(roots) ** 2))
    basis = j0(roots[:, None] * np.asarray(radii)[None, :] / p.R)
    g0 = env.value(0., kind)
    z = coeff * (initial - g0)
    events = np.unique(np.r_[0., env.t[(env.t > 0) & (env.t < max(times))], times])
    results = {0.: np.full(len(radii), initial)}
    prev = 0.
    for t in events[1:]:
        dt = t - prev
        slope = (env.value(t, kind) - env.value(prev, kind)) / dt
        z = np.exp(-lam * dt) * z + coeff * slope * np.expm1(-lam * dt) / lam
        results[float(t)] = env.value(t, kind) + z @ basis
        prev = t
    return np.array([results[float(t)] for t in np.sort(times)])


def manufactured(p):
    """非线性水分制造解，配源项与边界。"""
    def exact(t, r):
        t = np.asarray(t)
        r = np.asarray(r)
        return 2.2 + 0.2 * np.exp(-t / 600) * (r / p.R) ** 2
    def source(t, r):
        B = 0.2 * np.exp(-t / 600)
        C = exact(t, r)
        D = diffusivity(C, p)
        dD = D * p.a / C ** 2
        return -B / 600 * (r / p.R) ** 2 - 4 * B * D / p.R ** 2 - 4 * B ** 2 * r ** 2 * dD / p.R ** 4
    def boundary(t):
        B = 0.2 * np.exp(-t / 600)
        Cs = exact(t, p.R)
        return Cs + 2 * B * diffusivity(Cs, p) / (p.R * p.hm)
    return exact, source, boundary


def manufactured_errors(p):
    """返回制造解在各网格下的最大误差"""
    exact, src, bc = manufactured(p)
    env = Environment([0., 600.], [28., 28.], [2.2, 2.2], 'dummy')
    errors = []
    for N in (40, 80, 160, 320):
        grid = Grid.build(N, p.R)
        u0 = exact(0., grid.r)
        wet = integrate(Operator(grid, p, 'C', env, boundary=bc, source=src),
                        u0, end=600.0, rtol=1e-13, atol=1e-15)
        tt = np.array([0., 1., 10., 100., 300., 600.])
        err = np.max(np.abs(wet.at(tt).T - np.array([exact(t, grid.r) for t in tt])))
        errors.append({'N': N, 'error': float(err)})
    return errors


def surface_short_time(t, p, Ce):
    """常 D(C0) 半无限平面 Robin 边界近似下的表面浓度"""
    D0 = float(diffusivity(p.C0, p))
    z = p.hm * np.sqrt(np.asarray(t) / D0)
    return Ce + (p.C0 - Ce) * erfcx(z)


def spatial_convergence(env, p, Ns=(160, 320, 640, 1280, 2560, 5120)):
    """相邻网格的水分最大差，作为空间收敛的代理量"""
    times = np.arange(1.0, T_END + 1.0)
    radii = np.linspace(0.0, p.R, 21)
    prev = None
    pairs = []
    for N in Ns:
        grid, _, wet = solve_fields(env, p, N=N, end=T_END)
        cur = sample(wet, grid, times, radii)
        if prev is not None:
            diff = np.max(np.abs(cur - prev))
            pairs.append({'coarse': Ns[len(pairs)], 'fine': N, 'diff': diff})
        prev = cur
    return pairs


def temporal_convergence(env, p, tols=(1e-6, 1e-8, 1e-9)):
    """同一网格下不同时间容差的最大差，参考 rtol=1e-11"""
    times = np.arange(1.0, T_END + 1.0)
    radii = np.linspace(0.0, p.R, 21)
    grid_ref, _, wet_ref = solve_fields(env, p, N=160, rtol=1e-11, atol_T=1e-13, atol_C=1e-13)
    ref = sample(wet_ref, grid_ref, times, radii)
    out = []
    for tol in tols:
        grid, _, wet = solve_fields(env, p, N=160, rtol=tol, atol_T=tol * 0.01, atol_C=tol * 0.01)
        err = np.max(np.abs(sample(wet, grid, times, radii) - ref))
        out.append({'rtol': tol, 'atol': tol * 0.01, 'error': err})
    return out


def sensitivity(env, p, grid, wet, deltas=(0.02,)):
    """A 与 hm 在 ±2% 扰动下的归一化浓度灵敏度"""
    t_end = T_END
    C_nom = wet.at(t_end)
    V = grid.V
    V_sum = V.sum()
    results = {}
    for name, field in [('扩散前因子 A', 'A'), ('表面交换系数 hm', 'hm')]:
        sens = np.zeros_like(grid.r)
        for d in deltas:
            kwargs = {field: getattr(p, field) * (1.0 + d)}
            p_pert = replace(p, **kwargs)
            wet_pert = integrate(Operator(grid, p_pert, 'C', env),
                                 np.full(len(grid.r), p_pert.C0), end=t_end)
            C_pert = wet_pert.at(t_end)
            sens += ((C_pert - C_nom) / C_nom) / d
        sens /= len(deltas)
        surf = sens[-1]
        avg = (V @ sens) / V_sum
        results[name] = {'profile': sens, 'surface': surf, 'average': avg}
    return results


def surface_fluxes(times, heat, wet, env, p):
    """表面热通量（W/m²，负为吸热）与有效水分通量（以 1e-6 为单位）"""
    T = heat.at(times)
    C = wet.at(times)
    qT = p.h * (T[-1] - env.value(times, 'T'))
    qC = -p.hm * (C[-1] - env.value(times, 'C')) * 1e6
    return qT, qC


def fourier_numbers(times, p):
    """热 Fo 与用初始 D(C0) 估计的水分 Fo"""
    t = np.asarray(times)
    alpha = _alpha(p)
    D0 = float(diffusivity(p.C0, p))
    return t / 60.0, alpha * t / p.R ** 2, D0 * t / p.R ** 2


def dcurve(p):
    """问题一实际浓度范围内的 D(C) 曲线"""
    C = np.linspace(p.C0 * 0.5, p.C0, 200)
    return C, diffusivity(C, p)

"""问题1：温湿场计算。输入时间为秒，长度为米。"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.special import expi


def mesh(N=2560, R=0.02, L=0.25):
    x = np.linspace(0., 1., N + 1)
    r = R * (1 - np.sinh(2.5 * (1 - x)) / np.sinh(2.5))
    f = np.r_[0., (r[:-1] + r[1:]) / 2, R]
    return r, np.pi * L * np.diff(f**2), 2 * np.pi * L * f


def check_environment(env, end):
    env = np.asarray(env, dtype=float)
    if (env.ndim != 2 or env.shape[1] != 3
            or not np.isfinite(env).all() or end <= 0
            or env[0, 0] != 0 or env[-1, 0] < end
            or np.any(np.diff(env[:, 0]) <= 0)
            or np.any(env[:, 1] <= -273.15)
            or np.any(env[:, 2] <= 0)):
        raise ValueError('环境数组的维数、单位、正性或时间覆盖错误')
    return env


# BEGIN Q1_KERNEL
def diffusivity(C):
    if not np.isfinite(C).all() or np.any(C <= 0):
        raise ValueError('含水率非正或非有限，不进行裁剪')
    return 7e-9 * np.exp(-0.89 / C)


def potential_difference(right, left):
    # 两点含水率接近时用展开式，避免两个近似数相减
    diffusivity(right)
    diffusivity(left)
    d, c = right - left, (right + left) / 2
    close = abs(d) <= 1e-6 * np.maximum(1., abs(c))
    psi = lambda u: 7e-9 * (u * np.exp(-0.89/u)
                            + 0.89 * expi(-0.89/u))
    out = np.empty_like(c)
    out[~close] = psi(right[~close]) - psi(left[~close])
    u, v = c[close], d[close]
    D = diffusivity(u)
    D2 = D * (0.89**2 / u**4 - 2 * 0.89 / u**3)
    out[close] = D * v + D2 * v**3 / 24
    return out


def operator(r, V, S, env, field):
    heat = field == 'T'
    capacity = V * (820. * 2600. if heat else 1.)
    exchange = S[-1] * (25. if heat else 8e-7)
    factor = S[1:-1] / np.diff(r)

    def rhs(t, u):
        ambient = np.interp(t, env[:, 0], env[:, 1 if heat else 2])
        if heat:
            ambient += 273.15
        F = np.zeros(len(u) + 1)  # 轴心通量保持为零
        delta = 0.36 * np.diff(u) if heat else potential_difference(
            u[1:], u[:-1])
        F[1:-1] = -factor * delta
        F[-1] = exchange * (u[-1] - ambient)  # 表面 Robin 条件
        return -np.diff(F) / capacity

    def jac(t, u):
        D = np.full(len(u), 0.36) if heat else diffusivity(u)
        diagonal = np.zeros(len(u))
        diagonal[:-1] -= factor * D[:-1] / capacity[:-1]
        diagonal[1:] -= factor * D[1:] / capacity[1:]
        diagonal[-1] -= exchange / capacity[-1]
        return diags([factor * D[:-1] / capacity[1:], diagonal,
                      factor * D[1:] / capacity[:-1]],
                     [-1, 0, 1], format='csc')
    return rhs, jac


def solve(env, N=2560, end=1800., sample_s=1.):
    env = check_environment(env, end)
    r, V, S = mesh(N)
    shijian = np.unique(np.r_[np.arange(0., end, sample_s), end])
    fenduan_shijian = np.r_[env[env[:, 0] < end, 0], end]
    fields = []
    for field, initial in [('T', 301.15), ('C', 2.55)]:
        rhs, jac = operator(r, V, S, env, field)
        zhuangtai = np.full(N + 1, initial)
        jieguo = np.empty((len(shijian), N + 1))
        for a, b in zip(fenduan_shijian[:-1], fenduan_shijian[1:]):
            solution = solve_ivp(rhs, (a, b), zhuangtai, method='BDF',
                            jac=jac, rtol=1e-10, atol=1e-12,
                            dense_output=True)
            if not solution.success:
                raise RuntimeError(solution.message)
            caiyang_mask = (shijian >= a) & (shijian <= b)
            jieguo[caiyang_mask] = solution.sol(shijian[caiyang_mask]).T
            zhuangtai = solution.y[:, -1]  # 下一段继承末态，不重置初态
        fields.append(jieguo - 273.15 if field == 'T' else jieguo)
    return dict(t=shijian, r=r, T=fields[0], C=fields[1])
# END Q1_KERNEL

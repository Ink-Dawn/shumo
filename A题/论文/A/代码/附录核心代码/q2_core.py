"""问题2：温湿场计算。输入时间为秒，长度为米。"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags, kron


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


# BEGIN Q2_KERNEL
def sparsity(N):
    G = diags([np.ones(N), np.ones(N + 1), np.ones(N)],
              [-1, 0, 1], format='csr')
    return kron(G, np.ones((2, 2)), format='csc')


def make_rhs(r, V, S, boundary):
    # 用五点 Gauss 积分计算界面水分通量
    x, w = np.polynomial.legendre.leggauss(5)
    q, w = (x + 1) / 2, w / 2

    def rhs(t, y):
        T, C = y[::2], y[1::2]  # 按 T0、C0、T1、C1 的顺序存放
        if not np.isfinite(y).all() or min(C) <= 0 or min(T) <= -273.15:
            raise ValueError('非物理试探态，不进行裁剪')
        B = (650. + 128.*C) * (1450. + 2736.*C/(C + 1))
        k = 0.21 + 0.38*C/(C + 1)
        k_interface = 2*k[:-1]*k[1:] / (k[:-1] + k[1:])
        temperature_K = (T[:-1] + T[1:])/2 + 273.15
        d = np.diff(C)
        u = C[:-1][None, :] + q[:, None]*d[None, :]
        potential = d * np.sum(w[:, None]*np.exp(-0.45/u), axis=0)
        heat_flux, moisture_flux = np.zeros(len(T) + 1), np.zeros(len(C) + 1)
        heat_flux[1:-1] = -S[1:-1]*k_interface*np.diff(T)/np.diff(r)
        moisture_flux[1:-1] = (-S[1:-1]*2.4e-3*np.exp(-3850/temperature_K)
                     * potential/np.diff(r))
        ambient_T, Ce = boundary(t)
        heat_flux[-1], moisture_flux[-1] = S[-1]*25*(T[-1]-ambient_T), S[-1]*8e-7*(C[-1]-Ce)
        state_rate = np.empty_like(y)
        state_rate[::2] = -np.diff(heat_flux)/(B*V)  # 净热流除以当前热容和体积
        state_rate[1::2] = -np.diff(moisture_flux)/V
        return state_rate
    return rhs


def solve(env, N=2560, end=10800., sample_s=1.):
    env = check_environment(env, end)
    r, V, S = mesh(N)
    boundary = lambda t: (np.interp(t, env[:, 0], env[:, 1]),
                           np.interp(t, env[:, 0], env[:, 2]))
    rhs = make_rhs(r, V, S, boundary)
    shijian = np.unique(np.r_[np.arange(0., end, sample_s), end])
    jieguo = np.empty((len(shijian), 2*(N + 1)))
    y = np.tile([28., 2.55], N + 1)
    fenduan_shijian = np.r_[env[env[:, 0] < end, 0], end]
    for a, b in zip(fenduan_shijian[:-1], fenduan_shijian[1:]):
        solution = solve_ivp(rhs, (a, b), y, method='BDF',
                        rtol=1e-13, atol=np.tile([1e-14, 1e-15], N + 1),
                        max_step=5., jac_sparsity=sparsity(N),
                        dense_output=True)
        if not solution.success:
            raise RuntimeError(solution.message)
        caiyang_mask = (shijian >= a) & (shijian <= b)
        jieguo[caiyang_mask] = solution.sol(shijian[caiyang_mask]).T
        y = solution.y[:, -1]
    return dict(t=shijian, r=r, T=jieguo[:, ::2], C=jieguo[:, 1::2])
# END Q2_KERNEL

"""问题3：温湿场计算。输入时间为秒，长度为米。"""
import numpy as np
from scipy.integrate import BDF, solve_ivp
from scipy.optimize import brentq
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


# BEGIN EVENT_KERNEL
def platform(env):
    # 取最后一小时的时间平均值，作为后续环境
    t = env[:, 0]
    x = np.r_[t[-1]-3600., t[t > t[-1]-3600.]]
    return np.array([np.sum(np.diff(x)*(np.interp(x, t, env[:, j])[:-1]
                    + np.interp(x, t, env[:, j])[1:])/2)/3600.
                     for j in (1, 2)])


def event_integrate(factory, fenduan_shijian, y, cpart, pattern, rtol, atol,
                    late_step=150., first_step=None, fine=False,
                    xtol=1e-7, bracket_tol=1e-5, post_s=10., sample_s=60.):
    # 每段分别设置环境，4 h 前用观测值，之后用平台值
    shijian, zhuangtai_jilu, event = [0.], [y.copy()], None
    for a, b in zip(fenduan_shijian[:-1], fenduan_shijian[1:]):
        step = 10. if a < 14400. else late_step
        first = None if first_step is None else min(first_step, b - a, step)
        rhs = factory(a)
        solver = BDF(rhs, a, y, b, rtol=rtol, atol=atol,
                     max_step=step, first_step=first, jac_sparsity=pattern)
        while solver.status == 'running':
            left = solver.t
            message = solver.step()
            if solver.status == 'failed':
                raise RuntimeError(message)
            if not np.isfinite(solver.y).all() or min(solver.y[cpart]) <= 0:
                raise ValueError('接受步含非物理状态')
            right, dense = solver.t, solver.dense_output()
            # 每步检查几个中间时刻，避免漏掉首次达标
            jiance_shijian = np.linspace(left, right, 5)
            threshold_gap = np.max(dense(jiance_shijian)[cpart], axis=0)-0.15
            if fine and np.min(threshold_gap) <= 0.02:
                n = max(2, int(np.ceil(right - left))+1)
                jiance_shijian = np.unique(np.r_[jiance_shijian, np.linspace(left, right, n)])
                threshold_gap = np.max(dense(jiance_shijian)[cpart], axis=0)-0.15
            crossings = np.flatnonzero((threshold_gap[:-1] >= 0) & (threshold_gap[1:] < 0))
            if crossings.size:
                i = crossings[0]  # 只取第一次由未达标变为达标的位置
                lower, upper = jiance_shijian[i], jiance_shijian[i+1]
                g = lambda t: float(np.max(dense(t)[cpart])-0.15)
                root = brentq(g, lower, upper, xtol=xtol,
                              rtol=4*np.finfo(float).eps)
                while upper - lower > bracket_tol:
                    middle = (lower+upper)/2
                    if g(middle) >= 0:
                        lower = middle
                    else:
                        upper = middle
                event = dict(t=root, g=g(root), bracket=(lower, upper))
                right = root
            caiyang_shijian = np.arange((np.floor((shijian[-1]+1e-7)/sample_s)+1)
                           * sample_s, right+1e-8, sample_s)
            caiyang_shijian = caiyang_shijian[caiyang_shijian <= right]  # 只保存临界时刻之前的数据
            if len(caiyang_shijian):
                shijian.extend(caiyang_shijian.tolist())
                zhuangtai_jilu.extend(dense(caiyang_shijian).T)
            if event:
                y = dense(right).copy()
                if right > shijian[-1]+1e-7:
                    shijian.append(right)
                    zhuangtai_jilu.append(y.copy())
                break
        if event:
            break
        y = solver.y.copy()
    if not event and fenduan_shijian[-1] > shijian[-1]+1e-7:
        shijian.append(float(fenduan_shijian[-1]))
        zhuangtai_jilu.append(y.copy())
    if event:
        # 从临界状态再算一小段，检查含水率是否继续下降
        root = event['t']
        boundaries = np.r_[root, fenduan_shijian[(fenduan_shijian > root)
                                      & (fenduan_shijian < root + post_s)], root + post_s]
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            first = None if first_step is None else min(first_step, b - a)
            solution = solve_ivp(factory(a), (a, b), y, method='BDF',
                            rtol=rtol, atol=atol,
                            max_step=1., first_step=first, jac_sparsity=pattern)
            if not solution.success:
                raise RuntimeError(solution.message)
            y = solution.y[:, -1]
        event.update(post_t=root + post_s, post_state=y,
                     strict_post=bool(np.max(y[cpart]) < 0.15))
    return np.asarray(shijian), np.asarray(zhuangtai_jilu), event
# END EVENT_KERNEL


# BEGIN Q3_DRIVER
def solve(env, N=640, end=120*3600., sample_s=60.):
    env = check_environment(env, 14400.)
    if env[-1, 0] != 14400. or end <= 0:
        raise ValueError('问题三要求完整前 4 h 环境，积分终点应为正')
    r, V, S = mesh(N)
    pingtai_zhi = platform(env)

    def factory(a):
        if a < 14400.:
            boundary = lambda t: (np.interp(t, env[:, 0], env[:, 1]),
                                   np.interp(t, env[:, 0], env[:, 2]))
        else:
            boundary = lambda t: pingtai_zhi
        return make_rhs(r, V, S, boundary)

    fenduan_shijian = np.unique(np.r_[env[:, 0], np.arange(21600., end, 21600.), end])
    fenduan_shijian = fenduan_shijian[fenduan_shijian <= end]
    y = np.tile([28., 2.55], N + 1)
    shijian, zhuangtai_jilu, event = event_integrate(
        factory, fenduan_shijian, y, slice(1, None, 2), sparsity(N),
        1e-10, np.tile([1e-11, 1e-12], N + 1), sample_s=sample_s)
    return dict(t=shijian, r=r, T=zhuangtai_jilu[:, ::2], C=zhuangtai_jilu[:, 1::2],
                event=event, plateau=pingtai_zhi)
# END Q3_DRIVER

"""问题4：温湿场计算。输入时间为秒，长度为米。"""
import numpy as np
from scipy.integrate import BDF, solve_ivp
from scipy.optimize import brentq
from scipy.sparse import diags, bmat
from scipy.interpolate import PchipInterpolator


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


# BEGIN Q4_KERNEL
def material_mesh(N):
    xi = np.linspace(0., 1., N + 1)
    faces = np.r_[0., (xi[:-1]+xi[1:])/2, 1.]
    return xi, faces


def sparsity(N):
    G = diags([np.ones(N), np.ones(N + 1), np.ones(N)],
              [-1, 0, 1], format='csc')
    return bmat([[G, G], [G, G]], format='csc').astype(bool)


def make_rhs(xi, faces, radius, boundary):
    # 用五点 Gauss 积分计算界面水分通量
    x, w = np.polynomial.legendre.leggauss(5)
    q, w = (x+1)/2, w/2
    factor = 2*np.pi*0.25*faces[1:-1]/np.diff(xi)

    def rhs(t, y):
        n = len(xi)
        T, C = y[:n], y[n:]  # 前半段存温度，后半段存含水率
        if not np.isfinite(y).all() or min(C) <= 0 or min(T) <= -273.15:
            raise ValueError('非物理试探态，不进行裁剪')
        R = float(radius(t))
        if not np.isfinite(R) or R <= 0:
            raise ValueError('半径必须为有限正数')
        V = np.pi*0.25*R**2*np.diff(faces**2)
        area = 2*np.pi*0.25*R
        B = (760.+90.*C)*(1850.+2150.*C/(C + 1))
        k = 0.12+0.20*C/(C + 1)
        k_interface = 2*k[:-1]*k[1:]/(k[:-1]+k[1:])
        temperature_K, d = (T[:-1]+T[1:])/2+273.15, np.diff(C)
        u = C[:-1][None, :]+q[:, None]*d[None, :]
        potential = d*np.sum(w[:, None]*np.exp(-0.30/u), axis=0)
        heat_flux, moisture_flux = np.zeros(n+1), np.zeros(n+1)
        heat_flux[1:-1] = -factor*k_interface*np.diff(T)
        moisture_flux[1:-1] = -factor*4.2e-4*np.exp(-3850/temperature_K)*potential
        ambient_T, Ce = boundary(t)
        heat_flux[-1], moisture_flux[-1] = area*25*(T[-1]-ambient_T), area*8e-7*(C[-1]-Ce)
        # 材料坐标随药材收缩，半径变化已计入面积和体积
        return np.r_[-np.diff(heat_flux)/(B*V), -np.diff(moisture_flux)/V]
    return rhs


def physical_values(C, xi, R, targets):
    targets = np.asarray(targets, dtype=float)
    if np.any(targets < 0) or not np.isfinite(targets).all():
        raise ValueError('实际距离非法')
    jieguo = np.full(targets.shape, np.nan)
    inside = targets <= R+1e-12
    jieguo[inside] = np.interp(np.minimum(targets[inside], R)/R, xi, C)
    return jieguo  # 超出当前半径的位置记为 NaN


def solve(env, radius_data, N=5120, end=72*3600., sample_s=60.):
    env = check_environment(env, 14400.)
    data = np.asarray(radius_data, dtype=float)  # 两列：时间 s、半径 m
    if (env[-1, 0] != 14400. or end <= 0
            or data.ndim != 2 or data.shape[1] != 2
            or not np.isfinite(data).all() or data[0, 0] != 0
            or not np.isclose(data[0, 1], 0.02)
            or np.any(np.diff(data[:, 0]) <= 0)
            or np.any(data[:, 1] <= 0) or np.any(np.diff(data[:, 1]) > 0)):
        raise ValueError('环境或半径数据不符合题设，半径须以 m 输入')
    curve = PchipInterpolator(data[:, 0], data[:, 1], extrapolate=False)
    radius = lambda t: curve(np.minimum(t, data[-1, 0]))
    xi, faces = material_mesh(N)
    pingtai_zhi = platform(env)

    def factory(a):
        if a < 14400.:
            boundary = lambda t: (np.interp(t, env[:, 0], env[:, 1]),
                                   np.interp(t, env[:, 0], env[:, 2]))
        else:
            boundary = lambda t: pingtai_zhi
        return make_rhs(xi, faces, radius, boundary)

    fenduan_shijian = np.unique(np.r_[env[:, 0], np.arange(21600., end, 21600.),
                            data[-1, 0], end])
    fenduan_shijian = fenduan_shijian[fenduan_shijian <= end]
    y = np.r_[np.full(N + 1, 28.), np.full(N + 1, 2.55)]
    shijian, zhuangtai_jilu, event = event_integrate(
        factory, fenduan_shijian, y, slice(N + 1, None), sparsity(N),
        1e-9, np.r_[np.full(N + 1, 1e-10), np.full(N + 1, 1e-12)],
        late_step=300., first_step=0.001, fine=True,
        xtol=1e-5, bracket_tol=2e-5, post_s=120., sample_s=sample_s)
    return dict(t=shijian, xi=xi, R=radius(shijian), T=zhuangtai_jilu[:, :N + 1],
                C=zhuangtai_jilu[:, N + 1:], event=event, plateau=pingtai_zhi)
# END Q4_KERNEL

"""中文科研图：PNG 与矢量 SVG 同时输出。

共 12 组：01—04 为核心结果，05—08、10—12 为验证与诊断，09 为题定结果表。
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from .metrics import KEY_RADII_CM, KEY_TIMES, loss_depth
from .model import T_END, diffusivity
from .verification import (balance_history, dcurve, fourier_numbers,
                           manufactured_errors, sensitivity,
                           spatial_convergence, surface_fluxes,
                           surface_short_time, temporal_convergence)

_FONT_CANDIDATES = (
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/simhei.ttf',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
)

_COLORS = ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', '#949494', '#56B4E9']


def style():
    """挂上系统里的中文字体，并统一一套干净、印刷友好的样式。"""
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update({
        'axes.unicode_minus': False,
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'axes.prop_cycle': matplotlib.cycler('color', _COLORS),
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 120,
        'savefig.dpi': 220,
        'svg.fonttype': 'path',
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '-',
        'grid.linewidth': 0.4,
        'lines.linewidth': 1.6,
        'lines.markersize': 5,
    })


def save(fig, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{name}.png', bbox_inches='tight')
    fig.savefig(out / f'{name}.svg', bbox_inches='tight')
    plt.close(fig)


def _plot_01(out, env, p, grid, heat, wet):
    time = np.linspace(0.0, T_END, 301)
    r = grid.r * 100.0
    T = heat.at(time)
    C = wet.at(time)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    for ax, z, title, cmap, unit in zip(
            axs, [T, C], ['温度时空分布', '水分时空分布'],
            ['inferno', 'viridis'], ['℃', 'kg/kg']):
        im = ax.pcolormesh(time / 60, r, z, shading='auto', cmap=cmap, rasterized=True)
        fig.colorbar(im, ax=ax, label=unit)
        ax.set(xlabel='时间 / min', ylabel='到中心距离 / cm', title=title)
    fig.suptitle('附件1环境激励下的径向响应')
    save(fig, out, '01_温湿时空分布')


def _plot_02(out, p, grid, heat, wet):
    r = grid.r * 100.0
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    for t in (100, 300, 600, 1200, 1800):
        axs[0].plot(r, heat.at(t), label=f'{t} s')
        axs[1].plot(r, wet.at(t), label=f'{t} s')
    for ax, label in zip(axs, ['温度 / ℃', '干基含水率 / (kg/kg)']):
        ax.set(xlabel='到中心距离 / cm', ylabel=label)
        ax.legend(ncol=2, fontsize=8)
    fig.suptitle('不同时刻的径向剖面')
    save(fig, out, '02_径向剖面')


def _plot_03(out, env, p, grid, heat, wet):
    time = np.linspace(0.0, T_END, 301)
    T = heat.at(time)
    C = wet.at(time)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    for ax, field, kind in zip(axs, [T, C], ['T', 'C']):
        ax.plot(time / 60, field[0], label='中心')
        ax.plot(time / 60, field[-1], label='表面')
        ax.plot(time / 60, env.value(time, kind), '--', label='环境', color='#949494')
        ax.set(xlabel='时间 / min',
               ylabel='温度 / ℃' if kind == 'T' else '干基含水率 / (kg/kg)')
        ax.legend()
    fig.suptitle('环境变化与内部响应')
    save(fig, out, '03_环境与响应')


def _plot_04(out, p, grid, wet):
    time = np.linspace(0.0, T_END, 301)
    r = grid.r * 100.0
    C = wet.at(time)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    for eta in (0.05, 0.10, 0.20):
        depths = [loss_depth(grid, C[:, j], p.C0, eta) * 1000.0 for j in range(len(time))]
        axs[0].plot(time / 60, depths, label=f'{eta:.0%} 失水阈值')
    axs[0].set(xlabel='时间 / min', ylabel='表面连通失水深度 / mm')
    axs[0].legend()
    axs[1].plot(r, diffusivity(C[:, -1], p) / 1e-9)
    axs[1].set(xlabel='到中心距离 / cm',
               ylabel=r'扩散系数 / ($10^{-9}$ m²/s)', title=f'{T_END:g} s 的 D(C) 分布')
    fig.suptitle('表层失水与浓度依赖扩散')
    save(fig, out, '04_失水深度与扩散系数')


def _plot_05(out, env, p, grid, heat, wet):
    times = np.linspace(0.0, T_END, 901)
    hc = balance_history(heat, grid, env, p, 'T', times)
    wc = balance_history(wet, grid, env, p, 'C', times)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    axs[0].plot(hc['time'], hc['relative'])
    axs[0].set(xlabel='时间 / s', ylabel='相对收支残差', title='热量增量平衡')
    axs[1].plot(wc['time'], wc['relative'])
    axs[1].set(xlabel='时间 / s', ylabel='相对收支残差', title='含水率积分平衡')
    fig.suptitle('独立边界积分检查 | 附件1.xlsx')
    save(fig, out, '05_边界积分守恒检查')


def _plot_06(out, env, p):
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2), layout='constrained')

    # 空间收敛
    sp = spatial_convergence(env, p)
    Ns = [x['coarse'] for x in sp]
    errs = [x['diff'] for x in sp]
    axs[0].loglog(Ns, errs, 'o-', color=_COLORS[0])
    # 参考斜率 -2
    if len(Ns) >= 2:
        ref = errs[-1] * (np.array(Ns) / Ns[-1]) ** (-2.0)
        axs[0].loglog(Ns, ref, '--', color='#949494', label='斜率 -2')
    axs[0].set(xlabel='粗网格区间数 N', ylabel='最大水分分解差', title='全时段空间收敛')
    axs[0].legend()

    # 制造解收敛
    mms = manufactured_errors(p)
    Nm = [x['N'] for x in mms]
    em = [x['error'] for x in mms]
    axs[1].loglog(Nm, em, 's-', color=_COLORS[1])
    if len(Nm) >= 2:
        ref = em[-1] * (np.array(Nm) / Nm[-1]) ** (-2.0)
        axs[1].loglog(Nm, ref, '--', color='#949494', label='斜率 -2')
    axs[1].set(xlabel='N', ylabel='制造解最大误差', title='独立非线性制造解')
    axs[1].legend()

    # 时间收敛
    tc = temporal_convergence(env, p)
    tols = [x['rtol'] for x in tc]
    te = [x['error'] for x in tc]
    axs[2].loglog(tols, te, 'd-', color=_COLORS[2])
    if len(tols) >= 2:
        ref = te[-1] * (np.array(tols) / tols[-1]) ** 1.0
        axs[2].loglog(tols, ref, '--', color='#949494', label='斜率 1')
    axs[2].set(xlabel='相对容差', ylabel='相对严格时间解的最大差', title='同网格时间收敛')
    axs[2].legend()

    fig.suptitle('数值验证 | 附件1.xlsx')
    save(fig, out, '06_收敛与制造解验证')


def _plot_07(out, env, p, grid, wet):
    t = np.linspace(0.5, 10.0, 200)
    sqrt_t = np.sqrt(t)
    Ce = env.value(0.0, 'C')
    full_drop = p.C0 - wet.at(t)[-1]
    approx = surface_short_time(t, p, Ce)
    approx_drop = p.C0 - approx
    diff = approx_drop - full_drop

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    axs[0].plot(sqrt_t, full_drop, label='完整非线性圆柱解')
    axs[0].plot(sqrt_t, approx_drop, '--', label='常D半无限平面近似')
    axs[0].set(xlabel=r'$\sqrt{t}$ / s$^{1/2}$',
               ylabel='表面含水率下降 / (kg/kg)', title='表面含水率下降 vs 时间平方根')
    axs[0].legend()
    axs[1].plot(t, diff)
    axs[1].set(xlabel='时间 / s', ylabel='近似减完整解 / (kg/kg)',
               title='0—10 s 的近似偏差')
    fig.suptitle('短时近似仅用于机制解释 | 附件1，径向独立模型')
    save(fig, out, '07_短时表面近似')


def _plot_08(out, env, p, grid, wet):
    sens = sensitivity(env, p, grid, wet)
    r = grid.r * 100.0
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    names = []
    surfaces = []
    averages = []
    for name, data in sens.items():
        axs[0].plot(r, data['profile'], label=name)
        names.append(name)
        surfaces.append(data['surface'])
        averages.append(data['average'])
    axs[0].axhline(0, color='#949494', linewidth=0.8)
    axs[0].set(xlabel='到中心距离 / cm', ylabel='归一化浓度响应',
               title='1800 s，参数±2%')
    axs[0].legend()

    x = np.arange(len(names))
    width = 0.35
    axs[1].bar(x - width / 2, surfaces, width, label='表面')
    axs[1].bar(x + width / 2, averages, width, label='体积平均')
    axs[1].axhline(0, color='#949494', linewidth=0.8)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([n.split()[0] for n in names])
    axs[1].set(ylabel='归一化响应', title='内部补给与表面交换作用不同')
    axs[1].legend()

    fig.suptitle('局部灵敏度，不是参数概率区间 | 附件1')
    save(fig, out, '08_参数灵敏度')


def _plot_09(out, p, grid, heat, wet):
    fig, axs = plt.subplots(2, 1, figsize=(9, 7), layout='constrained')
    radii = KEY_RADII_CM / 100.0
    for ax, traj, title in zip(axs, [heat, wet], ['表1  温度 / ℃', '表2  干基含水率 / (kg/kg)']):
        vals = np.array([np.interp(radii, grid.r, traj.at(t)) for t in KEY_TIMES])
        ax.axis('off')
        ax.set_title(title, pad=10)
        table = ax.table(
            cellText=[[f'{t:.0f}', *[f'{v:.4f}' for v in row]]
                      for t, row in zip(KEY_TIMES, vals)],
            colLabels=['时间/s', *[f'{x:g} cm' for x in KEY_RADII_CM]],
            loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.7)
        for (i, _), cell in table.get_celld().items():
            cell.set_edgecolor('#c7d3dc')
            if i == 0:
                cell.set_facecolor('#245979')
                cell.set_text_props(color='white', weight='bold')
    fig.suptitle('题定时刻与位置的求解结果')
    save(fig, out, '09_题定结果表')


def _plot_10(out, env):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5), layout='constrained')
    t_dense = np.linspace(0.0, 300.0, 301)
    t_full_h = env.t / 3600.0
    q1_end = 0.5

    # 前 300 秒局部
    axs[0, 0].plot(t_dense, env.value(t_dense, 'T'), label='线性查询函数，每1 s取值')
    axs[0, 0].scatter(env.t[env.t <= 300], env.T[env.t <= 300], color=_COLORS[1],
                      zorder=5, label='原始实测，每60 s一点')
    axs[0, 0].set(xlabel='时间 / s', ylabel='环境温度 / ℃', title='前300秒输入对齐')
    axs[0, 0].legend(fontsize=8)

    axs[0, 1].plot(t_dense, env.value(t_dense, 'C'), label='线性查询函数，每1 s取值')
    axs[0, 1].scatter(env.t[env.t <= 300], env.C[env.t <= 300], color=_COLORS[1],
                      zorder=5, label='原始实测，每60 s一点')
    axs[0, 1].set(xlabel='时间 / s', ylabel='环境有效水分 / (kg/kg)', title='前300秒输入对齐')
    axs[0, 1].legend(fontsize=8)

    # 全部范围
    axs[1, 0].plot(t_full_h, env.T)
    axs[1, 0].axvspan(0, q1_end, color='#F0E6CC', alpha=0.6, label='问题一范围 0—0.5 h')
    axs[1, 0].set(xlabel='时间 / h', ylabel='环境温度 / ℃', title='原始附件1与本问使用范围')
    axs[1, 0].legend(fontsize=8)

    axs[1, 1].plot(t_full_h, env.C)
    axs[1, 1].axvspan(0, q1_end, color='#F0E6CC', alpha=0.6, label='问题一范围 0—0.5 h')
    axs[1, 1].set(xlabel='时间 / h', ylabel='环境有效水分 / (kg/kg)', title='原始附件1与本问使用范围')
    axs[1, 1].legend(fontsize=8)

    fig.suptitle('输入数据诊断 | 原节点插值一致不代表测点之间无误差')
    save(fig, out, '10_输入数据诊断')


def _plot_11(out, env, p, grid, heat, wet):
    t = np.linspace(0.0, T_END, 301)
    qT, qC = surface_fluxes(t, heat, wet, env, p)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    axs[0].plot(t / 60.0, qT)
    axs[0].axhline(0, color='#949494', linewidth=0.8)
    axs[0].set(xlabel='时间 / min', ylabel='外向热通量 / (W/m²)', title='负值代表药材吸热')
    axs[1].plot(t / 60.0, qC)
    axs[1].set(xlabel='时间 / min',
               ylabel=r'有效外向水分通量 / [$10^{-6}$ (kg/kg)·m/s]',
               title='未乘干物质密度，不是质量通量')
    fig.suptitle('表面Robin通量诊断 | 用实际结果判断趋势')
    save(fig, out, '11_表面Robin通量诊断')


def _plot_12(out, p):
    C, D = dcurve(p)
    t = np.linspace(0.0, T_END, 301)
    _, FoT, FoC = fourier_numbers(t, p)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    axs[0].plot(C, D / 1e-9)
    axs[0].set(xlabel='药材干基含水率 / (kg/kg)',
               ylabel=r'D / ($10^{-9}$ m²/s)', title='仅扫描问题一实际浓度范围')
    axs[1].plot(t / 60.0, FoT, label='热Fo')
    axs[1].plot(t / 60.0, FoC, label='水分Fo（初始D）')
    axs[1].set(xlabel='时间 / min', ylabel='Fo', title='Fo衡量径向扩散发展程度')
    axs[1].legend()
    fig.suptitle('问题一物性与时间尺度 | 不混入后续题的温度相关D')
    save(fig, out, '12_物性与时间尺度')


def field_plots(out, env, p, grid, heat, wet):
    """画全部 12 组中文图。"""
    style()
    _plot_01(out, env, p, grid, heat, wet)
    _plot_02(out, p, grid, heat, wet)
    _plot_03(out, env, p, grid, heat, wet)
    _plot_04(out, p, grid, wet)
    _plot_05(out, env, p, grid, heat, wet)
    _plot_06(out, env, p)
    _plot_07(out, env, p, grid, wet)
    _plot_08(out, env, p, grid, wet)
    _plot_09(out, p, grid, heat, wet)
    _plot_10(out, env)
    _plot_11(out, env, p, grid, heat, wet)
    _plot_12(out, p)

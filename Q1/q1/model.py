"""
问题一模型：固定半径圆柱内的径向导热与非线性水分扩散。

两场独立求解，单位轴向长度：
    rho*cp*dT/dt = (1/r) * d/dr( r*k*dT/dr )
    dC/dt        = (1/r) * d/dr( r*D(C)*dC/dr ),   D(C) = A*exp(-a/C)

中心对称，表面为 Robin 边界：
    -k*dT/dr  = h *(T_s - T_inf)
    -D*dC/dr  = hm*(C_s - C_e)
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import diags

# 问题一的讨论范围
T_END = 1800.0


@dataclass(frozen=True)
class Parameters:
    """参数初始化。"""

    R: float = 0.02        # 圆柱半径
    rho: float = 820.0     # 密度
    cp: float = 2600.0     # 比热容
    k: float = 0.36        # 导热系数
    h: float = 25.0        # 表面对流换热系数
    hm: float = 8e-7       # 表面质交换系数
    A: float = 7e-9        # 水分扩散系数前因子
    a: float = 0.89        # 水分扩散浓度指数
    T0: float = 28.0       # 初始温度, 摄氏度
    C0: float = 2.55       # 初始干基含水率


class Grid:
    """径向节点与有限体积几何量（体积按单位轴向长度计）"""

    def __init__(self, r, faces, V, g, beta):
        self.r = r          # 节点半径
        self.faces = faces  # 控制体界面半径
        self.V = V          # 控制体体积
        self.g = g          # 界面因子
        self.beta = beta

    @classmethod
    def build(cls, N=2560, R=0.02, beta=2.5):
        """生成表面加密网格"""
        x = np.linspace(0.0, 1.0, int(N) + 1)
        if beta < 1e-10:
            r = R * x
        else:
            r = R * (1.0 - np.sinh(beta * (1.0 - x)) / np.sinh(beta))
        faces = np.concatenate([[0.0], (r[1:] + r[:-1]) / 2.0, [R]])
        V = np.pi * np.diff(faces ** 2)
        g = 2.0 * np.pi * faces[1:-1] / np.diff(r)
        return cls(r, faces, V, g, beta)


@lru_cache(maxsize=None)
def _gauss(n):
    """[0,1] 上的 n 点 Gauss-Legendre 节点与权重。"""
    x, w = leggauss(n)
    return (x + 1.0) / 2.0, w / 2.0


def diffusivity(C, p):
    """
    水分扩散系数 D(C)
    隐式迭代的试探态可能出现 C<=0，此时返回 0 以避免指数溢出
    物理上非正的解由求解器直接判失败，不做结果截断
    """
    C = np.asarray(C, dtype=float)
    if p.a == 0.0:                      # 固定 D 的退化工况
        return np.full_like(C, p.A)
    D = np.zeros_like(C)
    pos = C > 0
    D[pos] = p.A * np.exp(-p.a / C[pos])
    return D


def potential_difference(left, right, p, order=8):
    """
    计算Kirchhoff 势差
    在浓度区间上做 Gauss 积分，避免两个接近的势函数值直接相减。
    """
    s, w = _gauss(order)
    dc = np.asarray(right) - np.asarray(left)
    samples = np.asarray(left)[..., None] + dc[..., None] * s
    return dc * (diffusivity(samples, p) @ w)


class Operator:
    """
    单场（T 或 C）的守恒半离散右端与稀疏 Jacobian。
    温度场内部变量取 theta = T - T0，输出时再加回 T0。
    """

    def __init__(self, grid, p, kind, env, flux='kirchhoff', order=8,
                 source=None, boundary=None):
        self.grid = grid
        self.p = p
        self.kind = kind # 'T' 或 'C'
        self.env = env
        self.flux = flux
        self.order = order
        self.source = source
        self.boundary = boundary
        self.mass = grid.V * (p.rho * p.cp if kind == 'T' else 1.0)
        self.B = 2.0 * np.pi * p.R * (p.h if kind == 'T' else p.hm)

    def ambient(self, t):
        """环境基准值（温度场扣除 T0）；制造解可覆盖边界函数"""
        if self.boundary is not None:
            return self.boundary(t)
        v = self.env.value(t, self.kind)
        return v - self.p.T0 if self.kind == 'T' else v

    def face_potential(self, u):
        """界面上的传导通量因子（未乘界面面积）。"""
        if self.kind == 'T':
            return self.p.k * np.diff(u)
        if self.flux == 'midpoint':
            return diffusivity((u[:-1] + u[1:]) / 2.0, self.p) * np.diff(u)
        return potential_difference(u[:-1], u[1:], self.p, self.order)

    def rhs(self, t, u):
        """du/dt = 通量散度 / 质量 + 表面 Robin 项 + 体积源（如有）。"""
        q = self.grid.g * self.face_potential(u)   # 界面外向通量的相反数
        du = np.zeros_like(u)
        du[:-1] += q
        du[1:] -= q
        du[-1] -= self.B * (u[-1] - self.ambient(t))
        du = du / self.mass
        if self.source is not None:
            du = du + self.source(t, self.grid.r)
        return du

    def jac(self, t, u):
        """解析 Jacobian（三对角）。"""
        g, m = self.grid.g, self.mass
        if self.kind == 'T':
            dl = dr = np.full_like(g, self.p.k)
        elif self.flux == 'kirchhoff':
            # 势差对两端浓度的导数即端点处的 D
            dl, dr = diffusivity(u[:-1], self.p), diffusivity(u[1:], self.p)
        else:
            mid = (u[:-1] + u[1:]) / 2.0
            D = diffusivity(mid, self.p)
            dD = np.zeros_like(mid)
            pos = mid > 0
            dD[pos] = D[pos] * self.p.a / mid[pos] ** 2
            dl, dr = D - 0.5 * dD * np.diff(u), D + 0.5 * dD * np.diff(u)
        main = np.zeros_like(u)
        main[:-1] -= g * dl / m[:-1]
        main[1:] -= g * dr / m[1:]
        main[-1] -= self.B / m[-1]
        return diags([g * dl / m[1:], main, g * dr / m[:-1]],
                     [-1, 0, 1], format='csc')

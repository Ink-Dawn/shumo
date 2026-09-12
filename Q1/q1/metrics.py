"""由解场导出的特征量"""
import numpy as np

# 时刻与位置
KEY_TIMES = np.array([100, 300, 600, 900, 1200, 1500, 1800])
KEY_RADII_CM = np.array([0.0, 0.5, 1.0, 1.5, 2.0])


def loss_depth(grid, C, C0, eta=0.05):
    """
    表面连通失水层厚度
    从表面向内找到 C = (1-eta)*C0 的等值面，在线性插值出的交点处截断
    表面还没降到阈值时返回 nan，整根都已降到阈值以下时返回 R
    """
    threshold = (1.0 - eta) * C0
    if C[-1] > threshold:
        return np.nan
    above = np.flatnonzero(C > threshold)
    if above.size == 0:
        return grid.r[-1]
    j = above[-1]
    rc = grid.r[j] + (threshold - C[j]) / (C[j + 1] - C[j]) * (grid.r[j + 1] - grid.r[j])
    return grid.r[-1] - rc

"""附件环境数据的读取与插值。

附件前三列依次是：时间 s、温度 ℃、水分浓度 kg/kg，默认第 1 行为表头。

第三列题名是「水分浓度」，模型把它当作表面传质的环境有效值 Ce 使用：由题面
给定，不做湿基/干基再换算，也不当成严格标定的吸附平衡含水率。本问用到的
0—1800 s 内它是 0.01963—0.03307，比表面含水率 2.55 小两个量级，所以即使按
湿基折算成干基（最大 3.4%），对结果的影响也只有 4e-4 kg/kg 量级。

数据保持原样，不做平滑也不补点。
"""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Environment:
    t: np.ndarray          # 时间, s
    T: np.ndarray          # 环境温度, 摄氏度
    C: np.ndarray          # 环境水分浓度（作 Ce 用）, kg/kg
    source: str = ''       # 数据来源，写进运行记录

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float)
        self.T = np.asarray(self.T, dtype=float)
        self.C = np.asarray(self.C, dtype=float)
        if not (len(self.t) == len(self.T) == len(self.C)):
            raise ValueError('三列长度不一致')
        if np.any(np.diff(self.t) <= 0):
            raise ValueError('时间列必须严格递增')

    def value(self, t, kind):
        """线性插值查询环境值；超出数据范围直接报错，不外推。"""
        a = np.asarray(t, dtype=float)
        if a.min() < self.t[0] - 1e-9 or a.max() > self.t[-1] + 1e-9:
            raise ValueError('查询时间超出环境数据范围')
        return np.interp(a, self.t, self.T if kind == 'T' else self.C)

    @classmethod
    def load(cls, path, sheet=0, skiprows=1):
        path = Path(path)
        if path.suffix.lower() == '.csv':
            with path.open(encoding='utf-8-sig', newline='') as f:
                rows = list(csv.reader(f))
        elif path.suffix.lower() == '.xlsx':
            from openpyxl import load_workbook
            wb = load_workbook(path, read_only=True, data_only=True)
            ws = wb.worksheets[sheet] if isinstance(sheet, int) else wb[sheet]
            rows = list(ws.values)
            wb.close()
        else:
            raise ValueError(f'只认 .xlsx 或 UTF-8 的 .csv，拿到的是 {path.suffix}')

        data = []
        for i, row in enumerate(rows[skiprows:], start=skiprows + 1):
            if not row or row[0] in (None, ''):
                continue
            try:
                data.append([float(v) for v in row[:3]])
            except (TypeError, ValueError):
                raise ValueError(f'第 {i} 行解析不出「时间, 温度, 水分浓度」三列数值：{row[:3]}')
        if not data:
            raise ValueError(f'{path} 里没读到有效数据')
        arr = np.asarray(data)
        return cls(arr[:, 0], arr[:, 1], arr[:, 2], str(path))

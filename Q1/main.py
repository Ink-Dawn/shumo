"""问题一：圆柱径向导热与非线性水分扩散求解。

用法：
    python main.py
    python main.py --N 5120 --out outputs/加密网格
"""
import argparse
import sys
from pathlib import Path
from q1.data import Environment
from q1.model import Parameters
from q1.output import export_run
from q1.plots import field_plots
from q1.solver import solve_fields

# 设置默认路径
ROOT = Path(__file__).resolve().parent


def parse_sheet(text):
    try:
        return int(text)
    except ValueError:
        return text


def build_parser():
    ap = argparse.ArgumentParser(description='问题一求解：圆柱径向导热与非线性水分扩散')
    ap.add_argument('--input', default=str(ROOT / 'data' / '附件1.xlsx'),
                    help='环境数据文件（xlsx/csv）')
    ap.add_argument('--template', default=str(ROOT / 'data' / 'result1_模板.xlsx'),
                    help='result1 输出模板')
    ap.add_argument('--sheet', default='0', help='工作表名或序号')
    ap.add_argument('--skiprows', type=int, default=1, help='表头行数')
    ap.add_argument('--out', default=str(ROOT / 'outputs'), help='输出目录')
    ap.add_argument('--N', type=int, default=2560, help='径向区间数')
    ap.add_argument('--beta', type=float, default=2.5, help='sinh 表面加密强度，0 表示均匀网格')
    ap.add_argument('--rtol', type=float, default=1e-10, help='时间积分相对容差')
    ap.add_argument('--atol-T', dest='atol_T', type=float, default=1e-10, help='温度场绝对容差')
    ap.add_argument('--atol-C', dest='atol_C', type=float, default=1e-12, help='水分场绝对容差')
    ap.add_argument('--flux', choices=['kirchhoff', 'midpoint'], default='kirchhoff',
                    help='水分界面通量：Kirchhoff 势差或中点 D')
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.N < 4:
        raise SystemExit('--N 至少为 4')
    if args.beta < 0:
        raise SystemExit('--beta 不能为负')
    if not Path(args.input).exists():
        raise SystemExit(f'找不到环境数据文件：{args.input}')

    env = Environment.load(args.input, parse_sheet(args.sheet), args.skiprows)
    p = Parameters()
    print(f'环境数据 {env.source}')
    print(f'  {len(env.t)} 条记录，覆盖 0—{env.t[-1]:g} s；'
          f'网格 N={args.N}，beta={args.beta:g}，通量方案 {args.flux}')

    try:
        grid, heat, wet = solve_fields(env, p, N=args.N, beta=args.beta, rtol=args.rtol,
                                       atol_T=args.atol_T, atol_C=args.atol_C,
                                       flux=args.flux)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f'求解失败：{exc}')

    print(f'温度场 {heat.stats["steps"]} 步、右端求值 {heat.stats["nfev"]} 次；'
          f'水分场 {wet.stats["steps"]} 步、右端求值 {wet.stats["nfev"]} 次')

    template = args.template if Path(args.template).exists() else None
    if template is None:
        print(f'没找到模板 {args.template}，直接新建工作表')
    meta = export_run(args.out, env, p, grid, heat, wet, template, flux=args.flux)
    field_plots(args.out, env, p, grid, heat, wet)

    print('\n题定时刻结果（各位置的完整取值见输出目录的表1、表2）')
    for row in meta['summary']:
        depth = row['depth_5pct_mm']
        print(f'  {row["time_s"]:>4d} s  '
              f'中心 {row["T_center"]:.4f} ℃ / {row["C_center"]:.4f} kg/kg   '
              f'表面 {row["T_surface"]:.4f} ℃ / {row["C_surface"]:.4f} kg/kg   '
              f'失水深度 ' + ('未达 5% 阈值' if depth is None else f'{depth:.4f} mm'))
    print(f'\n结果已写入 {Path(args.out).resolve()}')


if __name__ == '__main__':
    main()

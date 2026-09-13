"""Q2 response vectors at 3 h. Read saved outputs only; no model solve."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle,ConnectionPatch
import fitz

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
metrics_path=ROOT/'outputs/q2/analysis/basic_metrics.csv'
boundary_path=ROOT/'outputs/q2/analysis/boundary_sensitivity.csv'
frozen_path=ROOT/'outputs/q2/validation/cases/2ef06efbd4f98cfa.npz'
with metrics_path.open(encoding='utf-8-sig',newline='') as f:
    base=list(csv.DictReader(f))[-1]
assert float(base['time_s'])==10800
keys=['moisture_center','moisture_surface','moisture_volume_average']
baseline=np.array([float(base[k]) for k in keys])
with np.load(frozen_path) as a:
    assert a['time'][-1]==10800
    frozen=np.array([a['C'][-1,0],a['C'][-1,-1],a['average_C'][-1]])
fd=100*(frozen/baseline-1)
records=[]
with boundary_path.open(encoding='utf-8-sig',newline='') as f:
    for row in csv.DictReader(f):
        factor=float(row['factor'])
        values=np.array([float(row[k]) for k in keys])
        if factor==1:
            np.testing.assert_array_equal(values,baseline)
            continue
        delta=100*(values/baseline-1)
        records.append(dict(parameter=row['parameter'],factor=factor,
            delta_center_pct=delta[0],delta_surface_pct=delta[1],delta_mean_pct=delta[2],
            center_kg_kg=values[0],surface_kg_kg=values[1],mean_kg_kg=values[2]))
allrows=[dict(parameter='frozen',factor='',delta_center_pct=fd[0],delta_surface_pct=fd[1],delta_mean_pct=fd[2],
             center_kg_kg=frozen[0],surface_kg_kg=frozen[1],mean_kg_kg=frozen[2])]+records
with (HERE/'图6_响应数据.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
plt.rcParams.update({
    'font.family':['Times New Roman','SimSun'],'mathtext.fontset':'stix','axes.unicode_minus':False,
    'font.size':9,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
    'axes.linewidth':.75,'axes.edgecolor':'#48525A','axes.labelcolor':'#1E252B',
    'xtick.color':'#29323A','ytick.color':'#29323A','xtick.major.size':2.5,'ytick.major.size':2.5,
    'xtick.major.width':.6,'ytick.major.width':.6,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'figure.facecolor':'white','savefig.facecolor':'white',
})
purple,orange,gray='#643B91','#C36928','#7C858C'
fig=plt.figure(figsize=(14.66/2.54,7.3/2.54))
left=fig.add_axes([.09,.205,.37,.675])
right=fig.add_axes([.615,.205,.36,.675])
for ax in (left,right):
    ax.axhline(0,color=gray,lw=.75,zorder=1)
    ax.axvline(0,color=gray,lw=.75,zorder=1)
    ax.set_xlabel('中心含水率相对变化 / %',labelpad=6)
    ax.set_ylabel('表面含水率相对变化 / %',labelpad=5)
    ax.scatter([0],[0],marker='+',s=32,c='#34414C',linewidths=1,zorder=6)

def vector(ax,x,y,color,filled=True,linewidth=1.2,size=25):
    ax.annotate('',xy=(x,y),xytext=(0,0),arrowprops=dict(arrowstyle='->',color=color,lw=linewidth,
                shrinkA=2,shrinkB=2,mutation_scale=8),zorder=3)
    ax.scatter([x],[y],s=size,facecolors=color if filled else 'white',edgecolors=color,linewidths=1,zorder=5)

left.set(xlim=(-3,31),ylim=(-20,7))
left.set_xticks([0,10,20,30]);left.set_yticks([-20,-15,-10,-5,0,5])
left.set_title('初态物性冻结',loc='left',fontsize=10.5,pad=9)
# Keep the dashed shaft separate from the solid head; point-based clearance
# makes the head readable without touching the endpoint marker.
left.annotate('',xy=(fd[0],fd[1]),xytext=(0,0),
    arrowprops=dict(arrowstyle='-',color='#747D84',lw=1.2,
                    linestyle=(0,(4,3)),shrinkA=4,shrinkB=12),zorder=3)
left.annotate('',xy=(fd[0],fd[1]),xytext=(0.88*fd[0],0.88*fd[1]),
    arrowprops=dict(arrowstyle='-|>',color='#747D84',lw=1.2,
                    shrinkA=0,shrinkB=5,mutation_scale=12),zorder=4)
left.scatter([fd[0]],[fd[1]],s=30,facecolors=purple,edgecolors='white',linewidths=.5,zorder=5)
left.text(1,1.1,'变物性基准',fontsize=9,color='#465059')
left.text(2,5.3,f'平均含水率 {fd[2]:+.2f}%',fontsize=9,color=purple)
left.annotate(f'初态冻结\n({fd[0]:+.2f}%, {fd[1]:+.2f}%)',xy=(fd[0],fd[1]),
              xytext=(29,-15.5),textcoords='data',ha='right',va='top',fontsize=9,
              color=purple,linespacing=1.3)

right.set(xlim=(-3.2,3.2),ylim=(-9.5,10))
right.set_xticks([-3,0,3]);right.set_yticks([-8,-4,0,4,8])
right.set_title('表面交换系数 ±10%',loc='left',fontsize=10.5,pad=9)
for row in records:
    col=purple if row['parameter']=='h' else orange
    vector(right,row['delta_center_pct'],row['delta_surface_pct'],col,row['factor']<1)
    if row['parameter']=='hm':
        lower=row['factor']<1
        right.annotate('−10%' if lower else '+10%',
            (row['delta_center_pct'],row['delta_surface_pct']),xytext=(-2,7 if lower else -8),
            textcoords='offset points',ha='right' if lower else 'left',va='bottom' if lower else 'top',
            fontsize=9,color=col)

# Zoom only the h-response; axis limits are explicit and differ from main panel.
zoom=right.inset_axes([.13,.65,.35,.27])
zoom.set_facecolor('white')
for spine in zoom.spines.values():
    spine.set_visible(True);spine.set_color('#A0A7AC');spine.set_linewidth(.55)
zoom.axhline(0,color='#A1A8AE',lw=.5);zoom.axvline(0,color='#A1A8AE',lw=.5)
zoom.set(xlim=(-.32,.34),ylim=(-.11,.11))
zoom.set_xticks([-.2,.2]);zoom.set_yticks([-.1,.1])
zoom.tick_params(labelsize=7.5,length=2,pad=1.5)
zoom.set_title(r'$h$ 局部放大',fontfamily='SimSun',fontsize=8,pad=3)
for row in records:
    if row['parameter']=='h':
        vector(zoom,row['delta_center_pct'],row['delta_surface_pct'],purple,row['factor']<1,linewidth=.85,size=16)
box=Rectangle((-.32,-.11),.66,.22,facecolor='none',edgecolor='#828B92',linewidth=.65,zorder=7)
right.add_patch(box)
connector=ConnectionPatch(xyA=(-.32,.11),coordsA=right.transData,xyB=(.98,0),coordsB=zoom.transAxes,
                          color='#A1A8AE',linewidth=.6,linestyle=(0,(2,2)))
right.add_artist(connector)

handles=[Line2D([],[],color=purple,lw=1.2,label=r'$h$'),
         Line2D([],[],color=orange,lw=1.2,label=r'$h_m$'),
         Line2D([],[],ls='',marker='o',markerfacecolor='#45515A',markeredgecolor='#45515A',markersize=4,label='−10%'),
         Line2D([],[],ls='',marker='o',markerfacecolor='white',markeredgecolor='#45515A',markersize=4,label='+10%')]
right.legend(handles=handles,loc='lower right',bbox_to_anchor=(1,.02),ncol=2,frameon=False,
             fontsize=8.5,handlelength=1.1,columnspacing=.85,handletextpad=.45,borderpad=.1,labelspacing=.6)

fig.canvas.draw()
renderer=fig.canvas.get_renderer()
for ax in (left,right,zoom):
    bb=ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    assert bb.x0>=0 and bb.x1<=1 and bb.y0>=0 and bb.y1<=1,bb
name='图6_初态物性冻结与表面交换系数的响应平面'
for ext in ('pdf','svg','png'):
    fig.savefig(HERE/(name+'.'+ext),dpi=450)
plt.close(fig)
with fitz.open(HERE/(name+'.pdf')) as doc:
    doc[0].get_pixmap(matrix=fitz.Matrix(3,3)).save(HERE/'PDF校样.png')
meta=dict(figure=6,size_cm=[14.66,7.3],time_h=3,baseline=baseline.tolist(),frozen=frozen.tolist(),
    frozen_relative_changes_pct=fd.tolist(),boundary_scenarios=records,
    axes_definition='100*(scenario-baseline)/baseline; x=center, y=surface; baseline values specific to each location',
    arrows='changes in output states relative to baseline, NOT time trajectories',
    left_frozen_scope='all thermal storage/conductivity and diffusion coefficients frozen at initial properties',
    main_panels_have_different_limits=True,zoom_units='percent on both axes, h cases only',
    interpretation_limit='No physical validation or attribution to D alone; mean change is a separate scalar, not the norm of the vector',
    input_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [metrics_path,boundary_path,frozen_path]},
    no_model_rerun=True,randomness=None)
(HERE/(name+'.figure.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
(HERE/'图注与口径说明.md').write_text(
    '# 图6：初态物性冻结与表面交换系数的响应平面\n\n'
    '图注建议：横、纵坐标分别为3 h末中心和表面含水率相对变物性基准的变化率；原点表示基准状态。'
    '左图箭头为将整套物性冻结在初态后的响应，平均含水率变化另以文字标注。'
    '右图紫、橙色分别对应换热系数h、传质系数h_m，实心、空心点分别表示参数减小、增大10%；插图放大h的响应。'
    '箭头表示同一时刻的状态差异，不表示随时间演化；两面板及插图刻度不同，不直接比较跨面板的箭头长度。\n\n'
    '左图右下方向说明冻结组中心更湿、表面更干；此比较评估忽略物性更新的综合影响，不等价于证明变物性结果符合实测，也不能将全部差异归于D。'
    '右图表明在前3 h、相同相对扰动幅度下，含水率对h_m更敏感；不得直接推广至长期末段。\n\n'
    '数据由现有CSV和冻结计算存档提取；无需重跑模型。图中相对变化均以相应位置的基准含水率为分母；截面平均值采用已有体积加权结果。'
    'PDF为14.66×7.3 cm矢量图，PNG为450 dpi预览；原始响应表与可重建代码均在本目录。\n',encoding='utf-8')
print(json.dumps(meta,ensure_ascii=False,indent=2))

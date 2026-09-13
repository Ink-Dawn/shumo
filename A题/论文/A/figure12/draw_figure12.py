"""Geometry-driving force trajectory; matched absolute-time surface histories."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
a=np.genfromtxt(HERE/'trajectory.csv',delimiter=',',names=True)
t,g,q=a['time_h'],a['G'],a['Q']
assert np.all(np.diff(t)>0) and np.isfinite(q).all()
purple='#7954A1';orange='#D68A3A';gray='#96939B'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
 'font.size':8,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'axes.unicode_minus':False,
 'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6,
 'axes.spines.top':False,'axes.spines.right':False})
product=g*q
assert np.allclose(product,a['GQ']) and np.isfinite(g).all()
cross=np.flatnonzero((product[:-1]-1)*(product[1:]-1)<0)
assert len(cross)==1
i=cross[0]
tcross=float(t[i]+(1-product[i])/(product[i+1]-product[i])*(t[i+1]-t[i]))
fig=plt.figure(figsize=(18/2.54,8.8/2.54))
left=fig.add_axes([.085,.17,.375,.72])
ax=fig.add_axes([.60,.17,.375,.72])
left.plot(t,g,color=purple,lw=1.8,label=r'$G$：几何增益')
left.plot(t,q,color=orange,lw=1.8,ls='--',label=r'$Q$：表面驱动力比')
left.plot(t,product,color='#69636F',lw=2,label=r'$GQ$：平均失水速率比')
left.axhline(1,color=gray,lw=.9,ls=(0,(4,3)),zorder=0)
left.axvline(tcross,color=gray,lw=.8,ls=':',zorder=0)
left.scatter([tcross],[1],s=28,facecolors='white',edgecolors='#69636F',zorder=5)
left.annotate(f'{tcross:.2f} h',xy=(tcross,1),xytext=(21,1.13),fontsize=8,
    arrowprops={'arrowstyle':'-','color':gray,'lw':.7})
left.set(xlim=(4,53),ylim=(0,1.85),xlabel='时间 / h',ylabel='无量纲因子')
left.set_xticks([4,12,24,36,48]); left.set_yticks([0,.5,1,1.5])
legend_left=left.legend(loc='upper right',bbox_to_anchor=(.98,.79),ncol=1,frameon=False,handlelength=1.6,fontsize=7,labelspacing=.35,borderaxespad=0)
left.set_title('(a) 几何与驱动力的时间演化',fontsize=9,pad=10)
ax.set_title('(b) 几何—驱动力作用轨迹',fontsize=9,pad=10)
xx=np.linspace(1.25,1.78,400)
for level in [.4,.7,1.3]:
    ax.plot(xx,level/xx,c='#BDB7C3',lw=.7,zorder=1)
    xpos=1.285
    ax.text(xpos,level/xpos+.018,f'$GQ={level:g}$',color='#8D8795',fontsize=7,
        bbox={'facecolor':'white','edgecolor':'none','pad':.3})
ax.plot(xx,1/xx,c='#66616D',lw=1.6,zorder=2)
ax.text(1.275,1/1.275+.033,'$GQ=1$',fontsize=8,color='#55505C',
    bbox={'facecolor':'white','edgecolor':'none','pad':1})
ax.plot(g,q,c=purple,lw=2,zorder=4)
for start,end in [(4.8,5.5),(8,9),(15,17),(30,36)]:
    ax.annotate('',xy=(np.interp(end,t,g),np.interp(end,t,q)),
        xytext=(np.interp(start,t,g),np.interp(start,t,q)),
        arrowprops={'arrowstyle':'-|>','color':purple,'lw':1.6,'mutation_scale':12},zorder=5)
for h,offset in [(6,(18,12)),(12,(17,10)),(24,(-49,-5))]:
    i=np.argmin(abs(t-h));assert abs(t[i]-h)<1e-9
    ax.scatter(g[i],q[i],s=29,c=orange,edgecolors='white',linewidth=.7,zorder=6)
    ax.annotate(f'{h} h',xy=(g[i],q[i]),xytext=offset,textcoords='offset points',
        color='#9B6328',fontsize=8,bbox={'facecolor':'white','edgecolor':'none','pad':.5},arrowprops={'arrowstyle':'-','color':'#B49A7D','lw':.65})
ax.scatter(g[0],q[0],s=24,facecolors='white',edgecolors=purple,lw=1.2,zorder=6)
ax.annotate('起点 4 h',(g[0],q[0]),xytext=(-17,15),textcoords='offset points',fontsize=7,color=purple)
ax.scatter(g[-1],q[-1],s=32,c=purple,marker='s',zorder=6)
ax.annotate(f'临界 {t[-1]:.2f} h',(g[-1],q[-1]),xytext=(1.36,.15),textcoords='data',
    fontsize=8,color=purple,arrowprops={'arrowstyle':'-','color':purple,'lw':.6})
ax.text(1.62,.98,'$GQ>1$\n失水更快',fontsize=7,color='#746C7C')
ax.text(1.31,.43,'$GQ<1$：失水更慢',fontsize=7,color='#746C7C')
ax.set(xlim=(1.25,1.78),ylim=(.1,1.13),xlabel=r'几何因子 $G=R_0/R(t)$',
    ylabel=r'表面驱动力比 $Q$')
ax.set_xticks([1.3,1.4,1.5,1.6,1.7]);ax.set_yticks([.2,.4,.6,.8,1])
handles=[Line2D([],[],color=purple,lw=2,label='收缩轨迹（→ 时间）')]
legend_right=ax.legend(handles=handles,loc='upper right',frameon=False,fontsize=7,handlelength=1.5,borderpad=.2)

gx=float(np.interp(tcross,t,g)); qx=float(np.interp(tcross,t,q))
ax.scatter([gx],[qx],s=28,facecolors='white',edgecolors='#69636F',zorder=7)
ax.annotate(f'等速转折\n{tcross:.2f} h',xy=(gx,qx),xytext=(1.66,.77),fontsize=7,
    arrowprops={'arrowstyle':'-','color':gray,'lw':.7})
fig.canvas.draw();renderer=fig.canvas.get_renderer()
for artist in [item for panel in (left,ax) for item in [panel.xaxis.label,panel.yaxis.label,panel.title,*panel.get_xticklabels(),*panel.get_yticklabels(),*panel.texts]]+list(fig.texts)+[legend_left,legend_right]:
    box=artist.get_window_extent(renderer)
    assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height
for ext in ['pdf','png','svg']:
    fig.savefig(HERE/f'figure12.{ext}',dpi=450,facecolor='white')
plt.close(fig)
cross=np.flatnonzero((a['GQ'][:-1]-1)*(a['GQ'][1:]-1)<0)
cross_times=[float(t[i]+(1-a['GQ'][i])/(a['GQ'][i+1]-a['GQ'][i])*(t[i+1]-t[i])) for i in cross]
(HERE/'图12图注.md').write_text('图12：几何变化与表面驱动力对平均含水率下降速度的共同作用。横轴 G=R0/R，纵轴 Q=(Cs,收缩−Ce)/(Cs,固定−Ce)，两组均采用附录4物性与相同环境，在相同绝对时刻比较。紫色轨迹从4 h延伸至收缩组临界时刻，箭头表示时间方向，橙色点标注6、12、24 h。灰色背景线为GQ等值线，加粗线GQ=1为两组平均含水率瞬时下降速度相同的分界；线上方收缩组更快，下方更慢。该瞬时比值不等于累计失水量之比，也不直接决定总烘干时长。\n',encoding='utf-8')
(HERE/'图12图注.md').write_text('图12：收缩过程中几何增益、表面驱动力衰减及失水速率的阶段性变化。（a）G、Q及GQ随时间的变化；（b）G–Q作用轨迹及GQ等值线，加粗线为等速线GQ=1。G=R0/R，Q=(Cs,收缩−Ce)/(Cs,固定−Ce)，两组均采用附录4物性与相同环境，在相同绝对时刻比较。时间范围为4 h至收缩组临界时刻51.0934 h。箭头表示时间方向，橙色点标注6、12、24 h，空心灰色点标注约11.16 h的等速转折（保存数据线性插值）。按正文质量守恒关系，GQ为平均含水率瞬时下降速度之比；大于1时收缩组更快，小于1时更慢。瞬时比值不等于累计失水量之比，也不直接决定总干燥时长。\n',encoding='utf-8')
meta=json.loads((HERE/'metadata.json').read_text(encoding='utf-8'))
meta.update({'panels':['G, Q, GQ versus time','G-Q trajectory'],'crossing_time_h_linear_interpolation':tcross,'figure_size_cm':[18,8.8]})
(HERE/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print({'crossing_times_h_linear_interpolation':cross_times,'start_GQ':a['GQ'][0],'end_GQ':a['GQ'][-1],'layout_check':'passed'})

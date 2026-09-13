"""Geometry-driving force trajectory; matched absolute-time surface histories."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
a=np.genfromtxt(HERE/'trajectory.csv',delimiter=',',names=True)
t,g,q=a['time_h'],a['G'],a['Q']
assert np.all(np.diff(t)>0) and np.isfinite(q).all()
purple='#7954A1';orange='#D68A3A';gray='#96939B'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
 'font.size':8,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'axes.unicode_minus':False,
 'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6,
 'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(14/2.54,9/2.54))
ax=fig.add_axes([.14,.18,.79,.74])
xx=np.linspace(1.25,1.78,400)
for level in [.4,.7,1.3]:
    ax.plot(xx,level/xx,c='#BDB7C3',lw=.7,zorder=1)
    xpos=1.285
    ax.text(xpos,level/xpos+.018,f'$GQ={level:g}$',color='#8D8795',fontsize=7,
        bbox={'facecolor':'white','edgecolor':'none','pad':.3})
ax.plot(xx,1/xx,c='#66616D',lw=1.6,zorder=2)
ax.text(1.275,1/1.275+.033,'等速线  $GQ=1$',fontsize=8,color='#55505C',
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
ax.annotate('4 h（起点）',(g[0],q[0]),xytext=(-17,15),textcoords='offset points',fontsize=8,color=purple)
ax.scatter(g[-1],q[-1],s=32,c=purple,marker='s',zorder=6)
ax.annotate(f'{t[-1]:.4f} h（临界）',(g[-1],q[-1]),xytext=(1.40,.15),textcoords='data',
    fontsize=8,color=purple,arrowprops={'arrowstyle':'-','color':purple,'lw':.6})
ax.text(1.59,1.02,'$GQ>1$：收缩组更快',fontsize=8,color='#746C7C')
ax.text(1.31,.43,'$GQ<1$：收缩组更慢',fontsize=8,color='#746C7C')
ax.set(xlim=(1.25,1.78),ylim=(.1,1.13),xlabel=r'几何因子 $G=R_0/R(t)$',
    ylabel=r'表面驱动力比 $Q$')
ax.set_xticks([1.3,1.4,1.5,1.6,1.7]);ax.set_yticks([.2,.4,.6,.8,1])
fig.text(.535,.045,'两组均在相同绝对时刻比较；箭头表示时间方向',ha='center',fontsize=7,color='#706978')
fig.canvas.draw();renderer=fig.canvas.get_renderer()
for artist in [ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels(),*ax.texts,*fig.texts]:
    box=artist.get_window_extent(renderer)
    assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height
for ext in ['pdf','png','svg']:
    fig.savefig(HERE/f'figure12.{ext}',dpi=450,facecolor='white')
plt.close(fig)
cross=np.flatnonzero((a['GQ'][:-1]-1)*(a['GQ'][1:]-1)<0)
cross_times=[float(t[i]+(1-a['GQ'][i])/(a['GQ'][i+1]-a['GQ'][i])*(t[i+1]-t[i])) for i in cross]
(HERE/'图12图注.md').write_text('图12：几何变化与表面驱动力对平均含水率下降速度的共同作用。横轴 G=R0/R，纵轴 Q=(Cs,收缩−Ce)/(Cs,固定−Ce)，两组均采用附录4物性与相同环境，在相同绝对时刻比较。紫色轨迹从4 h延伸至收缩组临界时刻，箭头表示时间方向，橙色点标注6、12、24 h。灰色背景线为GQ等值线，加粗线GQ=1为两组平均含水率瞬时下降速度相同的分界；线上方收缩组更快，下方更慢。该瞬时比值不等于累计失水量之比，也不直接决定总烘干时长。\n',encoding='utf-8')
print({'crossing_times_h_linear_interpolation':cross_times,'start_GQ':a['GQ'][0],'end_GQ':a['GQ'][-1],'layout_check':'passed'})

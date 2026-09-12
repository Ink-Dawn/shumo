from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE=Path(__file__).resolve().parent
s=json.loads((HERE/'summary.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
 'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':7,
 'axes.unicode_minus':False,'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6})
fig=plt.figure(figsize=(18/2.54,8.4/2.54))
left=fig.add_axes([.075,.20,.39,.67])
right=fig.add_axes([.57,.20,.405,.67])
blue='#7954A1';orange='#D68A3A';gray='#96939B'
for filename, eventfile, color, style, label in [
 ('control_Cmax.csv','control_event.json',orange,'--','附录4固定半径'),
 ('full_Cmax.csv','full_event.json',blue,'-','附录4收缩')]:
    a=np.genfromtxt(HERE/filename,delimiter=',',names=True)
    e=json.loads((HERE/eventfile).read_text(encoding='utf-8'))
    keep=a['time_s']<e['time_s']
    times=np.r_[a['time_s'][keep]/3600,e['time_h']]
    values=np.r_[a['Cmax'][keep],e['at_root']['Cmax']]
    assert times[-1]==e['time_h'] and np.all(np.diff(times)>0)
    left.plot(times,values,c=color,ls=style,lw=1.4,label=label)
    left.scatter([times[-1]],[values[-1]],s=20,c=color,zorder=5)
    left.annotate(f'{times[-1]:.4f} h',xy=(times[-1],values[-1]),
        xytext=(times[-1]-15,.53 if color==blue else .83),fontsize=7,color=color,
        arrowprops={'arrowstyle':'-','color':color,'lw':.65})
left.axhline(.15,color='#77737C',lw=.75,ls=(0,(3,2)))
left.text(64,.39,r'$C_{\mathrm{crit}}=0.15\ \mathrm{kg/kg}$',fontsize=7,color='#59545E')
left.set(xlim=(0,140),ylim=(0,2.7),xlabel='时间 / h',ylabel=r'$M(t)$ / (kg/kg)',title='(a) 同物性下的收缩对照')
left.set_xticks([0,30,60,90,120]);left.set_yticks([0,.5,1,1.5,2,2.5])
left.grid(axis='y',alpha=.16,lw=.5)
left.legend(frameon=False,loc='upper right',fontsize=7)

times=[s['q3_tcrit_h'],s['control_tcrit_h'],s['tcrit_h']]
width=.62
# Start total followed by two floating changes; connectors mark intermediate totals.
right.bar(0,times[0],width,color=gray)
right.bar(1,times[1]-times[0],width,bottom=times[0],color=orange)
right.bar(2,times[1]-times[2],width,bottom=times[2],color=blue)
for x,h in enumerate(times):
    right.plot([x-width/2,x+width/2],[h,h],color='#514A58',lw=1.3)
    right.text(x,h+4,f'{h:.4f}',ha='center',va='bottom',fontsize=8,fontweight='bold')
for x,h in enumerate(times[:-1]):
    right.plot([x+width/2,x+1-width/2],[h,h],color='#AAA5AD',ls='--',lw=.8)
for x, delta, desc, y in [(1,times[1]-times[0],'更换物性',(times[0]+times[1])/2),
                           (2,times[2]-times[1],'引入收缩',(times[2]+times[1])/2)]:
    right.text(x,y,f'{delta:+.4f}\nh',ha='center',va='center',fontsize=7.5,color='white',fontweight='bold')
    right.text(x,12,desc,ha='center',fontsize=7,color='#625B69')
right.set_xticks([0,1,2],['附录3固定','附录4固定','附录4收缩'])
right.set(xlim=(-.55,2.55),ylim=(0,151),ylabel='临界时间 / h',title='(b) 三种模型的临界时间变化')
right.set_yticks([0,30,60,90,120,150]);right.grid(axis='y',alpha=.16,lw=.5)
right.set_axisbelow(True)
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
for ax in [left,right]:
    for a in [ax.xaxis.label,ax.yaxis.label,ax.title,*ax.get_xticklabels(),*ax.get_yticklabels(),*ax.texts]:
        b=a.get_window_extent(renderer)
        assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,a.get_text()
for ext in ['pdf','png','svg']:
    fig.savefig(HERE/f'figure11.{ext}',dpi=450,facecolor='white')
plt.close(fig)
(HERE/'图11图注.md').write_text('图11：同物性下的收缩对照及三种模型的临界时间比较。左：附录4固定半径组与收缩组的全域最大含水率 M(t)，分别绘制至各自临界时刻，水平虚线为达标阈值。右：起始柱表示附录3固定半径的临界时间，两根浮动柱分别表示更换为附录4物性和引入收缩后的时间变化；各阶段临界时间标在对应水平端点。两段差值是依次改变模型设置的受控对比，不表示无交互作用的普适分解。数值来自原始未舍入结果，显示四位小数。\n',encoding='utf-8')
print({'times_h':times,'changes_h':list(np.diff(times)),'bounds_check':'passed'})

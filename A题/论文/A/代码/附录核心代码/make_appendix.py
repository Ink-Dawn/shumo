"""由真实核心文件生成LaTeX列表，行号来自构建清单，不手抄代码。"""
from pathlib import Path
import json, re, ast
HERE=Path(__file__).resolve().parent
m=json.loads((HERE/'code_manifest.json').read_text(encoding='utf-8'))
header=r'''% 在 main.tex 原空代码附录位置使用 \input{代码/附录核心代码/附录核心代码.tex}
% 主文件需加载 listings、xcolor，并使用 XeLaTeX。
\providecommand{\AppendixCodeDir}{代码/附录核心代码}
\begingroup
\normalsize
\lstdefinestyle{qcore}{language=Python,
  basicstyle=\linespread{1}\fontsize{8.5}{10.5}\selectfont\ttfamily,
  keywordstyle=\bfseries,commentstyle=\color{black!60},
  stringstyle=\color{black},numbers=left,
  numberstyle=\fontsize{7}{8}\selectfont\color{black!50},
  numbersep=6pt,frame=single,framerule=0.3pt,
  rulecolor=\color{black!30},xleftmargin=1.8em,xrightmargin=0.2em,
  breaklines=true,breakatwhitespace=false,columns=fullflexible,
  keepspaces=true,showstringspaces=false,tabsize=4,
  aboveskip=6pt,belowskip=8pt}
\section*{附录1：四问核心求解代码}
以下摘录四问的主要计算代码，完整程序见支撑材料。
时间用秒、长度用米，环境温度用摄氏度，含水率为干基值。
\(N\) 为径向区间数，四问分别取 \(2560,2560,640,5120\)。

\subsection*{问题一：非线性扩散与分段隐式积分}
\texttt{env} 依次存放时间、环境温度和环境含水率。
温度与水分分开计算；第一问内部温度用 K，输出时换回摄氏度。
\lstinputlisting[style=qcore,firstline=4,lastline=14,firstnumber=4]{\AppendixCodeDir/q1_core.py}
\lstinputlisting[style=qcore,firstline=30,lastline=101,firstnumber=30]{\AppendixCodeDir/q1_core.py}

\subsection*{问题二：状态依赖物性下的热湿耦合}
网格沿用问题一，每次计算通量时更新物性，并联立推进温度和含水率。
'''
def listing(file,key):
    a,b=m[file]['ranges'][key]
    return ('\\lstinputlisting[style=qcore,firstline='+str(a)+',lastline='+str(b)
            +',firstnumber='+str(a)+']{\\AppendixCodeDir/'+file+'}\n')
s=header+listing('q2_core.py','Q2_KERNEL')
s+=r'''
\subsection*{问题三：长期环境延拓与首次达标事件}
物性与空间离散沿用问题二。4 h 后使用末一小时的平均环境，每6 h 分段。
检查全部节点的含水率，以最大值首次降到 \(0.15\) 的时刻为临界时间。
\texttt{cpart} 表示含水率的位置；\texttt{factory(a)} 设置本段环境。
'''+listing('q3_core.py','EVENT_KERNEL')+'\\newpage\n'+listing('q3_core.py','Q3_DRIVER')
s+=r'''
\subsection*{问题四：材料坐标下的收缩模型}
\texttt{radius\_data} 的两列为时间与半径，半径需由附件2的厘米换算为米。
采用附录4物性，按 \(r=R(t)\xi\) 更新面积和体积。
事件判断沿用问题三；超出当前半径的位置记为 \texttt{NaN}。
'''+listing('q4_core.py','Q4_KERNEL')
s+='\n'+r'\endgroup'+'\n'
# 第一问的导入和网格按函数位置选取，核心部分按标记选取。
source=(HERE/'q1_core.py').read_text(encoding='utf-8')
nodes=ast.parse(source).body
start=next(n.lineno for n in nodes if isinstance(n,ast.Import))
end=next(n.end_lineno for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='mesh')
first='\\lstinputlisting[style=qcore,firstline='+str(start)+',lastline='+str(end)+',firstnumber='+str(start)+']{\\AppendixCodeDir/q1_core.py}'
s=re.sub(r'\\lstinputlisting\[[^\n]+\]\{\\AppendixCodeDir/q1_core.py\}',lambda match: first if 'firstline=4,' in match.group() else listing('q1_core.py','Q1_KERNEL').rstrip(),s)
# 附录只保留标题和代码，省去代码外的说明段落
s=re.sub(r'(\\(?:sub)?section\*\{[^\n]*\}\n)[\s\S]*?(?=\\(?:subsection\*|lstinputlisting))',r'\1',s)
(HERE/'附录核心代码.tex').write_text(s,encoding='utf-8')
wrapper=r'''\documentclass[UTF8,a4paper]{ctexart}
\usepackage[top=2.54cm,bottom=2.54cm,left=3.17cm,right=3.17cm]{geometry}
\usepackage{amsmath,amssymb,listings,xcolor}
\renewcommand{\baselinestretch}{1.3}
\setlength{\parindent}{2em}
\newcommand{\AppendixCodeDir}{.}
\begin{document}
\input{附录核心代码.tex}
\end{document}
'''
(HERE/'附录核心代码_预览.tex').write_text(wrapper,encoding='utf-8')

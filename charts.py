"""Figures from saved simulations. The first path is selected before simulation."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np


def rows(file):
    with file.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def make_figures(directory):
    directory = Path(directory)
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    cjk = next((f for f in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "PingFang SC")
                if f in fonts), None)
    def label(chinese, english):
        return chinese if cjk else english
    plt.rcParams.update({"font.family":cjk or "DejaVu Sans", "axes.unicode_minus":False,
                         "axes.spines.top":False, "axes.spines.right":False,
                         "axes.labelcolor":"#314451","xtick.color":"#41505a","ytick.color":"#41505a",
                         "axes.edgecolor":"#c2ccd1","font.size":10,"figure.facecolor":"white"})
    pathfile = directory/"BBM_A_path.csv"
    if pathfile.exists():
        path = rows(pathfile)
        get = lambda k: np.array([float(r[k]) for r in path])
        t = get("day")/252
        fig, axes = plt.subplots(3, 1, figsize=(10,7.0), sharex=True,
                                 gridspec_kw={"height_ratios":[1.4,0.9,0.9]})
        fig.subplots_adjust(left=.12, right=.97, bottom=.10, top=.87, hspace=.24)
        fig.text(.12,.944,label("一次模拟中，繁荣与下跌如何交替","Booms and drawdowns in one simulated market"),
                 fontsize=19,color="#162f3a",weight="bold")
        fig.text(.12,.903,label("BBM · 1996–2010 窗口输入 · 第一个预设种子 · 横轴为模拟时间",
                                "BBM · 1996–2010 inputs · first preset seed · simulated time"),
                 color="#52636d")
        price = get("index_base_100")
        axes[0].plot(t,price,color="#12605e",lw=1.15)
        axes[0].set_ylabel(label("指数（期初 100）","Index (start = 100)"))
        epsfile = directory/"BBM_A_path_episodes.csv"
        if epsfile.exists():
            for e in rows(epsfile):
                peak,trough = int(e["peak"]),int(e["trough"])
                axes[0].axvspan(t[peak],t[trough],color="#d99a7c",alpha=.22,lw=0)
        dd = get("drawdown")
        axes[1].fill_between(t,dd,0,color="#ab5b45",alpha=.70,lw=0)
        axes[1].axhline(-.2,ls="--",lw=1,color="#834932",alpha=.9)
        axes[1].set_ylabel(label("距历史高点回撤","Drawdown from high"))
        axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"{v:.0%}"))
        axes[2].plot(t,get("x"),color="#35689a",lw=1)
        axes[2].axhline(0,color="#536775",lw=.8,ls="--")
        axes[2].set_ylabel(label("市值加权错价 x","Cap-weighted x"))
        axes[2].set_xlabel(label("模拟年份","Simulated years"))
        for ax in axes:
            ax.grid(axis="y",color="#dce4e6",alpha=.8,ls=":",lw=.8)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.margins(x=0)
        fig.savefig(directory/"bbm_path.png",dpi=170)
        plt.close(fig)

    compfile = directory/"comparison.csv"
    if not compfile.exists():
        return
    comp = rows(compfile)
    metrics = ("bears_20_per_century","ou_entropy")
    picked = {(r["metric"],r["scenario"]):r for r in comp if r["table"] in ("33","34")
              and r["metric"] in metrics}
    if len(picked) != 4:
        return
    fig, axes = plt.subplots(1,2,figsize=(10,4.7))
    fig.subplots_adjust(left=.085,right=.97,bottom=.17,top=.77,wspace=.30)
    fig.text(.085,.927,label("关闭价格反馈后，熊市频率与熵产生率如何变化",
                             "Bear frequency and entropy production after switching off impact"),
             fontsize=17,weight="bold",color="#162f3a")
    fig.text(.085,.856,label("窗口 A · 点为跨路径均值，误差线为一个标准误",
                             "Window A · across-path mean and one standard error"),color="#52636d")
    for ax, metric, title in zip(axes,metrics,(label("20% 下跌次数 / 百年","20% falls per century"),
                                             label("OU 熵产生率 / 纳特每年","OU entropy / nats per year"))):
        for offset, key, legend, col in ((-.10,"center",label("论文","Paper"),"#788795"),
                                         (.10,"reconstructed",label("本次复现","Reconstruction"),"#007e73")):
            vals,errs = [],[]
            for scen in ("baseline","no_feedback"):
                r = picked[(metric,scen)]
                v=float(r[key])
                upper=float(r["high" if key=="center" else "our_high"])
                vals.append(v)
                errs.append(max(upper-v,0))
            ax.errorbar(np.array([0.,1.])+offset,vals,yerr=errs,fmt="o",ms=7,capsize=4,
                        elinewidth=1.6,color=col,label=legend)
        ax.set_xlim(-.4,1.4)
        ax.set_ylim(bottom=0)
        ax.set_xticks([0,1],[label("完整模型","Baseline"),label("关闭反馈 c = 0","No impact c = 0")])
        ax.set_title(title,loc="left",fontsize=12,color="#293e49",pad=12)
        ax.grid(axis="y",color="#dce4e6",ls=":",lw=.8)
    axes[0].legend(loc="lower left",frameon=False,fontsize=9)
    fig.savefig(directory/"feedback.png",dpi=170)
    plt.close(fig)

# -*- coding: utf-8 -*-

import numpy as np
import logging, sys, operator

from matplotlib.colors import Normalize
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from .parser import unique


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def zscore(data2d, axis=0):
    """Standardize the mean and variance of the data axis Parameters.

    :param data2d: DataFrame to normalize.
    :param axis: int, Which axis to normalize across. If 0, normalize across rows,
                  if 1, normalize across columns. If None, don't change data
                  
    :Returns: Normalized DataFrame. Normalized data with a mean of 0 and variance of 1
              across the specified axis.

    """
    if axis is None:
        # normalized to mean and std using entire matrix
        # z_scored = (data2d - data2d.values.mean()) / data2d.values.std(ddof=1)
        return data2d
    assert axis in [0,1]
    z_scored = data2d.apply(lambda x: (x-x.mean())/x.std(ddof=1), 
                            axis=operator.xor(1, axis))
    return z_scored

def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

def _skip_ticks(labels, tickevery):
    """Return ticks and labels at evenly spaced intervals."""
    n = len(labels)
    if tickevery == 0:
        ticks, labels = [], []
    elif tickevery == 1:
        ticks, labels = np.arange(n) + .5, labels
    else:
        start, end, step = 0, n, tickevery
        ticks = np.arange(start, end, step) + .5
        labels = labels[start:end:step]
    return ticks, labels

def _auto_ticks(ax, labels, axis):
    transform = ax.figure.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(transform)
    size = [bbox.width, bbox.height][axis]
    axis = [ax.xaxis, ax.yaxis][axis]
    tick, = ax.xaxis.set_ticks([0])
    fontsize = tick.label1.get_size()
    max_ticks = int(size // (fontsize / 72))
    if max_ticks < 1: 
        tickevery = 1
    else:
        tickevery = len(labels) // max_ticks + 1
    return tickevery


def heatmap(df, z_score=None, title='', figsize=(5,5), cmap='RdBu_r', 
            xticklabels=True, yticklabels=True, ofname=None, **kwargs):
    """Visualize the dataframe.

    :param df: DataFrame from expression table.
    :param z_score: z_score axis{0, 1}. If None, don't normalize data.
    :param title: gene set name.
    :param outdir: path to save heatmap.
    :param figsize: heatmap figsize.
    :param cmap: matplotlib colormap.
    :param ofname: output file name. If None, don't save figure 

    """
    df = zscore(df, axis=z_score)
    df = df.iloc[::-1]
    # If working on commandline, don't show figure
    if hasattr(sys, 'ps1') and (ofname is None): 
        fig = plt.figure(figsize=figsize)
    else:
        fig = Figure(figsize=figsize)
        canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    vmin = np.percentile(df.min(), 2)
    vmax =  np.percentile(df.max(), 98)
    matrix = ax.pcolormesh(df.values, cmap=cmap, vmin=vmin, vmax=vmax)
    xstep = _auto_ticks(ax, df.columns.values, 0)
    ystep = _auto_ticks(ax, df.index.values, 1)
    xticks, xlabels = _skip_ticks(df.columns.values, tickevery=xstep)
    yticks, ylabels = _skip_ticks(df.index.values, tickevery=ystep)
    ax.set_ylim([0,len(df)])
    ax.set(xticks=xticks, yticks=yticks)
    ax.set_xticklabels(xlabels if xticklabels else '', fontsize=14, rotation=90)
    ax.set_yticklabels(ylabels if yticklabels else '',  fontsize=14)
    ax.set_title("%s\nHeatmap of the Analyzed Geneset"%title, fontsize=20)
    ax.tick_params(axis='both', which='both', bottom=False, top=False,
                   right=False, left=False)
    # cax=fig.add_axes([0.93,0.25,0.05,0.20])
    # cbar = fig.colorbar(matrix, cax=cax)
    cbar = colorbar(matrix)
    cbar.ax.tick_params(axis='both', which='both', bottom=False, top=False,
                        right=False, left=False)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
        cbar.ax.spines[side].set_visible(False)
    # cbar.ax.set_title('',loc='left')

    if ofname is not None: 
        # canvas.print_figure(ofname, bbox_inches='tight', dpi=300)
        fig.savefig(ofname, bbox_inches='tight', dpi=300)
    return

class GSEAPlot(object):
    def __init__(self, rank_metric, term, hit_indices, nes, pval, fdr, RES,
                 pheno_pos='', pheno_neg='', figsize=(6, 5.5), 
                 cmap='seismic', ofname=None, **kwargs):
        # center color map at midpoint = 0
        self._norm = MidpointNormalize(midpoint=0)
        #dataFrame of ranked matrix scores
        self._x = np.arange(len(rank_metric))
        self.rankings = rank_metric.values
        self.RES = RES
        self._im_matrix = np.tile(self.rankings, (2,1))

        self.figsize = figsize
        self.term = term
        self.cmap=cmap
        self.ofname=ofname
        
        self._pos_label = pheno_pos 
        self._neg_label = pheno_neg
        self._zero_score_ind = np.abs(self.rankings).argmin()
        self._z_score_label = 'Zero score at ' + str(self._zero_score_ind)
        self._hit_indices = hit_indices
        self.module = 'tmp' if ofname is  None else ofname.split(".")[-2]
        if self.module == 'ssgsea':
            self._nes_label = 'ES: '+ "{:.3f}".format(float(nes))
            self._pval_label='Pval: invliad for ssgsea'
            self._fdr_label='FDR: invalid for ssgsea'
        else:
            self._nes_label = 'NES: '+ "{:.3f}".format(float(nes))
            self._pval_label = 'Pval: '+ "{:.3f}".format(float(pval))
            self._fdr_label = 'FDR: '+ "{:.3f}".format(float(fdr))

        # output truetype
        plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
        # in most case, we will have many plots, so do not display plots
        # It's also usefull to run this script on command line.

        # GSEA Plots
        if hasattr(sys, 'ps1') and (self.ofname is None):
            # working inside python console, show figure
            self.fig = plt.figure(figsize=self.figsize)
        else:
            # If working on command line, don't show figure
            self.fig = Figure(figsize=self.figsize)
            self._canvas = FigureCanvas(self.fig) 
        
        self.fig.suptitle(self.term, fontsize=16, fontweight='bold')

    def axes_rank(self, rect):
        """
        rect : sequence of float
               The dimensions [left, bottom, width, height] of the new axes. All
               quantities are in fractions of figure width and height.
        """
        # Ranked Metric Scores Plot
        ax1 = self.fig.add_axes(rect, sharex=self.ax)
        if self.module == 'ssgsea':
            ax1.fill_between(self._x, y1=np.log(self.rankings), y2=0, color='#C9D3DB')
            ax1.set_ylabel("log ranked metric", fontsize=14)
        else:
            ax1.fill_between(self._x, y1=self.rankings, y2=0, color='#C9D3DB')
            ax1.set_ylabel("Ranked list metric", fontsize=14)

        ax1.text(.05, .9, self._pos_label, color='red',
                horizontalalignment='left', verticalalignment='top',
                transform=ax1.transAxes)
        ax1.text(.95, .05, self._neg_label, color='Blue',
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax1.transAxes)
        # the x coords of this transformation are data, and the y coord are axes
        trans1 = transforms.blended_transform_factory(ax1.transData, ax1.transAxes)
        ax1.vlines(self._zero_score_ind, 0, 1, linewidth=.5, 
                    transform=trans1, linestyles='--', color='grey')

        hap = self._zero_score_ind / max(self._x) 
        if hap < 0.25:
            ha = 'left'
        elif hap > 0.75:
            ha = 'right'
        else:
            ha = 'center'  
        ax1.text(hap, 0.5, self._z_score_label,
                    horizontalalignment=ha,
                    verticalalignment='center',
                    transform=ax1.transAxes)
        ax1.set_xlabel("Rank in Ordered Dataset", fontsize=14)
        ax1.spines['top'].set_visible(False)
        ax1.tick_params(axis='both', which='both', top=False, right=False, left=False)
        ax1.locator_params(axis='y', nbins=5)
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda tick_loc,tick_num :  '{:.1f}'.format(tick_loc) ))


    def axes_hits(self, rect):
        """
        rect : sequence of float
               The dimensions [left, bottom, width, height] of the new axes. All
               quantities are in fractions of figure width and height.
        """
        # gene hits
        ax2 = self.fig.add_axes(rect, sharex=self.ax)
        # the x coords of this transformation are data, and the y coord are axes
        trans2 = transforms.blended_transform_factory(ax2.transData, ax2.transAxes)
        ax2.vlines(self._hit_indices, 0, 1, linewidth=.5, transform=trans2)
        ax2.spines['bottom'].set_visible(False)
        ax2.tick_params(axis='both', which='both', 
                        bottom=False, top=False,
                        right=False, left=False, 
                        labelbottom=False, labelleft=False)

    def axes_cmap(self, rect):
        """
        rect : sequence of float
               The dimensions [left, bottom, width, height] of the new axes. All
               quantities are in fractions of figure width and height.
        """
        # colormap
        ax3 =  self.fig.add_axes(rect, sharex=self.ax)
        ax3.imshow(self._im_matrix, aspect='auto', norm=self._norm, 
                   cmap=self.cmap, interpolation='none') # cm.coolwarm
        ax3.spines['bottom'].set_visible(False)
        ax3.tick_params(axis='both', which='both', 
                        bottom=False, top=False,
                        right=False, left=False, 
                        labelbottom=False, labelleft=False)

    def axes_stat(self, rect):
        """
        rect : sequence of float
               The dimensions [left, bottom, width, height] of the new axes. All
               quantities are in fractions of figure width and height.
        """
        # Enrichment score plot
        
        ax4 = self.fig.add_axes(rect)
        ax4.plot(self._x, self.RES, linewidth=4, color ='#88C544')
        ax4.text(.1, .1, self._fdr_label, transform=ax4.transAxes)
        ax4.text(.1, .2, self._pval_label, transform=ax4.transAxes)
        ax4.text(.1, .3, self._nes_label, transform=ax4.transAxes)

        # the y coords of this transformation are data, and the x coord are axes
        trans4 = transforms.blended_transform_factory(ax4.transAxes, ax4.transData)
        ax4.hlines(0, 0, 1, linewidth=.5, transform=trans4, color='grey')
        ax4.set_ylabel("Enrichment Score", fontsize=14)
        #ax4.set_xlim(min(self._x), max(self._x))
        ax4.tick_params(axis='both', which='both', 
                        bottom=False, top=False, right=False,
                        labelbottom=False)
        ax4.locator_params(axis='y', nbins=5)
        # FuncFormatter need two argument, I don't know why. this lambda function used to format yaxis tick labels.
        ax4.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda tick_loc,tick_num :  '{:.1f}'.format(tick_loc)) )

        self.ax = ax4

    def add_axes(self):
        """
        Please check matplotlib docs about how to `add_axes` to figure.

        Here is a more flexible way to create a new gseaplot.
        For example, don't show ranking and merge hits and colormap together 
        just used:
        
            self.axes_stat([0.1,0.2,0.8,0.8]) # axes_stat should be called first
            self.axes_cmap([0.1,0.1,0.8,0.1])
            self.axes_hits([0.1,0.1,0.8,0.1])

        """
        self.axes_stat([0.1,0.5,0.8,0.4])
        self.axes_hits([0.1,0.45,0.8,0.05])
        self.axes_cmap([0.1,0.40,0.8,0.05])
        self.axes_rank([0.1,0.1,0.8,0.3])
        # self.fig.subplots_adjust(hspace=0)
        # self.fig.tight_layout()

    def savefig(self, bbox_inches='tight', dpi=300):   
        
        #if self.ofname is not None: 
        if hasattr(sys, 'ps1') and (self.ofname is not None):
            self.fig.savefig(self.ofname, bbox_inches=bbox_inches, dpi=dpi)
        elif self.ofname is None:
            return
        else:
            self._canvas.print_figure(self.ofname, bbox_inches=bbox_inches, dpi=300)
        return


def gseaplot(rank_metric, term, hit_indices, nes, pval, fdr, RES,
              pheno_pos='', pheno_neg='', figsize=(6,5.5), 
              cmap='seismic', ofname=None, **kwargs):
    """This is the main function for reproducing the gsea plot.

    :param rank_metric: pd.Series for rankings, rank_metric.values.
    :param term: gene_set name
    :param hit_indices: hits indices of rank_metric.index presented in gene set S.
    :param nes: Normalized enrichment scores.
    :param pval: nominal p-value.
    :param fdr: false discovery rate.
    :param RES: running enrichment scores.
    :param pheno_pos: phenotype label, positive correlated.
    :param pheno_neg: phenotype label, negative correlated.
    :param figsize: matplotlib figsize.
    :param ofname: output file name. If None, don't save figure 

    """
    g = GSEAPlot(rank_metric, term, hit_indices, nes, pval, fdr, RES,
                 pheno_pos, pheno_neg, figsize, cmap, ofname)
    g.add_axes()
    g.savefig()


def isfloat(x):
        try:
            float(x)
        except:
            return False
        else:
            return True

def dotplot(df, column='Adjusted P-value', title='', cutoff=0.05, top_term=10, 
            sizes=None, norm=None, legend=True, figsize=(6, 5.5), 
            cmap='RdBu_r', ofname=None, **kwargs):
    """Visualize enrichr results.

    :param df: GSEApy DataFrame results.
    :param column: which column of DataFrame to show. Default: Adjusted P-value
    :param title: figure title
    :param cutoff: terms with 'column' value < cut-off are shown.
    :param top_term: number of enriched terms to show.
    :param ascending: bool, the order of y axis.
    :param sizes: tuple, (min, max) scatter size. Not functional for now
    :param norm: maplotlib.colors.Normalize object.
    :param legend: bool, whether to show legend.
    :param figsize: tuple, figure size. 
    :param cmap: matplotlib colormap
    :param ofname: output file name. If None, don't save figure 

    """


    colname = column    
    # sorting the dataframe for better visualization
    if colname in ['Adjusted P-value', 'P-value']:
        # check if any values in `df[colname]` can't be coerced to floats
        can_be_coerced = df[colname].map(isfloat)
        if np.sum(~can_be_coerced) > 0:
            raise ValueError('some value in %s could not be typecast to `float`'%colname)
        else:
            df.loc[:, colname] = df[colname].map(float)
        df = df[df[colname] <= cutoff]
        if len(df) < 1: 
            msg = "Warning: No enrich terms when cutoff = %s"%cutoff
            return msg
        df = df.assign(logAP=lambda x: - x[colname].apply(np.log10))
        colname='logAP'
    df = df.sort_values(by=colname).iloc[-top_term:,:]
    # 
    temp = df['Overlap'].str.split("/", expand=True).astype(int)
    df = df.assign(Hits=temp.iloc[:,0], Background=temp.iloc[:,1])
    df = df.assign(Hits_ratio=lambda x:x.Hits / x.Background)
    # x axis values
    x = df.loc[:, colname].values
    combined_score = df['Combined Score'].round().astype('int')
    # y axis index and values
    y = [i for i in range(0,len(df))]
    ylabels = df['Term'].values
    # Normalise to [0,1]
    # b = (df['Count']  - df['Count'].min())/ np.ptp(df['Count'])
    # area = 100 * b
    
    # control the size of scatter and legend marker
    levels = numbers = np.sort(df.Hits.unique())
    if norm is None:
        norm = Normalize()
    elif isinstance(norm, tuple):
        norm = Normalize(*norm)
    elif not isinstance(norm, Normalize):
        err = ("``size_norm`` must be None, tuple, "
                "or Normalize object.")
        raise ValueError(err)
    min_width, max_width = np.r_[20, 100] * plt.rcParams["lines.linewidth"]
    norm.clip = True
    if not norm.scaled():
        norm(np.asarray(numbers))
    size_limits = norm.vmin, norm.vmax
    scl = norm(numbers)
    widths = np.asarray(min_width + scl * (max_width - min_width))
    if scl.mask.any():
        widths[scl.mask] = 0
    sizes = dict(zip(levels, widths))
    df['sizes'] = df.Hits.map(sizes)
    area = df['sizes'].values

    # create scatter plot
    if hasattr(sys, 'ps1') and (ofname is None):
        # working inside python console, show figure
        fig, ax = plt.subplots(figsize=figsize)
    else:
        # If working on commandline, don't show figure
        fig = Figure(figsize=figsize)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
    vmin = np.percentile(combined_score.min(), 2)
    vmax =  np.percentile(combined_score.max(), 98)
    sc = ax.scatter(x=x, y=y, s=area, edgecolors='face', c=combined_score,
                    cmap=cmap, vmin=vmin, vmax=vmax)

    if column in ['Adjusted P-value', 'P-value']:
        xlabel = "-log$_{10}$(%s)"%column
    else:
        xlabel = column 
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.yaxis.set_major_locator(plt.FixedLocator(y))
    ax.yaxis.set_major_formatter(plt.FixedFormatter(ylabels))
    ax.set_yticklabels(ylabels, fontsize=16)
    
    # ax.set_ylim([-1, len(df)])
    ax.grid()
    # colorbar
    cax=fig.add_axes([0.95,0.20,0.03,0.22])
    cbar = fig.colorbar(sc, cax=cax,)
    cbar.ax.tick_params(right=True)
    cbar.ax.set_title('Combined\nScore',loc='left', fontsize=12)

    # for terms less than 3
    if len(df) >= 3:
        # find the index of the closest value to the median
        idx = [area.argmax(), np.abs(area - area.mean()).argmin(), area.argmin()]
        idx = unique(idx)
    else:
        idx = range(len(df))
    label = df.iloc[idx, df.columns.get_loc('Hits')]
    
    if legend:
        handles, _ = ax.get_legend_handles_labels()
        legend_markers = []
        for ix in idx: 
            legend_markers.append(ax.scatter([],[], s=area[ix], c='b'))
        # artist = ax.scatter([], [], s=size_levels,) 
        ax.legend(legend_markers, label, title='Hits')
    ax.set_title(title, fontsize=20, fontweight='bold')
    
    if ofname is not None: 
        # canvas.print_figure(ofname, bbox_inches='tight', dpi=300)
        fig.savefig(ofname, bbox_inches='tight', dpi=300)
        return
    return ax

def barplot(df, column='Adjusted P-value', title="", cutoff=0.05, top_term=10,
            figsize=(6.5,6), color='salmon', ofname=None, **kwargs):
    """Visualize enrichr results.

    :param df: GSEApy DataFrame results.
    :param column: which column of DataFrame to show. Default: Adjusted P-value
    :param title: figure title.
    :param cutoff: terms with 'column' value < cut-off are shown.
    :param top_term: number of top enriched terms to show.
    :param figsize: tuple, matplotlib figsize.
    :param color: color for bars.
    :param ofname: output file name. If None, don't save figure    
    
    """

    colname = column   
    if colname in ['Adjusted P-value', 'P-value']: 
        # check if any values in `df[colname]` can't be coerced to floats
        can_be_coerced = df[colname].map(isfloat)
        if np.sum(~can_be_coerced) > 0:
            raise ValueError('some value in %s could not be typecast to `float`'%colname)
        else:
            df.loc[:, colname] = df[colname].map(float)
        df = df[df[colname] <= cutoff]
        if len(df) < 1: 
            msg = "Warning: No enrich terms using library %s when cutoff = %s"%(title, cutoff)
            return msg
        df = df.assign(logAP = lambda x: - x[colname].apply(np.log10))
        colname = 'logAP' 
        
    dd = df.sort_values(by=colname).iloc[-top_term:,:]
    # create bar plot
    if hasattr(sys, 'ps1') and (ofname is None):
        # working inside python console, show (True) figure
        fig = plt.figure(figsize=figsize)
    else:
        # If working on commandline, don't show figure
        fig = Figure(figsize=figsize)
        canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    bar = dd.plot.barh(x='Term', y=colname, color=color, 
                       alpha=0.75, fontsize=16, ax=ax)
    
    if column in ['Adjusted P-value', 'P-value']:
        xlabel = "-log$_{10}$(%s)"%column
    else:
        xlabel = column 
    bar.set_xlabel(xlabel, fontsize=16, fontweight='bold')
    bar.set_ylabel("")
    bar.set_title(title, fontsize=24, fontweight='bold')
    bar.xaxis.set_major_locator(MaxNLocator(integer=True))
    bar.legend_.remove()
    adjust_spines(ax, spines=['left','bottom'])

    if hasattr(sys, 'ps1') and (ofname is not None): 
        # canvas.print_figure(ofname, bbox_inches='tight', dpi=300)
        fig.savefig(ofname, bbox_inches='tight', dpi=300)
        return
    elif ofname is None:
        return
    elif not hasattr(sys, 'ps1') and (ofname is not None):
        canvas.print_figure(ofname, bbox_inches='tight', dpi=300)
        return
    return ax

def adjust_spines(ax, spines):
    """function for removing spines and ticks.

    :param ax: axes object
    :param spines: a list of spines names to keep. e.g [left, right, top, bottom]
                    if spines = []. remove all spines and ticks.

    """
    for loc, spine in ax.spines.items():
        if loc in spines:
            # spine.set_position(('outward', 10))  # outward by 10 points
            # spine.set_smart_bounds(True)
            continue
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

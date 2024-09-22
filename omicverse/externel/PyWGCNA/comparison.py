import math
import sys

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import networkx as nx
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

# bcolors

HEADER = "\033[95m"
OKBLUE = "\033[94m"
OKCYAN = "\033[96m"
OKGREEN = "\033[92m"
WARNING = "\033[93m"
FAIL = "\033[91m"
ENDC = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"


class Comparison:
    """
    A class used to compare PyWGCNA to another PyWGCNA or any gene marker table

    :param geneModules: gene modules of networks
    :type geneModules: dict
    :param jaccard_similarity: jaccard similarity of common genes between each modules
    :type jaccard_similarity: pandas dataframe
    :param P_value: P value of common genes between each modules
    :type P_value: pandas dataframe
    :param fraction: fraction of common genes between each modules
    :type fraction: pandas dataframe

    """

    def __init__(self, geneModules=None):
        self.geneModules = geneModules

        self.jaccard_similarity = None
        self.P_value = None
        self.fraction = None

    def calculateJaccardSimilarity(self):
        """
        Calculate jaccard similarity matrix along multiple networks

        :return: dataframe containing jaccard similarity between all modules in all PyWGCNA objects
        :rtype: pandas dataframe
        """
        num = 0
        names = []
        for network in self.geneModules.keys():
            num = num + len(self.geneModules[network].moduleColors.unique())
            tmp = [f"{network}:" + s for s in self.geneModules[network].moduleColors.unique().tolist()]
            names = names + tmp
        jaccard_similarity = pd.DataFrame(0.0, columns=names, index=names)

        for network1 in self.geneModules.keys():
            for network2 in self.geneModules.keys():
                if network1 != network2:
                    modules1 = self.geneModules[network1].moduleColors.unique().tolist()
                    modules2 = self.geneModules[network2].moduleColors.unique().tolist()
                    for module1 in modules1:
                        for module2 in modules2:
                            list1 = self.geneModules[network1].index[
                                self.geneModules[network1].moduleColors == module1].tolist()
                            list2 = self.geneModules[network2].index[
                                self.geneModules[network2].moduleColors == module2].tolist()
                            jaccard_similarity.loc[
                                f"{network1}:{module1}", f"{network2}:{module2}"] = self.jaccard(list1,
                                                                                                               list2)
                else:
                    modules = self.geneModules[network1].moduleColors.unique().tolist()
                    for module in modules:
                        jaccard_similarity.loc[f"{network1}:{module}", f"{network1}:{module}"] = 1.0

        self.jaccard_similarity = jaccard_similarity

        return jaccard_similarity

    def calculateFraction(self):
        """
        Calculate common fraction along multiple networks

        :return: dataframe containing fraction between all modules in all netwroks
        :rtype: pandas dataframe
        """

        num = 0
        names = []
        for network in self.geneModules.keys():
            num = num + len(self.geneModules[network].moduleColors.unique())
            tmp = [f"{network}:" + s for s in self.geneModules[network].moduleColors.unique().tolist()]
            names = names + tmp
        fraction = pd.DataFrame(0, columns=names, index=names)

        for network1 in self.geneModules.keys():
            for network2 in self.geneModules.keys():
                if network1 != network2:
                    modules1 = self.geneModules[network1].moduleColors.unique().tolist()
                    modules2 = self.geneModules[network2].moduleColors.unique().tolist()
                    for module1 in modules1:
                        for module2 in modules2:
                            list1 = self.geneModules[network1].index[
                                self.geneModules[network1].moduleColors == module1].tolist()
                            list2 = self.geneModules[network2].index[
                                self.geneModules[network2].moduleColors == module2].tolist()
                            num = np.intersect1d(list1, list2)
                            fraction.loc[f"{network1}:{module1}", f"{network2}:{module2}"] = len(num) / len(list2) * 100
                else:
                    modules = self.geneModules[network1].moduleColors.unique().tolist()
                    for module in modules:
                        fraction.loc[f"{network1}:{module}", f"{network1}:{module}"] = 1.0
        self.fraction = fraction

        return fraction

    def calculatePvalue(self):
        """
        Calculate pvalue of fraction along multiple networks

        :return: dataframe containing pvalue between all modules in all netwroks
        :rtype: pandas dataframe
        """

        num = 0
        names = []
        for network in self.geneModules.keys():
            num = num + len(self.geneModules[network].moduleColors.unique())
            tmp = [f"{network}:" + s for s in self.geneModules[network].moduleColors.unique().tolist()]
            names = names + tmp
        pvalue = pd.DataFrame(0, columns=names, index=names)

        genes = []
        for network in self.geneModules.keys():
            genes = genes + self.geneModules[network].index.tolist()

        genes = list(set(genes))
        nGenes = len(genes)

        for network1 in self.geneModules.keys():
            for network2 in self.geneModules.keys():
                if network1 != network2:
                    modules1 = self.geneModules[network1].moduleColors.unique().tolist()
                    modules2 = self.geneModules[network2].moduleColors.unique().tolist()
                    for module1 in modules1:
                        for module2 in modules2:
                            list1 = self.geneModules[network1].index[
                                self.geneModules[network1].moduleColors == module1].tolist()
                            list2 = self.geneModules[network2].index[
                                self.geneModules[network2].moduleColors == module2].tolist()
                            number = self.fraction.loc[f"{network1}:{module1}", f"{network2}:{module2}"] * len(
                                list2) / 100
                            table = np.array(
                                [[nGenes - len(list1) - len(list2) + number, len(list1) - number],
                                 [len(list2) - number, number]])
                            oddsr, p = fisher_exact(table, alternative='two-sided')
                            pvalue.loc[f"{network1}:{module1}", f"{network2}:{module2}"] = p
        self.P_value = pvalue

        return pvalue

    def compareNetworks(self):
        """
        compare Networks
        """
        self.calculateJaccardSimilarity()
        self.calculateFraction()
        self.calculatePvalue()

    def plotHeatmapComparison(self,
                              color="jaccard_similarity",
                              row_cluster=True,
                              col_cluster=True,
                              save=True,
                              plot_show=True,
                              plot_format="pdf",
                              file_name="heatmap_comparison"):
        """
        plot heatmap comparison

        :param color: how to color heatmap (options: jaccard_similarity or fraction) default: jaccard_similarity
        :type color: str
        :param row_cluster: If True, cluster the rows. (default True)
        :type row_cluster: bool
        :param col_cluster: If True, cluster the columns. (default True)
        :type col_cluster: bool
        :param save: if you want to save plot as comparison.png near to your script
        :type save: bool
        :param plot_show: indicate if you want to show the plot or not (default: True)
        :type plot_show: bool
        :param plot_format: indicate the format of plot (default: pdf)
        :type plot_format: str
        :param file_name: name and path of the plot use for save (default: heatmap_comparison)
        :type file_name: str

        """
        if color == "jaccard_similarity":
            tmp1 = self.jaccard_similarity
        elif color == "fraction":
            tmp1 = self.fraction
        else:
            sys.exit("Color is not correct!")

        reds = cm.get_cmap('Reds', 256)
        newcolors = reds(np.linspace(0, 1, 256))
        white = np.array([255 / 256, 255 / 256, 255 / 256, 1])
        newcolors[:1, :] = white
        newcmp = ListedColormap(newcolors)

        np.fill_diagonal(tmp1.values, 0)
        labels = self.P_value.round(decimals=2)
        labels[tmp1 == 0] = ""
        labels = (np.asarray(["{0}".format(pvalue)
                              for pvalue in labels.values.flatten()])) \
            .reshape(self.jaccard_similarity.shape)

        sns.set(font_scale=1.5)
        res = sns.clustermap(tmp1, annot=labels, fmt="", cmap=newcmp,
                             row_cluster=row_cluster, col_cluster=col_cluster,
                             figsize=(tmp1.shape[0] + 3, tmp1.shape[1]),
                             annot_kws={'size': 20, "weight": "bold"})
        plt.setp(res.ax_heatmap.xaxis.get_majorticklabels(), fontsize=20, fontweight="bold", rotation=90)
        plt.setp(res.ax_heatmap.yaxis.get_majorticklabels(), fontsize=20, fontweight="bold")
        plt.yticks(rotation=0)
        res.fig.suptitle(f"comparison heatmap", fontsize=30, fontweight="bold")

        if save:
            res.savefig(f"{file_name}.{plot_format}")
        if plot_show:
            plt.show()
        else:
            plt.close()

    def plotJaccardSimilarity(self,
                              color=None,
                              cutoff=0.1,
                              figsize=None,
                              save=True,
                              plot_show=True,
                              plot_format="png",
                              file_name="jaccard_similarity"):
        """
        Plot jaccard similarity matrix as a network

        :param color: if you want to color nodes for each networks separately
        :type color: dict
        :param cutoff: threshold you used for filtering jaccard similarity
        :type cutoff: double
        :param figsize: indicate the size of plot (default is base on the number of nodes that pass cutoff)
        :type figsize: tuple of int
        :param save: indicate if you want to save the plot or not (default: True)
        :type save: bool
        :param plot_show: indicate if you want to show the plot or not (default: True)
        :type plot_show: bool
        :param plot_format: indicate the format of plot (default: png)
        :type plot_format: str
        :param file_name: name and path of the plot use for save (default: jaccard_similarity)
        :type file_name: str
        """
        df = self.jaccard_similarity
        np.fill_diagonal(df.values, 0)
        df = pd.DataFrame(df.stack())
        df.reset_index(inplace=True)
        df = df[df[0] >= cutoff]
        df.columns = ['source', 'dest', 'weight']
        if df.shape[0] == 0:
            print(f"{WARNING}None of the connections pass the cutoff{ENDC}")
            return None

        G = nx.from_pandas_edgelist(df, 'source', 'dest', 'weight')
        node_labels = {}
        nodes = list(G.nodes())
        for i in range(len(nodes)):
            node_labels[nodes[i]] = nodes[i].split(":")[1]
        edges = G.edges()
        weights = [G[u][v]['weight'] * 10 for u, v in edges]
        edge_labels = {}
        for u, v in edges:
            edge_labels[u, v] = str(round(G[u][v]['weight'], 2))

        color_map = []
        if color is None:
            color_map = None
        else:
            for node in G:
                color_map.append(color[node.split(":")[0]])

        if figsize is None:
            figsize = (len(G.nodes()) / 2, len(G.nodes()) / 2)
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        pos = nx.spring_layout(G, k=1 / math.sqrt(len(G.nodes()) / 2))
        nx.draw_networkx(G,
                         pos=pos,
                         node_color=color_map,
                         width=weights,
                         labels=node_labels,
                         font_size=8,
                         node_size=500,
                         with_labels=True,
                         ax=ax)

        nx.draw_networkx_edge_labels(G,
                                     pos,
                                     edge_labels=edge_labels,
                                     font_size=7)

        if color is not None:
            for label in color:
                ax.plot([0], [0], color=color[label], label=label)

        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(f"{file_name}.{plot_format}")
        if plot_show:
            plt.show()
        else:
            plt.close()

    def plotBubbleComparison(self,
                             bubble_size="jaccard_similarity",
                             cutoff=0.01,
                             color=None,
                             order1=None,
                             order2=None,
                             figsize=None,
                             save=True,
                             plot_show=True,
                             plot_format="png",
                             file_name="bubble_comparison"):
        """
        plot comparison matrix as a bubble plot

        :param bubble_size: which information you want to use for size of bubble (options: jaccard_similarity or fraction) default: jaccard_similarity
        :type bubble_size: str
        :param cutoff: threshold you used for defining significant comparison
        :type cutoff: double
        :param color: if you want to color tick labels for each networks separately
        :type color: dict
        :param order1: order of modules in PyWGCNA1 you want to show in plot (name of each elements should mapped the name of modules in your first PyWGCNA)
        :type order1: list of str
        :param order2: order of modules in PyWGCNA2 you want to show in plot (name of each elements should mapped the name of modules in your second PyWGCNA)
        :type order2: list of str
        :param figsize: indicate the size of plot (default is base on the number of modules)
        :type figsize: tuple of int
        :param save: if you want to save plot as comparison.png near to your script
        :type save: bool
        :param save: indicate if you want to save the plot or not (default: True)
        :type save: bool
        :param plot_show: indicate if you want to show the plot or not (default: True)
        :type plot_show: bool
        :param plot_format: indicate the format of plot (default: png)
        :type plot_format: str
        :param file_name: name and path of the plot use for save (default: jaccard_similarity)
        :type file_name: str
        """
        viridis = cm.get_cmap('viridis', 256)
        newcolors = viridis(np.linspace(0, 1, 256))
        grey = np.array([128 / 256, 128 / 256, 128 / 256, 1])
        newcolors[:round(256 * cutoff), :] = grey
        newcmp = ListedColormap(newcolors)

        P_value = self.P_value.copy(deep=True)
        P_value = -1 * np.log10(P_value.astype(np.float64))

        if bubble_size == "jaccard_similarity":
            size = self.jaccard_similarity.copy(deep=True)
        elif bubble_size == "fraction":
            size = self.fraction.copy(deep=True)
        else:
            sys.exit(f"bubble_size={bubble_size} is not correct!")
        np.fill_diagonal(size.values, 0)

        P_value[size == 0] = np.nan
        P_value.replace([np.inf], -1, inplace=True)
        P_value[P_value == -1] = P_value.max(numeric_only=True).max() + 1

        size[size == 0] = np.nan

        if order1 is not None:
            P_value = P_value.reindex(columns=order1)
            size = size.reindex(columns=order1)

        if order2 is not None:
            P_value = P_value.reindex(order2)
            size = size.reindex(order2)

        df = pd.DataFrame(size.stack())
        df.reset_index(inplace=True)
        df.columns = ['x', 'y', 'size']

        P_value = pd.DataFrame(P_value.stack())
        P_value.reset_index(inplace=True)
        df['-log10(P_value)'] = P_value[0].values.tolist()

        if figsize is None:
            figsize = (df.shape[0] / 70 + df.shape[0] / 400, df.shape[0] / 70)
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')

        ax = sns.scatterplot(data=df,
                             x="x",
                             y="y",
                             hue="-log10(P_value)",
                             size="size",
                             palette=newcmp)

        norm = plt.Normalize(0, df['-log10(P_value)'].max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])

        # Remove the legend and add a colorbar
        ax.get_legend().remove()
        fig.colorbar(sm, shrink=0.25, label='-log10(P_value)', ax=ax)

        fig.canvas.draw()

        if color is not None:
            xticks = []
            yticks = []
            for xtick, ytick in zip(ax.get_xticklabels(), ax.get_yticklabels()):
                tmp = xtick.get_text().split(":")
                xtick.set_color(color[tmp[0]])
                xticks.append(tmp[1])

                tmp = ytick.get_text().split(":")
                ytick.set_color(color[tmp[0]])
                yticks.append(tmp[1])

            ax.set_xticklabels(xticks, rotation=90)
            ax.set_yticklabels(yticks)
        else:
            ax.set_xticklabels(rotation=90)

        ax.set_xlabel('')
        ax.set_ylabel('')

        if bubble_size == "jaccard_similarity":
            legend_title = "Jaccard Index"
        elif bubble_size == "fraction":
            legend_title = "Fraction"

        handles, labels = ax.get_legend_handles_labels()
        entries_to_skip = labels.index('size') + 1
        handles = handles[entries_to_skip:]
        labels = labels[entries_to_skip:]
        #for h in handles[1:]:
        #    sizes = [s for s in h.get_sizes()]
        #    h.set_sizes(sizes)
        labels = labels[:1] + [f'{float(lab):.2f}' for lab in labels[1:]]
        legend_size = ax.legend(handles, labels,
                                title=legend_title,
                                bbox_to_anchor=(1, 1),
                                loc=2,
                                borderaxespad=0.,
                                frameon=False, 
                                title_fontsize=15)
        plt.gca().add_artist(legend_size)

        legend_elements = []
        if color is not None:
            for label in color:
                tmp = Line2D([0], [0], color=color[label], label=label)
                legend_elements.append(tmp)

            ax.legend(handles=legend_elements,
                      title='Networks',
                      bbox_to_anchor=(1.04, 0),
                      loc=3,
                      borderaxespad=0.,
                      frameon=False, 
                      title_fontsize=15)

        if save:
            plt.savefig(f"{file_name}.{plot_format}")
        if plot_show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def jaccard(list1, list2):
        """
        Calculate jaccard similarity matrix for two lists

        :param list1: first list containing the data
        :type list1: list
        :param list2: second list containing the data
        :type list2: list

        :return: jaccard similarity
        :rtype: double
        """
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union

    def saveComparison(self, name="comparison"):
        """
        save comparison object as comparison.p near to the script

        :param name: name of the pickle file (default: comparison.p)
        :type name: str
        
        """
        print(f"{BOLD}{OKBLUE}Saving comparison as {name}.p{ENDC}")

        picklefile = open(f"{name}.p", 'wb')
        pickle.dump(self, picklefile)
        picklefile.close()

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def logistic_model(data, cell_types, sparsity=0.2, fraction=0.5):
    X = (data / data.std()).dropna(1)
    X_train, X_test, y_train, y_test = \
    train_test_split(X, cell_types, test_size=fraction)

    lr = LogisticRegression(penalty='l1', C=sparsity)
    lr.fit(X_train, y_train)
    y_prob = lr.predict_proba(X_test)

    print((lr.coef_ > 0).sum(1))

    lr_res = pd.DataFrame.from_records(lr.coef_, columns=X.columns)

    return y_prob, y_test, lr_res, lr


def create_colors(lr):
    n_cts = lr.classes_.shape[0]
    color_norm = colors.Normalize(vmin=-n_cts / 3, vmax=n_cts)
    ct_arr = np.arange(n_cts)
    ct_colors = cm.YlOrRd(color_norm(ct_arr))

    return ct_colors


def plot_roc(y_prob, y_test, lr):
    ct_colors = create_colors(lr)

    for i, cell_type in enumerate(lr.classes_):
        fpr, tpr, _ = metrics.roc_curve(y_test == cell_type, y_prob[:, i])
        plt.plot(fpr, tpr, c=ct_colors[i], lw=2)

    plt.plot([0, 1], [0, 1], color='k', ls=':')
    plt.xlabel('FPR')
    plt.ylabel('TPR')


def get_top_markers(lr_res, N):
    top_markers = \
    lr_res \
        .assign(cluster=lr_res.index) \
        .melt(id_vars=['cluster'], var_name='gene', value_name='weight') \
        .sort_values('weight', ascending=False) \
        .groupby('cluster') \
        .head(N) \
        .sort_values(['cluster', 'weight'], ascending=[True, False])

    return top_markers


def plot_marker_map(data, cell_types, top_markers):
    sizes = pd.Series(cell_types, index=data.index).value_counts()
    c_idx = pd.Series(cell_types, index=data.index).map(sizes).sort_values(ascending=False).index

    g_idx = top_markers \
        .assign(sizes=top_markers.cluster.map(sizes)) \
        .sort_values('sizes', ascending=False) \
        .gene[::-1]

    marker_map = data.reindex(index=c_idx, columns=g_idx)

    plt.pcolormesh(marker_map.T, cmap=cm.gray_r)
    plt.colorbar(label='Expression')

    for vl in sizes.cumsum():
        plt.axvline(vl, lw=0.66, c='r')

    plt.ylabel('Marker genes')
    plt.xlabel('Cells')


def plot_marker_table(top_markers, lr, n_columns=5, max_rows=10):
    ct_colors = create_colors(lr)

    for i, m in enumerate(top_markers['cluster'].unique()):
        plt.subplot(max_rows, n_columns, i + 1)
        g = top_markers.query('cluster == @m')
        plt.title(m, size=12, weight='bold', ha='left')
        for j, gn in enumerate(g.iterrows()):
            _, gn = gn
            plt.annotate(f'{gn.weight:.2f} - {gn.gene}', (0, 0.2 * j), )

        plt.axis('off')
        plt.ylim((n_columns + 1) * 0.2, -0.2)
        ax = plt.gca()
        ax.plot([0.5, 1], [1, 1], transform=ax.transAxes, lw=3, c=ct_colors[i])

    plt.tight_layout()


def html_marker_table(top_markers, n_columns=5):
    html_str = '<table>'
    for i, m in enumerate(top_markers['cluster'].unique()):
        if i % n_columns == 0:
            html_str += '<tr>'
            
        html_str += '<td style="text-align: left;"><pre>'
        g = top_markers.query('cluster == @m')
        html_str += f'{m}\n-----\n'
        for j, gn in enumerate(g.iterrows()):
            _, gn = gn
            if gn.weight == 0:
                html_str += '\n'
                continue

            html_str += f'{gn.weight:.2}\t{gn.gene}\n'
            
        html_str += '</pre></td>'
        
        if i % n_columns == n_columns - 1:
            html_str += '</tr>'
            
    html_str += '</table>'
        
    return html_str


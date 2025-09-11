import matplotlib.pyplot as plt
import numpy as np

def plotcluster2(X, lab):
    """
      This function plots the embedding.

      Parameters are:

      'X'      - N by D matrix. Each row in X represents an observation.
      'lab'    - True annotations of data X.

    """
    if np.all(lab >= 0) and np.all(lab.astype(int) == lab):
        colors = [[180, 180, 180], [31, 119, 179], [251, 130, 20], [43, 159, 46], [210, 33, 33],
                  [143, 99, 187], [140, 87, 76], [255, 116, 192], [125, 125, 125],
                  [184, 187, 29], [30, 191, 208], [218, 165, 32], [65, 105, 225],
                  [255, 99, 71], [147, 112, 219], [255, 215, 0], [50, 205, 50],
                  [174, 199, 232], [255, 187, 120], [152, 223, 138], [255, 152, 150],
                  [196, 177, 213], [196, 155, 147], [219, 219, 141], [135, 206, 235],
                  [255, 165, 0], [144, 238, 144], [1, 0, 0], [0, 0, 1], [0, 1, 0],
                  [1, 1, 0], [1, 0, 1], [0, 1, 1], [160, 0, 160], [12, 128, 144],
                  [255, 69, 0], [140, 86, 75], [160, 82, 45], [0, 139, 139],
                  [175, 238, 238], [233, 150, 122], [143, 188, 143], [106, 90, 205],
                  [60, 179, 113], [220, 20, 60], [65, 105, 225], [147, 112, 219],
                  [20, 206, 209]]
        colors = [[a[0]/255, a[1]/255, a[2]/255] for a in colors]

        for i in range(len(lab)):
            if lab[i] <= 40:
                plt.scatter(X[i, 0], X[i, 1], c=[colors[int(lab[i])]], s=3)
            elif lab[i] > 40:
                plt.scatter(X[i, 0], X[i, 1], c=[colors[41 + int(lab[i] - 41 % 7)]], s=3)
        plt.show()
    else:
        print('WARNING: clustering annotation must be a non-negative integer!')




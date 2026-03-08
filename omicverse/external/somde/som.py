
import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
import logging

from matplotlib.colors import LinearSegmentedColormap
from .util import *



class SomNode:
    def __init__(self, X, k,homogeneous_codebook=True):
        import somoclu
        self.X = X
        self.somn = int(np.sqrt(X.shape[0]//k))
        self.ndf = None
        self.ninfo = None
        self.nres=None
        if homogeneous_codebook:
            xmin,ymin = X.min(0)
            xmax,ymax = X.max(0)
            cobx,coby = np.meshgrid(np.linspace(xmin,xmax,self.somn),np.linspace(ymin,ymax,self.somn))
            self.inicodebook = np.transpose(np.array([cobx.ravel(),coby.ravel()],np.float32),(1,0))
            print('using {0}*{0} SOM nodes for {1} points'.format(self.somn,X.shape[0]))
            self.som = somoclu.Somoclu(self.somn, self.somn,initialcodebook=self.inicodebook.copy())
        else:
            self.som = somoclu.Somoclu(self.somn, self.somn)
        self.som.train(X,epochs=10)
        
    def reTrain(self,ep):
        self.som.train(self.X,epochs=ep)
        
        
    def viewIniCodebook(self,orisize=3,codesize=20):
        plt.scatter(self.X[:,0], self.X[:,1],s=orisize, label='original')
        plt.scatter(self.inicodebook[:,0], self.inicodebook[:,1],s=codesize, label='initialCodebook')
        plt.legend()
        
        
    def view(self,raw = True,c=False,line=False):
        rr = self.som.codebook
        sizenum = np.ones([rr.shape[0],rr.shape[1]])*30
        rr = np.reshape(rr,[-1,2])
        if raw:
            plt.scatter(self.X[:,0], self.X[:,1],s=3, label='original')
        # Vectorize sizenum accumulation
        vs = self.som.bmus[:, 0]
        us = self.som.bmus[:, 1]
        np.add.at(sizenum, (us, vs), 2)
        if line:
            for i in range(self.X.shape[0]):
                v, u = self.som.bmus[i]
                plt.plot([self.X[i,0],self.som.codebook[u,v,0]],[self.X[i,1],self.som.codebook[u,v,1]])
        sizenum = np.reshape(sizenum,[-1,])
        if c:
            plt.scatter(rr[:,0],rr[:,1],s=sizenum,label=str(self.somn)+'X'+str(self.somn)+' SOM nodes',c=sizenum,cmap='hot')
            plt.colorbar()
        else:
            plt.scatter(rr[:,0],rr[:,1],s=sizenum,label=str(self.somn)+'X'+str(self.somn)+' SOM nodes',c='r')
        plt.legend()
        
    def mtx(self,df,alpha=0.5):
        bsmc = self.som.bmus
        # Vectorize flat-index computation: u=bsmc[:,0], v=bsmc[:,1]
        soml = bsmc[:, 1] * self.somn + bsmc[:, 0]

        vals = df.values                 # (G, N_spots) numpy array — avoids pandas overhead
        unique_nodes = np.unique(soml)
        n_nodes = len(unique_nodes)
        G = vals.shape[0]

        ndf_value = np.zeros((G, n_nodes))
        xs = np.empty(n_nodes)
        ys = np.empty(n_nodes)

        for tmp, node in enumerate(unique_nodes):
            mask = soml == node
            cols = vals[:, mask]         # (G, count_i)
            ndf_value[:, tmp] = alpha * cols.max(1) + (1 - alpha) * cols.mean(1)
            coor = self.som.codebook[node // self.somn, node % self.somn]
            xs[tmp] = round(float(coor[0]), 4)
            ys[tmp] = round(float(coor[1]), 4)

        ndf = pd.DataFrame(ndf_value, index=list(df.T))
        # Build ninfo as dict — avoids slow row-by-row loc assignment
        ninfo = pd.DataFrame({'x': xs, 'y': ys, 'total_count': ndf.sum(0).values})
        self.ndf = ndf
        self.ninfo = ninfo
        return ndf, ninfo
    
    def norm(self):
        if self.ninfo is None:
            print('generate mtx first')
            return
        dfm = stabilize(self.ndf)
        self.nres = regress_out(self.ninfo, dfm, 'np.log(total_count)').T
        return self.nres
    

    
    def Sparun(self, X, exp_tab, kernel_space=None, n_jobs=1, use_tqdm=True):
        '''Perform SpatialDE test.'''
        if kernel_space is None:
            l_min, l_max = get_l_limits(X)
            kernel_space = {
                'SE': np.logspace(np.log10(l_min), np.log10(l_max), 10),
                'const': 0,
            }

        logging.info('Performing DE test')
        results = dyn_de(X, exp_tab, kernel_space, n_jobs=n_jobs, use_tqdm=use_tqdm)
        mll_results = get_mll_results(results)

        mll_results['pval'] = 1 - stats.chi2.cdf(mll_results['LLR'], df=1)
        mll_results['qval'] = qvalue(mll_results['pval'])

        return mll_results

    def run(self, n_jobs=1, use_tqdm=True):
        if self.nres is None:
            print('norm mtx first')
            self.norm()
        X = self.ninfo[['x', 'y']].values.astype(float)
        result = self.Sparun(X, self.nres, n_jobs=n_jobs, use_tqdm=use_tqdm)
        result.sort_values('LLR', inplace=True, ascending=False)
        number_q = result[result.qval < 0.05].shape[0]
        return result, number_q


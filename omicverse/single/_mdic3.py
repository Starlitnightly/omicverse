import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import scipy.stats as stat
import math
import os
import scipy
import time
import scipy.sparse as sp
from scipy.stats import pearsonr




def readexp(expression_txt):
    f = open(expression_txt)
    gene = []
    gene_exp = {}
    cellname = []
    AA = []
    flag = 0
    for p in f:
        t = p.split()
        flag += 1
        if flag == 1:
            cellname = t
            # cellnum = len(t)
            continue
        gene.append(t[0])
        gene_exp[t[0]] = [float(t[i]) for i in range(1, len(t))]
        tt = list(map(float, t[1:]))
        AA.append(tt)
    f.close()
    
    return AA,gene_exp,cellname


def readlabel(labelname):

    type_cell = {}
    f = open(labelname)
    for p in f:
        t = p.split('\t')
        t[1] = t[1].split('\n')[0]
        if t[1] not in type_cell.keys():
            type_cell[t[1]] = []
        type_cell[t[1]].append(t[0])
    f.close()
    
    f = open(labelname)
    label = []
    labels = set()
    for p in f:
        t = p.split()
        label.append(t[1])
        labels.add(str(t[1]))
    labelsl = list(labels)
    label_index = {}
    for i in labelsl:
        label_index[i] = []
    flag = 0
    for i in label:
        flag += 1
        label_index[i].append(flag - 1)
    f.close()
    
    return labelsl,label_index,type_cell



def LassoRegression(degree, alpha):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("lasso_reg", Lasso(alpha=alpha))])


def lasso_regress(gene_1, gene_2, degree, alpha):

    le_g1 = len(gene_1)
    t = np.linspace(min(gene_1), max(gene_1), num=le_g1)
    a1, b1 = (np.array(x).reshape(-1, 1) for x in zip(*sorted(zip(gene_1, gene_2), key=lambda x: x[0])))
    lasso1_reg = LassoRegression(degree, alpha)
    lasso1_reg.fit(a1, t)
    gene1 = lasso1_reg.predict(a1)
    gene2 = lasso1_reg.predict(b1)

    return gene1, gene2


def rss_calculate(model, right_v, left1_v, left2_v):

    model.fit(left1_v, right_v)
    rssu_1 = np.sum((model.predict(left1_v) - right_v) ** 2)
    model.fit(left2_v, right_v)
    rssr_1 = np.sum((model.predict(left2_v) - right_v) ** 2)

    return rssu_1, rssr_1


def granger(gene1, gene2, cellnum):
    import statsmodels.api as sm
    sort_1 = list(sorted(zip(gene1, gene2), key=lambda x: x[0]))

    a1, b1 = (list(x) for x in zip(*sort_1))
    a1_cha = ((300 * pd.Series(a1)) / 90001).tolist()
    b1_cha = ((300 * pd.Series(b1)) / 90001).tolist()

    model = LinearRegression()
    b1_t_1 = np.array(b1_cha[0:cellnum - 1])
    a1_t1_t2_b1_1 = sm.add_constant(np.array(pd.concat([pd.Series(a1_cha),
                                                        pd.Series(b1_cha)], axis=1)[1:]))
    b1_t1_t2_1 = sm.add_constant(np.array(b1_cha[1:cellnum]))
    b1_rssu_1, b1_rssr_1 = rss_calculate(model, b1_t_1, a1_t1_t2_b1_1, b1_t1_t2_1)
    f1_1 = ((b1_rssr_1 - b1_rssu_1) / 1) / (b1_rssu_1 / (cellnum - 3))
    p1_1 = scipy.stats.f.sf(f1_1, 1, cellnum - 3)

    b1_t = np.array(b1_cha[0:cellnum - 2])
    a1_t1_t2_b1 = sm.add_constant(np.array(pd.concat([pd.Series(a1_cha)[1:cellnum - 1].reset_index(drop=True),
                                                      pd.Series(a1_cha)[2:cellnum].reset_index(drop=True),
                                                      pd.Series(b1_cha)[1:cellnum - 1].reset_index(drop=True),
                                                      pd.Series(b1_cha)[2:cellnum].reset_index(drop=True)], axis=1)))
    b1_t1_t2 = sm.add_constant(np.array(pd.concat([pd.Series(b1_cha)[1:cellnum - 1].reset_index(drop=True),
                                                   pd.Series(b1_cha)[2:cellnum].reset_index(drop=True)], axis=1)))
    b1_rssu_2, b1_rssr_2 = rss_calculate(model, b1_t, a1_t1_t2_b1, b1_t1_t2)
    f1_2 = ((b1_rssr_2 - b1_rssu_2) / 2) / (b1_rssu_2 / (cellnum - 2 - 2 - 1))
    p1_2 = scipy.stats.f.sf(f1_2, 2, cellnum - 2 - 2 - 1)

    b1_t_3 = np.array(b1_cha[0:cellnum - 3])
    a1_t1_t2_b1_3 = sm.add_constant(np.array(pd.concat([pd.Series(a1_cha)[1:cellnum - 2].reset_index(drop=True),
                                                        pd.Series(a1_cha)[3:cellnum].reset_index(drop=True),
                                                        pd.Series(a1_cha)[2:cellnum - 1].reset_index(drop=True),
                                                        pd.Series(b1_cha)[1:cellnum - 2].reset_index(drop=True),
                                                        pd.Series(b1_cha)[3:cellnum].reset_index(drop=True),
                                                        pd.Series(b1_cha)[2:cellnum - 1].reset_index(drop=True)],
                                                       axis=1)))
    b1_t1_t2_3 = sm.add_constant(np.array(pd.concat([pd.Series(b1_cha)[1:cellnum - 2].reset_index(drop=True),
                                                     pd.Series(b1_cha)[3:cellnum].reset_index(drop=True),
                                                     pd.Series(b1_cha)[2:cellnum - 1].reset_index(drop=True)], axis=1)))
    b1_rssu_3, b1_rssr_3 = rss_calculate(model, b1_t_3, a1_t1_t2_b1_3, b1_t1_t2_3)
    f1_3 = ((b1_rssr_3 - b1_rssu_3) / 3) / (b1_rssu_3 / (cellnum - 7))
    p1_3 = scipy.stats.f.sf(f1_3, 3, cellnum - 7)

    p_1 = min([p1_1, p1_2, p1_3])

    return p_1




def granger_(gene1, gene2, cellnum):
    import statsmodels.api as sm
    from sklearn.linear_model import LinearRegression
    from statsmodels.tsa.stattools import grangercausalitytests
    # Convert gene lists to NumPy array and sort by gene1
    genes = np.array(sorted(zip(gene1, gene2), key=lambda x: x[0]))
    a1_cha, b1_cha = (300 * genes / 90001).T  # Vectorized scaling

    # Initialize the LinearRegression model
    model = LinearRegression()
    
    # Prepare the variables and pre-compute as much as possible
    p_values = []
    for lag in range(1, 4):
        lag_slice = slice(lag, cellnum)
        reset_index = lambda series: series.reset_index(drop=True)
        
        # Previous values (up to lag) for regression calculation
        b1_t = b1_cha[:-lag]
        
        # Lagged values of a1 and b1
        lagged_a1 = [reset_index(pd.Series(a1_cha)[i:cellnum - lag]) for i in range(lag)]
        lagged_b1 = [reset_index(pd.Series(b1_cha)[i:cellnum - lag]) for i in range(lag)]

        # Combine arrays for the regression
        a1_t1_t2_b1 = sm.add_constant(np.column_stack(lagged_a1 + lagged_b1))
        b1_t1_t2 = sm.add_constant(np.column_stack(lagged_b1))
        
        # Calculate RSS for unrestricted and restricted models
        rssu, rssr = rss_calculate(model, b1_t, a1_t1_t2_b1, b1_t1_t2)

        # Compute the F-statistic
        f_stat = ((rssr - rssu) / lag) / (rssu / (cellnum - lag * 2 - 1))

        # Compute the p-value for the F-statistic
        p_value = scipy.stats.f.sf(f_stat, lag, cellnum - lag * 2 - 1)
        p_values.append(p_value)

    # Return the minimum p-value from the tests for each lag
    return min(p_values)




def calculate_block(spaceblocki, gene, AA, gene_exp, ax_nl, cellnum, gene_num,pear):
    i1, i2, j1, j2 = [int(spaceblocki[i]) for i in range(len(spaceblocki))]
   # print(i1,i2,j1,j2)
    #PP = sp.dok_matrix((gene_num,gene_num))
    # Convert to more efficient sparse matrix format for arithmetics
    PP = sp.lil_matrix((gene_num, gene_num))

    # Precompute the maximum non-one value in pear multiplied by 0.3
    pear_max_multiplied = np.max(np.abs(pear - np.eye(*pear.shape)), axis=1) * 0.3
    
    for i in range(i1, i2):
        for j in range(j1, j2):
            if i == j:
                continue
            

            """
            entry_11 = gene[i]
            entry_21 = gene[j]
            mp = list(pear[i])
            if 1 in pear[i]:
                mp.remove(1)
            mp = abs(np.array(mp))
            mpmax = max(mp)
            if abs(stat.pearsonr(AA[i],AA[j])[0]) < mpmax * 0.3:
                continue
            if abs(pear[i,j]) < mpmax * 0.3:
                continue
            """
            # Skip calculation when condition is not met
            if abs(pear[i, j]) < pear_max_multiplied[i]:
                continue

            Gene1, Gene2 = lasso_regress(gene_exp[gene[i]], gene_exp[gene[j]], 30, ax_nl)

            if (Gene1 == Gene2).all() == True:
                continue

            p = granger(Gene1, Gene2, cellnum)
           # print(i,j,p)

            Pi = gene.index(gene[i])
            Pj = gene.index(gene[j])
            PP[Pi,Pj] += p
            # Update the PP matrix
            #PP[i, j] += p

    return PP.todok()


def celltype_score(labelsl, label_index, M):

    llen = len(labelsl) 
    N = np.zeros((llen, llen))
    for l1 in labelsl:
        I = label_index[l1]
        for l2 in labelsl:
            J = label_index[l2]
            if l1 == l2:
                for i in I:
                    for j in J:
                        N[labelsl.index(l1)][labelsl.index(l2)] += abs(M[i][j])
            if l1 != l2:
                for i in I:
                    for j in J:
                        if M[i][j] >= 0:
                            N[labelsl.index(l1)][labelsl.index(l2)] += M[i][j]
                        if M[i][j] < 0:
                            N[labelsl.index(l2)][labelsl.index(l1)] += abs(M[i][j])
    Nn = np.zeros((llen, llen))
    for i in range(len(N)):
        for j in range(len(N)):
            if N[i][j] == 0:
                continue
            Nn[i][j] += math.log(N[i][j], 10)

    return Nn



def GRN_GNIPLR(AA,gene_exp,step,process):
    
        print(time.asctime()+ ': Start calculating the GRN adjacency matrix.')
    
        
        gene_num = len(gene_exp)
        gene = list(gene_exp.keys())
        cellnum = len(AA[0])
    
        '''
        Pear = np.zeros((len(AA),len(AA)))
        for i in range(len(AA)):
            for j in range(i,len(AA)):
                Pear[i][j] += stat.pearsonr(AA[i], AA[j])[0]
        Pear = np.where(Pear,Pear,Pear.T)
        '''
        
        Pear = np.corrcoef(AA)
        print(time.asctime()+ ': Pearson correlation coefficient calculation completed.')
    
        space = list(np.linspace(start=0, stop=step * int(gene_num / step), num=int(gene_num / step) + 1))
        if (gene_num / step == int(gene_num / step)) == False:
            space.append(gene_num)
        space1 = []
        for i in range(len(space) - 1):
            space1.append((space[i], space[i + 1]))
        space_block = []
        for i in space1:
            for j in space1:
                space_block.append(i + j)
        print(time.asctime()+ ': GRN subprocesses start: ',len(space_block))     
        import multiprocessing
        pool = multiprocessing.Pool(process)
    
        GRN = sp.dok_matrix((gene_num, gene_num))
    
        rest = []
        from tqdm import tqdm
        for si in tqdm(space_block):
            rest.append(pool.apply_async(func=calculate_block, args=(si, gene, AA, gene_exp, 0.1, cellnum, gene_num,Pear,)))

        print(time.asctime()+ ': Waiting for all GRN subprocesses done...')
        for r in tqdm(rest):
            GRN+=r.get()
    
        print(time.asctime()+ ': Waiting for all subprocesses done...')
        pool.close()
        pool.join()
    
        print(time.asctime()+ ": Complete the GRN calculation.")

        GRN = GRN.toarray()
        
        return GRN



def MDIC3_score(AA, GRN,labelsl, label_index):
    
    begin=time.asctime()
    
    print(begin+ ': Start calculating cell-cell communication.')

    U1, S1, V1 = np.linalg.svd(AA)
    shapeA = np.shape(AA)
    d = min(shapeA) - len(S1)
    dd = []
    for i in range(d):
        dd.append(0)
    S11 = np.append(S1, dd)

    S11 = np.diag(S11)
    if shapeA[0] < shapeA[1]:
        b = np.zeros((shapeA[0], shapeA[1] - shapeA[0]))
        S11 = np.column_stack((S11, b))
    if shapeA[0] > shapeA[1]:
        b = np.zeros((shapeA[0] - shapeA[1], shapeA[1]))
        S11 = np.vstack((S11, b))

    T = np.dot(GRN, S11)
    Tp = np.linalg.pinv(T)
    ccc_adjacency = np.dot(Tp, AA)
    type_adjacency = celltype_score(labelsl, label_index, ccc_adjacency)
    print(time.asctime()+ ": Complete the cell-cell communication calculation.")

    return ccc_adjacency,type_adjacency
    
    
    
def MDIC3_scoresave(ccc_adjacency,type_adjacency,labelsl):

    print('Saving cell-cell communication results...')
    
    fw = open('celltype_communication.txt', 'w')
    for tt in range(len(labelsl)):
        if tt < len(labelsl) - 1:
            fw.write(labelsl[tt] + '\t')
        if tt == len(labelsl) - 1:
            fw.write(labelsl[tt] + '\n')
    for i in range(len(type_adjacency)):
        Ni = list(type_adjacency[i])
        fw.write(str(labelsl[i]) + '\t')
        for ni in range(len(Ni)):
            if ni < len(Ni) - 1:
                fw.write(str(Ni[ni]) + '\t')
            if ni == len(Ni) - 1:
                fw.write(str(Ni[ni]) + '\n')
    fw.close()
    
    np.savetxt('cellular_communication.txt', ccc_adjacency)
    
    
    





def readLRDB(LRDB_txt):
    L = []
    R = []
    LR = []
    f = open(LRDB_txt)
    flag = 0
    for p in f:
        flag += 1
        if flag == 1:
            continue
        t1 = p.split('\n')[0]
        LR.append(t1)
        tlr = t1.split(' - ')
        tlr[0] = tlr[0].split(' ')[0]
        l = []
        r = []
        l.append(tlr[0])
        L.append(l)
        if tlr[1][0] == '(':
            tr = tlr[1].split('+')
            r.append(tr[0][1:])
            r.append(tr[1][:-1])
            R.append(r)
        else:
            r.append(tlr[1])
            R.append(r)

    f.close()
    
    return L,R,LR



def check_zero_majority(nums):
    count_zeros = sum(1 for num in nums if num == 0)

    return count_zeros > len(nums) * 0.7


def pearson(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0
    else:
        corr, p = pearsonr(x, y)
        return corr, p


def FDRP(lrcorr):
    
    from statsmodels.stats.multitest import multipletests

    p_values = [v[1] for v in lrcorr.values()]
    corrected_p_values = multipletests(p_values, method='bonferroni')[1]
    p_LR_corr = {}
    index = 0
    for key, value in lrcorr.items():
        p_LR_corr[key] = [value[0], corrected_p_values[index]]
        index += 1
    p2_LR_corr = {key: value for key, value in p_LR_corr.items() if value[1] <= 0.01}

    return p2_LR_corr



def LR_exp(LR,L,R,geneexp,type_cell,cellname):
    
    celltype = list(type_cell.keys())
    typeindex = {}
    for i in celltype:
        cell0 = type_cell[i]
        cell0index = []
        for j in cell0:
            cell0index.append(cellname.index(j))
        typeindex[i] = cell0index
    
    gene2exp = {}
    for i in range(len(LR)):
    
        if len(L[i]) == 1 and L[i][0] not in geneexp.keys():
            continue
    
        if len(R[i]) == 1 and R[i][0] not in geneexp.keys():
            continue
    
        if len(R[i]) > 1:
            for r in R[i]:
                r0 = R[i][0]
                r1 = R[i][1]
            if r0 not in geneexp.keys():
                continue
            if r1 not in geneexp.keys():
                continue
    
        if len(L[i]) == 1:
            lexp = geneexp[L[i][0]]
            gene2exp[L[i][0]] = geneexp[L[i][0]]
    
        if len(R[i]) == 1:
            rexp = geneexp[R[i][0]]
            gene2exp[R[i][0]] = geneexp[R[i][0]]
    
        if len(R[i]) > 1:
            for r in R[i]:
                r0 = R[i][0]
                r1 = R[i][1]
            gene2exp[r0] = geneexp[r0]
            gene2exp[r1] = geneexp[r1]
    
    geneexp2 = {}
    for k in gene2exp.keys():
        n = 0
        exp = gene2exp[k]
        kexp = []
        for tt in celltype:
            for t in typeindex[tt]:
                kexp.append(exp[t])
            count = sum(1 for num in kexp if num > 0)
            if count > len(kexp) * 1 / 2:
                n += 1
        if n >= len(type_cell) / 2:
            continue
        geneexp2[k] = gene2exp[k]
    
    LRexp = {}
    for i in range(len(LR)):
        if len(L[i]) == 1 and L[i][0] not in geneexp2.keys():
            continue
        if len(R[i]) == 1 and R[i][0] not in geneexp2.keys():
            continue
        if len(R[i]) > 1:
            for r in R[i]:
                r0 = R[i][0]
                r1 = R[i][1]
            if r0 not in geneexp2.keys():
                continue
            if r1 not in geneexp2.keys():
                continue
        LRexp[LR[i]] = []
        if len(L[i]) == 1:
            lexp = geneexp2[L[i][0]]
        if len(R[i]) == 1:
            rexp = geneexp2[R[i][0]]
        if len(R[i]) > 1:
            for r in R[i]:
                r0 = R[i][0]
                r1 = R[i][1]
            r01 = geneexp2[r0] + geneexp2[r1]
            rexp = [x / 2 for x in r01]
        LRexp[LR[i]].append(lexp)
        LRexp[LR[i]].append(rexp)
    
    return LRexp
    
   
def MDIC3_SortLR(L,R,LR,target,type_cell,cellname,geneexp):
    
    
    LRexp = LR_exp(LR,L,R,geneexp,type_cell,cellname)
    cell0 = type_cell[target[0]]
    cell0index = []
    for i in cell0:
        cell0index.append(cellname.index(i))
    cell1 = type_cell[target[1]]
    cell1index = []
    for i in cell1:
        cell1index.append(cellname.index(i))

    targetexp = {}
    for key, value in LRexp.items():
        new_lists = []

        list0 = [value[0][i] for i in cell0index]
        list1 = [value[1][i] for i in cell1index]

        list0.sort(reverse=True)
        list1.sort(reverse=True)

        result0 = check_zero_majority(list0)
        result1 = check_zero_majority(list1)

        if result0 == True:
            continue
        if result1 == True:
            continue

        if len(list0) < len(list1):
            list1 = list1[:len(list0)]
        else:
            list0 = list0[:len(list1)]

        new_lists.append(list0)
        new_lists.append(list1)

        targetexp[key] = new_lists

    LRcorr = {}
    LRPcorr = {}
    for key, value in targetexp.items():
        x = value[0]
        y = value[1]
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        correlation, p = pearsonr(x, y)
        LRPcorr[key] = [correlation, p]
        if p > 0.05:
            continue
        LRcorr[key] = [correlation, p]

    p2_LR_corr = FDRP(LRcorr)
    from operator import itemgetter
    sorted_LRcorr = dict(sorted(p2_LR_corr.items(), key=itemgetter(1), reverse=True))
    

    return sorted_LRcorr



def MDIC3_LRsave(sorted_LRcorr):

    print('Saving LR identification results...')

    fw = open('target_LR.txt', 'w')
    fw.write('L_R_pair' + '\t' + 'pearson_corr' + '\t' + 'P_value' + '\n')
    for k in sorted_LRcorr.keys():
        fw.write(str(k) + '\t')
        kvalue = sorted_LRcorr[k]
        fw.write(str(kvalue[0]) + '\t' + str(kvalue[1]) + '\n')
    fw.close()



class pyMDIC3(object):
    
    def __init__(self,adata,clusters):
        AA1=adata.X.T
        from scipy.sparse import issparse, csr_matrix
        if issparse(AA1):
            AA1=AA1.toarray()
        gene_exp1={}
        for gene in adata.var_names:
            gene_exp1[gene]=adata[:,gene].to_df().values.reshape(-1).tolist()
        cellname1=adata.obs_names.tolist()
        
        labels1=list(set(adata.obs[clusters]))
        label_index1={}
        for idx,cellobs in enumerate(adata.obs[clusters].tolist()):
            if cellobs not in label_index1.keys():
                label_index1[cellobs]=[idx]
            else:
                label_index1[cellobs].append(idx)
        label_cell1={}
        for cellobs in labels1:
            label_cell1[cellobs]=adata.obs.loc[adata.obs[clusters]==cellobs].index.tolist()
            
        self.adata=adata
        self.AA=AA1
        self.gene_exp=gene_exp1
        self.cellname=cellname1
        self.labels=labels1
        self.label_index=label_index1
        self.label_cell=label_cell1
            
    def GRN_train(self,step=15,process=4):
        GRN = GRN_GNIPLR(self.AA, self.gene_exp, step, process)
        self.GRN=GRN
        return GRN
    
    def CCC_cal(self,):
        # Infer the cell-cell communication
        ccc_adjacency, type_adjacency = MDIC3_score(self.AA, self.GRN, self.labels, self.label_index)
        return ccc_adjacency, type_adjacency
import os
import numpy as np
from scipy.stats import rankdata
from scipy.special import binom #faster than comb
from ._df_apply import gen_row,gen_col,apply
from functools import partial



chunkSize = 100

#Function to index flat matrix as squareform matrix
def dist_index(i, j, matrix):
    
    if i == j:
        return(0.0)
    
    l = len(matrix)
    n = 0.5*(np.sqrt((8*l)+1)+1)
    index = int(l - binom(n-min(i, j), 2) + (max(i, j) - min(i, j) - 1))
    
    return(matrix[index])
    

#Function to index flat matrix as squareform matrix
def dist_multi_index(_array, matrix):
    
    results = np.zeros((len(_array), len(_array)))
    for i in range(len(_array)):
        for j in range(i, len(_array)):
            score = dist_index(_array[i], _array[j], matrix)
            results[i,j] = score
            results[j,i] = score
    
    return(results)
    

#Function to index rows of flat matrix as squareform matrix
def get_rows(_array, matrix):
    
    l = len(matrix)
    n = int(0.5*(np.sqrt((8*l)+1)+1))
    
    if _array.dtype != "bool":
        results = np.zeros((len(_array), n))
    
        for row, i in enumerate(_array):
            for j in range(n):
                index = int(l - binom(n - min(i, j), 2) + (max(i, j) - min(i, j) - 1))
                results[row,j] = matrix[index]
    
        return(results)
        
    else:
        results = np.zeros((np.sum(_array), n))
        row = 0
        for i, b in enumerate(_array):
            if b == True:
                for j in range(n):
                    index = int(l - binom(n - min(i, j), 2) + (max(i, j) - min(i, j) - 1))
                    results[row, j] = matrix[index]
                
                row += 1
                
        return(results)
    
    
#The following are supporting function for GetClusters.
def CoreSize(BranchSize, minClusterSize):
    
    BaseCoreSize = minClusterSize / 2 + 1
    if BaseCoreSize < BranchSize:
        CoreSize = BaseCoreSize + np.sqrt(BranchSize - BaseCoreSize)
    else:
        CoreSize = BranchSize
    
    return(int(CoreSize))
    

# This assumes the diagonal of the distance matrix
# is zero, BranchDist is a square matrix whose dimension is at least 2.
def CoreScatter(BranchDist, minClusterSize):
    
    nPoints = BranchDist.shape[0]
    PointAverageDistances = np.sum(BranchDist, axis=1) / (nPoints - 1)
    CoreSize = minClusterSize / 2 + 1

    if CoreSize < nPoints:
        EffCoreSize = CoreSize + np.sqrt(nPoints - CoreSize)
        order = np.argsort(PointAverageDistances)
        Core = order[np.arange(EffCoreSize)]
    else:
        Core = np.arange(nPoints)
        EffCoreSize = nPoints

    CoreAverageDistances = np.sum(BranchDist[Core, Core], axis=1) / (EffCoreSize - 1)
    
    return(np.mean(CoreAverageDistances))
    

def interpolate(data, index):
    
    i = np.round(index)
    n = len(data)
    if i < 0: return(data[0])
    if i >= n: return(data[-1])

    r = index - i
        
    return(data[i-1] * (1 - r) + data[i] * r)


def cutreeHybrid(link, distM,
                 cutHeight = None, minClusterSize = 20, deepSplit = 1,
                 maxCoreScatter = None, minGap = None,
                 maxAbsCoreScatter = None, minAbsGap = None,
                 minSplitHeight = None, minAbsSplitHeight = None,
                 externalBranchSplitFnc = None, minExternalSplit = None,
                 externalSplitOptions = [],
                 externalSplitFncNeedsDistance = None,
                 assumeSimpleExternalSpecification = True,
                 pamStage = True, pamRespectsDendro = True,
                 useMedoids = False,
                 maxPamDist = None,
                 respectSmallClusters = True,
                 verbose = 2, indent = 0):
    r"""Perform dynamic tree cutting for hierarchical clustering.
    
    This function implements the dynamic tree cutting algorithm for identifying
    clusters in hierarchical clustering dendrograms.
    
    Arguments:
        link: Linkage matrix from hierarchical clustering
        distM: Distance matrix used for clustering
        cutHeight: Height at which to cut the dendrogram (default: None, auto-calculated)
        minClusterSize: Minimum cluster size (default: 20)
        deepSplit: Controls sensitivity of cluster splitting (default: 1, range: 0-4)
        maxCoreScatter: Maximum scatter for cluster cores (default: None)
        minGap: Minimum gap between clusters (default: None)
        maxAbsCoreScatter: Absolute maximum core scatter (default: None)
        minAbsGap: Absolute minimum gap (default: None)
        minSplitHeight: Minimum split height (default: None)
        minAbsSplitHeight: Absolute minimum split height (default: None)
        externalBranchSplitFnc: External splitting function (default: None)
        minExternalSplit: Minimum external split threshold (default: None)
        externalSplitOptions: Options for external splitting (default: [])
        externalSplitFncNeedsDistance: Whether external function needs distance (default: None)
        assumeSimpleExternalSpecification: Simplify external specifications (default: True)
        pamStage: Whether to perform PAM stage (default: True)
        pamRespectsDendro: Whether PAM respects dendrogram (default: True)
        useMedoids: Whether to use medoids in PAM (default: False)
        maxPamDist: Maximum distance for PAM assignment (default: None)
        respectSmallClusters: Whether to respect small clusters (default: True)
        verbose: Verbosity level (default: 2)
        indent: Indentation level for output (default: 0)
        
    Returns:
        results: Dictionary containing cluster assignments and diagnostic information
    """
    
    
    dendro_height = get_heights(link)
    dendro_merge = get_merges(link)

    if maxPamDist == None:
        maxPamDist = cutHeight
        
    nMerge = len(dendro_height)
    refQuantile = 0.05
    refMerge = np.round(nMerge * refQuantile)

    if refMerge < 1: refMerge = 1

    refHeight = dendro_height[int(refMerge) - 1]

    if cutHeight == None:
        cutHeight = 0.99 * (np.max(dendro_height) - refHeight) + refHeight
        print("..cutHeight not given, setting it to", cutHeight, 
              " ===>  99% of the (truncated) height range in dendro.")
    else:
        if cutHeight > np.max(dendro_height): cutHeight = np.max(dendro_height)
    
    if maxPamDist == None: maxPamDist = cutHeight

    nMergeBelowCut = np.sum(dendro_height <= cutHeight)

    if nMergeBelowCut < minClusterSize:
        print("cutHeight set too low; no merges below the cut.")
        return(np.zeros(nMerge+1))
    
    # fill in this section once understood better
    if externalBranchSplitFnc != None:
        nExternalSplits = len(externalBranchSplitFnc)
        if len(minExternalSplit) < 1:
            raise AttributeError("minExternalBranchSplit must be given.")
        if assumeSimpleExternalSpecification and nExternalSplits == 1:
            pass
    else:
        nExternalSplits = 0


    MxBranches = nMergeBelowCut
    branch_isBasic = np.repeat(True, MxBranches)
    branch_isTopBasic = np.repeat(True, MxBranches)
    branch_failSize = np.repeat(False, MxBranches)
    branch_rootHeight = np.repeat(np.nan, MxBranches)
    branch_size = np.repeat(2, MxBranches)
    branch_nMerge = np.repeat(1, MxBranches)
    branch_nSingletons = np.repeat(2, MxBranches)
    branch_nBasicClusters = np.repeat(0, MxBranches)
    branch_mergedInto = np.repeat(0, MxBranches)
    branch_attachHeight = np.repeat(np.nan, MxBranches)
    #branch_singletons = np.zeros(MxBranches)
    branch_singletons = [np.nan] * MxBranches
    #branch_basicClusters = pd.Series(np.zeros(MxBranches))
    branch_basicClusters = [np.nan] * MxBranches
    #branch_mergingHeights = pd.Series(np.zeros(MxBranches))
    branch_mergingHeights = [np.nan] * MxBranches
    #branch_singletonHeights = pd.Series(np.zeros(MxBranches))
    branch_singletonHeights = [np.nan] * MxBranches


    nBranches = 0

    spyIndex = None
    if os.path.isfile(".dynamicTreeCutSpyFile"):
        spyIndex = pd.read_csv(".dynamicTreeCutSpyFile")
        print("Found 'spy file' with indices of objects to watch for.")
        spyIndex = spyIndex.iloc[:,1].values
    

    defMCS = np.array([0.64, 0.73, 0.82, 0.91, 0.95])
    defMG = (1 - defMCS) * 3 / 4.0

    nSplitDefaults = len(defMCS)

    if type(deepSplit) == bool: deepSplit = int(deepSplit) * (nSplitDefaults - 2)
    deepSplit = deepSplit + 1

    if deepSplit < 1 or deepSplit > nSplitDefaults:
        raise IndexError("Parameter deepSplit (value", deepSplit,
                         ") out of range: allowable range is 0 through",
                         nSplitDefaults - 1)
    
    if maxCoreScatter == None: maxCoreScatter = interpolate(defMCS, deepSplit)
    if minGap == None: minGap = interpolate(defMG, deepSplit)

    if maxAbsCoreScatter == None:
        maxAbsCoreScatter = refHeight + maxCoreScatter * (cutHeight - refHeight)
    if minAbsGap == None:
        minAbsGap = minGap * (cutHeight - refHeight)

    if minSplitHeight == None: minSplitHeight = 0

    if minAbsSplitHeight == None:
        minAbsSplitHeight = refHeight + minSplitHeight * (cutHeight - refHeight)
    
    nPoints = nMerge + 1

    IndMergeToBranch = np.repeat(0, nMerge)

    onBranch = np.repeat(0, nPoints)

    RootBranch = 0

    mergeDiagnostics = dict(smI = np.repeat(np.nan, nMerge), smSize = np.repeat(np.nan, nMerge), 
                            smCrSc = np.repeat(np.nan, nMerge), smGap = np.repeat(np.nan, nMerge), 
                            lgI = np.repeat(np.nan, nMerge), lgSize = np.repeat(np.nan, nMerge), 
                            lgCrSc = np.repeat(np.nan, nMerge), lgGap = np.repeat(np.nan, nMerge),
                            merged = np.repeat(np.nan, nMerge))

    if nExternalSplits > 0:
        #externalMergeDiags = pd.DataFrame(np.repeat(np.nan, nMerge*nExternalSplits).reshape(nMerge, nExternalSplits))
        #externalMergeDiags.columns = paste("externalBranchSplit", nExternalSplits, sep = ".")
        pass

    extender = np.zeros(chunkSize, dtype=int)
    
    for merge in range(nMerge):
        if dendro_height[merge] <= cutHeight:
            # are both merged objects singletons?
            if dendro_merge[merge, 0] < 0 and dendro_merge[merge, 1] < 0:
                nBranches = nBranches + 1
                branch_isBasic[nBranches - 1] = True
                branch_isTopBasic[nBranches - 1] = True
                branch_singletons[nBranches - 1] = np.append(-dendro_merge[merge,], extender)
                branch_basicClusters[nBranches - 1] = extender
                branch_mergingHeights[nBranches - 1] = np.append(np.repeat(dendro_height[merge], 2), extender)
                branch_singletonHeights[nBranches - 1] = np.append(np.repeat(dendro_height[merge], 2), extender)
                IndMergeToBranch[merge] = nBranches
                RootBranch = nBranches
            elif sign(dendro_merge[merge,0]) * sign(dendro_merge[merge,1]) < 0:
                clust = IndMergeToBranch[int(np.max(dendro_merge[merge,])) - 1]
                if clust == 0: raise ValueError("a previous merge has no associated cluster. Sorry!")
                gene = -np.min(dendro_merge[merge,])
                ns = branch_nSingletons[clust - 1] + 1
                nm = branch_nMerge[clust - 1] + 1
                if branch_isBasic[clust - 1]:
                    if ns > len(branch_singletons[clust - 1]):
                        branch_singletons[clust - 1] = np.append(branch_singletons[clust - 1], extender)
                        branch_singletonHeights[clust - 1] = np.append(branch_singletonHeights[clust - 1], extender)
                    branch_singletons[clust - 1][ns - 1] = gene
                    branch_singletonHeights[clust - 1][ns - 1] = dendro_height[merge]
                else:
                    onBranch[int(gene) - 1] = clust
                
                if nm >= len(branch_mergingHeights[clust - 1]):
                    branch_mergingHeights[clust - 1] = np.append(branch_mergingHeights[clust - 1], extender)
                branch_mergingHeights[clust - 1][nm - 1] = dendro_height[merge]
                branch_size[clust - 1] = branch_size[clust - 1] + 1
                branch_nMerge[clust - 1] = nm
                branch_nSingletons[clust - 1] = ns
                IndMergeToBranch[merge] = clust
                RootBranch = clust
            else:
                # attempt to merge two branches:
                clusts = IndMergeToBranch[dendro_merge[merge,] - 1]
                sizes = branch_size[clusts - 1]
                # Note: for 2 elements, rank and order are the same.
                rnk = rankdata(sizes, method = "ordinal")
                small = clusts[rnk[0] - 1]
                large = clusts[rnk[1] - 1]
                sizes = sizes[rnk - 1]
                branch1 = np.nan if np.any(np.isnan(branch_singletons[large - 1])) else branch_singletons[large - 1][np.arange(sizes[1])]
                branch2 = np.nan if np.any(np.isnan(branch_singletons[small - 1])) else branch_singletons[small - 1][np.arange(sizes[0])]
                spyMatch = False
                if spyIndex != None:
                    n1 = len(set(branch1) & set(spyIndex))
                    if n1 / len(branch1) > 0.99 and n1 / len(spyIndex) > 0.99:
                        print("Found spy match for branch 1 on merge", merge)
                        spyMatch = True
                    n2 = len(set(branch2) & set(spyIndex))
                    if n2 / len(branch1) > 0.99 and n2 / len(spyIndex) > 0.99:
                        print("Found spy match for branch 2 on merge", merge)
                        spyMatch = True
            
                if branch_isBasic[small - 1]:
                    coresize = CoreSize(branch_nSingletons[small - 1], minClusterSize)
                    Core = np.array(branch_singletons[small - 1][np.arange(int(coresize))], dtype=int)
                    # SmAveDist = mean(apply(distM[Core, Core], 2, sum)/(coresize-1))
                    SmAveDist = np.mean(np.sum(dist_multi_index(Core - 1, distM), axis=1) / (coresize - 1))     
                else:
                    SmAveDist = 0
            
                if branch_isBasic[large - 1]:
                    coresize = CoreSize(branch_nSingletons[large - 1], minClusterSize)
                    Core = np.array(branch_singletons[large - 1][np.arange(int(coresize))], dtype=int)
                    LgAveDist = np.mean(np.sum(dist_multi_index(Core - 1, distM), axis=1) / (coresize -1 ))
                else:
                    LgAveDist = 0
                
                for key in mergeDiagnostics:
                    if key == "smI":
                        mergeDiagnostics[key][merge] = small
                    elif key == "smSize":
                        mergeDiagnostics[key][merge] = branch_size[small - 1]
                    elif key == "smCrSc":
                        mergeDiagnostics[key][merge] = SmAveDist
                    elif key == "smGap":
                        mergeDiagnostics[key][merge] = dendro_height[merge] - SmAveDist
                    elif key == "lgI":
                        mergeDiagnostics[key][merge] = large
                    elif key == "lgSize":
                        mergeDiagnostics[key][merge] = branch_size[large - 1]
                    elif key == "lgCrSc":
                        mergeDiagnostics[key][merge] = LgAveDist
                    elif key == "lgGap":
                        mergeDiagnostics[key][merge] = dendro_height[merge] - LgAveDist
                    elif key == "merged":
                        mergeDiagnostics[key][merge] = np.nan
                        
            
                # We first check each cluster separately for being too small, too diffuse, or too shallow:
                SmallerScores = [branch_isBasic[small - 1], 
                                 branch_size[small - 1] < minClusterSize,
                                 SmAveDist > maxAbsCoreScatter, 
                                 dendro_height[merge] - SmAveDist < minAbsGap,
                                 dendro_height[merge] < minAbsSplitHeight]
            
                if SmallerScores[0] * np.sum(SmallerScores[1:]) > 0:
                    DoMerge = True
                    SmallerFailSize = ~np.logical_or(SmallerScores[2], SmallerScores[3])  # Smaller fails only due to size
                else:
                    LargerScores = [branch_isBasic[large - 1], 
                                    branch_size[large - 1] < minClusterSize,
                                    LgAveDist > maxAbsCoreScatter, 
                                    dendro_height[merge] - LgAveDist < minAbsGap,
                                    dendro_height[merge] < minAbsSplitHeight]
                    if LargerScores[0] * np.sum(LargerScores[1:]) > 0:
                        # Actually: the large one is the one to be merged
                        DoMerge = True
                        SmallerFailSize = ~np.logical_or(LargerScores[2], LargerScores[3])  # cluster fails only due to size
                        x = small
                        small = large
                        large = x
                        sizes = sizes[::-1]
                    else:
                        DoMerge = False # None of the two satisfies merging criteria
            
                if DoMerge:
                    mergeDiagnostics["merged"][merge] = 1

                if ~DoMerge and nExternalSplits > 0 and branch_isBasic[small - 1] and branch_isBasic[large - 1]:
                    if verbose > 4: print("Entering external split code on merge ", merge)
                    branch1 = branch_singletons[large - 1][np.arange(sizes[1])]
                    branch2 = branch_singletons[small - 1][np.arange(sizes[0])]

                    if verbose > 4 or spyMatch: print("  ..branch lengths: ", sizes[0], ", ", sizes[1])
                    #if (any(is.na(branch1)) || any(branch1==0)) browser();
                    #if (any(is.na(branch2)) || any(branch2==0)) browser();
                
                
                    ##### fix after External Splits is understood better
                    es = 0
                    while es < nExternalSplits and ~DoMerge:
                        es = es + 1
                        args = externalSplitOptions[es - 1]
                        args = [args, list(branch1 = branch1, branch2 = branch2)]
                        #extSplit = do.call(externalBranchSplitFnc[es], args)
                        if spyMatch:
                            print(" .. external criterion ", es, ": ", extSplit)
                        DoMerge = extSplit < minExternalSplit[es - 1]
                        externalMergeDiags[merge, es - 1] = extSplit
                        if DoMerge:
                            mergeDiagnostics_merged[merge] = 2
                        else:
                            mergeDiagnostics_merged[merge] = 0
            
                if DoMerge:
                    # merge the small into the large cluster and close it.
                    branch_failSize[small - 1] = SmallerFailSize
                    branch_mergedInto[small - 1] = large
                    branch_attachHeight[small - 1] = dendro_height[merge]
                    branch_isTopBasic[small - 1] = False
                    nss = branch_nSingletons[small - 1]
                    nsl = branch_nSingletons[large - 1]
                    ns = nss + nsl
                
                    if branch_isBasic[large - 1]: 
                        nExt = np.ceil(  (ns - len(branch_singletons[large - 1])) / chunkSize  )
                    
                        if nExt > 0:
                            if verbose > 5:
                                print("Extending singletons for branch", large, "by", nExt, " extenders.")
                        
                            branch_singletons[large - 1] = np.append(branch_singletons[large - 1], np.repeat(extender, nExt))
                            branch_singletonHeights[large - 1] = np.append(branch_singletonHeights[large - 1], np.repeat(extender, nExt))
                    
                        branch_singletons[large - 1][np.arange(nsl,ns)] = branch_singletons[small - 1][np.arange(nss)]
                        branch_singletonHeights[large - 1][np.arange(nsl,ns)] = branch_singletonHeights[small - 1][np.arange(nss)]
                        branch_nSingletons[large - 1] = ns
                    
                    else:
                        if ~branch_isBasic[small - 1]:
                            raise ValueError("merging two composite clusters. Sorry!")
                    
                        onBranch[ branch_singletons[small - 1][branch_singletons[small - 1] != 0] - 1 ] = large
                
                    nm = branch_nMerge[large - 1] + 1
                
                    if nm > len(branch_mergingHeights[large - 1]):
                        branch_mergingHeights[large - 1] = np.append(branch_mergingHeights[large - 1], extender)
                
                    branch_mergingHeights[large - 1][nm - 1] = dendro_height[merge]
                    branch_nMerge[large - 1] = nm
                    branch_size[large - 1] = branch_size[small - 1] + branch_size[large - 1]
                    IndMergeToBranch[merge] = large
                    RootBranch = large
                else:
                    # start or continue a composite cluster.

                    # If large is basic and small is not basic, switch them.
                    if branch_isBasic[large - 1] and ~branch_isBasic[small - 1]:
                        x = large
                        large = small
                        small = x
                        sizes = sizes[::-1]
                
                    # Note: if pamRespectsDendro, need to start a new composite cluster every time two branches merge,
                    # otherwise will not have the necessary information.
                    # Otherwise, if the large cluster is already composite, I can simply merge both clusters into 
                    # one of the non-composite clusters.

                    if branch_isBasic[large - 1] or pamStage and pamRespectsDendro:
                        nBranches = nBranches + 1
                        branch_attachHeight[[large - 1, small - 1]] = dendro_height[merge]
                        branch_mergedInto[[large - 1, small - 1]] = nBranches
                        if branch_isBasic[small - 1]:
                            addBasicClusters = small # add basic clusters
                        else:
                            addBasicClusters = branch_basicClusters[small - 1]
                        if branch_isBasic[large - 1]:
                            addBasicClusters = np.append(addBasicClusters, large)
                        else:
                            addBasicClusters = np.append(addBasicClusters, branch_basicClusters[large - 1])
                        # print(paste("  Starting a composite cluster with number", nBranches));
                        branch_isBasic[nBranches - 1] = False
                        branch_isTopBasic[nBranches - 1] = False
                        branch_basicClusters[nBranches - 1] = addBasicClusters
                        branch_mergingHeights[nBranches - 1] = np.append(np.repeat(dendro_height[merge], 2), extender)
                        branch_nMerge[nBranches - 1] = 2
                        branch_size[nBranches - 1] = np.sum(sizes)
                        branch_nBasicClusters[nBranches - 1] = len(addBasicClusters)
                        IndMergeToBranch[merge] = nBranches
                        RootBranch = nBranches
                    else:
                        # Add small branch to the large one 
                        addBasicClusters = small if branch_isBasic[small - 1] else branch_basicClusters[small - 1]
                        nbl = branch_nBasicClusters[large - 1]
                        #small might be an int
                        try:
                            nb = branch_nBasicClusters[large - 1] + len(addBasicClusters)
                        except TypeError:
                            nb = branch_nBasicClusters[large - 1] + 1
                                                
                        if nb > len(branch_basicClusters[large - 1]):
                            nExt = np.ceil(  ( nb - len(branch_basicClusters[large - 1])) / chunkSize)
                            branch_basicClusters[large - 1] = np.append(branch_basicClusters[large - 1], np.repeat(extender, nExt))
                    
                        branch_basicClusters[large - 1][np.arange(nbl,nb)] = addBasicClusters
                        branch_nBasicClusters[large - 1] = nb
                        branch_size[large - 1] = branch_size[large - 1] + branch_size[small - 1]
                        nm = branch_nMerge[large - 1] + 1
                    
                        if nm > len(branch_mergingHeights[large - 1]):
                            branch_mergingHeights[large - 1] = np.append(branch_mergingHeights[large - 1], extender) 
                    
                        branch_mergingHeights[large - 1][nm - 1] = dendro_height[merge]
                        branch_nMerge[large - 1] = nm
                        branch_attachHeight[small - 1] = dendro_height[merge]
                        branch_mergedInto[small - 1] = large
                        IndMergeToBranch[merge] = large
                        RootBranch = large
        
    if verbose > 2: print("..Going through detected branches and marking clusters..")
            
    isCluster = np.repeat(False, nBranches)
    SmallLabels = np.repeat(0, nPoints)

    for clust in range(nBranches):
    
        if np.isnan(branch_attachHeight[clust]): branch_attachHeight[clust] = cutHeight
        if branch_isTopBasic[clust]:
            coresize = CoreSize(branch_nSingletons[clust], minClusterSize)
            Core = branch_singletons[clust][np.arange(coresize)]
            CoreScatter = np.mean(np.sum(dist_multi_index(Core - 1, distM), axis=1) / (coresize - 1))
            isCluster[clust] = np.logical_and(np.logical_and(branch_isTopBasic[clust],
                                                             branch_size[clust] >= minClusterSize),
                                              np.logical_and(CoreScatter < maxAbsCoreScatter,
                                                             branch_attachHeight[clust] - CoreScatter > minAbsGap))
        else:
            CoreScatter = 0
        
        if branch_failSize[clust]: SmallLabels[branch_singletons[clust][branch_singletons[clust] != 0] - 1] = clust + 1
    
    if not respectSmallClusters: SmallLabels = np.repeat(0, nPoints)

    if verbose > 2: print(spaces, "..Assigning Tree Cut stage labels..")

    Colors = np.repeat(0, nPoints)
    coreLabels = np.repeat(0, nPoints)
    clusterBranches = np.arange(nBranches)[isCluster]
    branchLabels = np.repeat(0, nBranches)
    color = 0
                    
    for clust in clusterBranches:
        color = color + 1
        Colors[branch_singletons[clust][branch_singletons[clust] != 0] - 1] = color
        SmallLabels[branch_singletons[clust][branch_singletons[clust] != 0] - 1] = 0
        coresize = CoreSize(branch_nSingletons[clust], minClusterSize)
        Core = branch_singletons[clust][np.arange(coresize)]
        coreLabels[Core - 1] = color
        branchLabels[clust] = color

    Labeled = np.arange(nPoints)[Colors != 0]
    Unlabeled = np.arange(nPoints)[Colors == 0]
    nUnlabeled = len(Unlabeled)
    UnlabeledExist = nUnlabeled > 0

    if len(Labeled) > 0:
        LabelFac = factor(Colors[Labeled])
        nProperLabels = nlevels(LabelFac)
    else:
        nProperLabels = 0


    if pamStage and UnlabeledExist and nProperLabels > 0:
        if verbose > 2: print(spaces, "..Assigning PAM stage labels..")
        nPAMed = 0
        # Assign some of the grey genes to the nearest module. Define nearest as the distance to the medoid,
        # that is the point in the cluster that has the lowest average distance to all other points in the
        # cluster. First get the medoids.
        if useMedoids:
            Medoids = np.repeat(0, nProperLabels)
            ClusterRadii = np.repeat(0.0, nProperLabels)
            for cluster in range(1, nProperLabels + 1):
                InCluster = np.arange(1,nPoints+1)[Colors == cluster]
                DistInCluster = dist_multi_index(InCluster - 1, distM)
                #DistInCluster = distM[InCluster, InCluster]
                DistSums = np.sum(DistInCluster, axis=1)
                Medoids[cluster - 1] = InCluster[np.argmin(DistSums)]
                ClusterRadii[cluster - 1] = np.max(DistInCluster[:, np.argmin(DistSums)])
            # If small clusters are to be respected, assign those first based on medoid-medoid distances.
            if respectSmallClusters:
                FSmallLabels = factor(SmallLabels)
                SmallLabLevs = levels(FSmallLabels)
                nSmallClusters = nlevels(FSmallLabels) - (SmallLabLevs[0] == 0)
                if nSmallClusters > 0 :
                    for sclust in SmallLabLevs[SmallLabLevs != 0]:
                        InCluster = np.arange(nPoints)[SmallLabels == sclust]
                        if pamRespectsDendro:
                            onBr = np.unique(onBranch[InCluster])
                            if len(onBr) > 1:
                                raise ValueError("Internal error: objects in a small cluster are marked to belong",
                                                 "\nto several large branches:")
                            if onBr > 0:
                                basicOnBranch = branch_basicClusters[onBr[0] - 1]
                                labelsOnBranch = branchLabels[basicOnBranch - 1]
                            else:
                                labelsOnBranch = None
                        else:
                            labelsOnBranch = np.arange(1, nProperLabels + 1)
                        # printFlush(paste("SmallCluster", sclust, "has", length(InCluster), "elements."));
                        DistInCluster = dist_multi_index(InCluster, distM)
                        #DistInCluster = distM[InCluster, InCluster]
                        if len(labelsOnBranch) > 0:
                            if len(InCluster) > 1:
                                DistSums = apply(np.sum, DistInCluster, 1)
                                smed = InCluster[np.argmin(DistSums)]
                                DistToMeds = get_rows(Medoids[labelsOnBranch - 1][Medoids[labelsOnBranch - 1] != 0] - 1, distM)[:, smed]
                                closest = np.argmin(DistToMeds)
                                DistToClosest = DistToMeds[closest]
                                closestLabel = labelsOnBranch[closest]
                                if DistToClosest < ClusterRadii[closestLabel - 1] or DistToClosest <  maxPamDist:
                                    Colors[InCluster] = closestLabel
                                    nPAMed = nPAMed + len(InCluster)
                                else: Colors[InCluster] = -1  # This prevents individual points from being assigned later 
                        else:
                            Colors[InCluster] = -1
         
                # Assign leftover unlabeled objects to clusters with nearest medoids
                Unlabeled = np.arange(nPoints)[Colors == 0]
                if len(Unlabeled > 0):
                    for obj in Unlabeled:
                        if pamRespectsDendro:
                            onBr = onBranch[obj]
                            if onBr > 0:
                                basicOnBranch = branch_basicClusters[onBr - 1]
                                labelsOnBranch = branchLabels[basicOnBranch - 1]
                            else:
                                labelsOnBranch = None
                        else:
                            labelsOnBranch = np.arange(nProperLabels)
                        if labelsOnBranch != None:
                            UnassdToMedoidDist = get_rows(Medoids[labelsOnBranch - 1] - 1, distM)[:,obj]
                            #UnassdToMedoidDist = distM[Medoids[labelsOnBranch], obj]
                            nearest= np.argmin(UnassdToMedoidDist)
                            NearestCenterDist = UnassdToMedoidDist[nearest]
                            nearestMed = labelsOnBranch[nearest]
                            if NearestCenterDist < ClusterRadii[nearestMed - 1] or NearestCenterDist < maxPamDist:
                                Colors[obj] = nearestMed
                                nPAMed = nPAMed + 1

                UnlabeledExist = np.sum(Colors == 0) > 0
        else: # Instead of medoids, use average distances
        # This is the default method, so I will try to tune it for speed a bit.
            ClusterDiam = np.repeat(0, nProperLabels)
            for cluster in range(nProperLabels):
                InCluster = np.arange(nPoints)[Colors == cluster]
                nInCluster = len(InCluster)
                DistInCluster = dist_multi_index(InCluster, distM)
                #DistInCluster = distM[InCluster, InCluster]
                if nInCluster > 1:
                    AveDistInClust = np.sum(DistInCluster, axis=1) / (nInCluster - 1)
                    ClusterDiam[cluster] = np.max(AveDistInClust)
                else:
                    ClusterDiam[cluster] = 0

            # If small clusters are respected, assign them first based on average cluster-cluster distances.
            ColorsX = Colors.copy()
            if respectSmallClusters:
                FSmallLabels = factor(SmallLabels) #### think about
                SmallLabLevs = levels(FSmallLabels) ##### think about
                nSmallClusters = nlevels(FSmallLabels) - (SmallLabLevs[0] == 0)
                if nSmallClusters > 0:
                    if pamRespectsDendro:
                        for sclust in SmallLabLevs[SmallLabLevs != 0]:
                            InCluster = np.arange(nPoints)[SmallLabels == sclust]
                            onBr = np.unique(onBranch[InCluster])

                            if len(onBr) > 1:
                                raise ValueError("objects in a small cluster are marked to belong",
                                                 "\nto several large branches:")
                            if onBr > 0:
                                basicOnBranch = branch_basicClusters[onBr[0] - 1]
                                labelsOnBranch = branchLabels[basicOnBranch - 1]
                                useObjects = np.isin(ColorsX, np.unique(labelsOnBranch))
                                DistSClustClust = get_rows(InCluster, distM)[:,useObjects]
                                #DistSClustClust = distM[InCluster, useObjects]
                                MeanDist = np.mean(DistSClustClust, axis=0)
                                useColorsFac = factor(ColorsX[useObjects]) ### think about
                                MeanMeanDist = tapply(MeanDist, useColorsFac, np.mean) ## think about
                                nearest = np.argmin(MeanMeanDist)
                                NearestDist = MeanMeanDist[nearest]
                                nearestLabel = levels(useColorsFac)[nearest] ## think about
                                if NearestDist < ClusterDiam[nearestLabel - 1] or NearestDist <  maxPamDist:
                                    Colors[InCluster] = nearestLabel
                                    nPAMed = nPAMed + len(InCluster)
                                else:
                                    Colors[InCluster] = -1  # This prevents individual points from being assigned later
 
                    else:
                        labelsOnBranch = np.arange(nProperLabels)
                        useObjects = np.arange(nPoints)[ColorsX != 0]
                        for sclust in SmallLabLevs[SmallLabLevs != 0]:
                            InCluster = np.arange(nPoints)[SmallLabels == sclust]
                            DistSClustClust = get_rows(InCluster, distM)[:,useObjects]
                            #DistSClustClust = distM[InCluster, useObjects]
                            MeanDist = np.mean(DistSClustClust, axis=0)
                            useColorsFac = factor(ColorsX[useObjects]) ### think about
                            MeanMeanDist = tapply(MeanDist, useColorsFac, np.mean) ### think about
                            nearest = np.argmin(MeanMeanDist)
                            NearestDist = MeanMeanDist[nearest]
                            nearestLabel = levels(useColorsFac)[nearest] ## think about
                            if NearestDist < ClusterDiam[nearestLabel - 1] or NearestDist <  maxPamDist:
                                Colors[InCluster] = nearestLabel
                                nPAMed = nPAMed + len(InCluster)
                            else:
                                Colors[InCluster] = -1  # This prevents individual points from being assigned later

            # Assign leftover unlabeled objects to clusters with nearest medoids
            Unlabeled = np.arange(nPoints)[Colors == 0]
            #ColorsX = Colors;
            if len(Unlabeled) > 0:
                if pamRespectsDendro:
                    unlabOnBranch = Unlabeled[onBranch[Unlabeled] > 0]
                    for obj in unlabOnBranch:
                        onBr = onBranch[obj]
                        basicOnBranch = branch_basicClusters[onBr - 1]
                        labelsOnBranch = branchLabels[basicOnBranch - 1]
                        useObjects = np.isin(ColorsX, np.unique(labelsOnBranch))
                        useColorsFac = factor(ColorsX[useObjects]) ### think about
                        #UnassdToClustDist = tapply(distM[useObjects, obj], useColorsFac, mean) ### think about
                        UnassdToClustDist = tapply(get_rows(useObjects, distM)[:,obj], useColorsFac, np.mean) ### think about
                        nearest = np.argmin(UnassdToClustDist)
                        NearestClusterDist = UnassdToClustDist[nearest]
                        nearestLabel = levels(useColorsFac)[nearest] ### think about
                        if NearestClusterDist < ClusterDiam[nearestLabel - 1] or NearestClusterDist < maxPamDist:
                            Colors[obj] = nearestLabel
                            nPAMed = nPAMed + 1

                else:
                    useObjects = np.arange(nPoints)[ColorsX != 0]
                    useColorsFac = factor(ColorsX[useObjects]) ## think about
                    nUseColors = nlevels(useColorsFac) ### think about
                    UnassdToClustDist = tapply_df(get_rows(useObjects, distM)[:,Unlabeled], useColorsFac, np.mean, 1)
                    #UnassdToClustDist = df_apply.apply(distM[useObjects, Unlabeled], 1, tapply, useColorsFac, mean) ### think about
                    # Fix dimensions for the case when there's only one cluster
                    #dim(UnassdToClustDist) = np.append(nUseColors, len(Unlabeled)) ### think about
                    nearest = apply(np.argmin, UnassdToClustDist, 1)
                    nearestDist = apply(np.min, UnassdToClustDist, 1)
                    nearestLabel = levels(useColorsFac)[nearest - 1] ### think about
                    assign = np.logical_or(nearestDist < ClusterDiam[nearestLabel - 1], nearestDist < maxPamDist)
                    Colors[Unlabeled[assign]] = nearestLabel[assign]
                    nPAMed = nPAMed + np.sum(assign)

        if verbose > 2: print("....assigned", nPAMed, "objects to existing clusters.")

    
    # Relabel labels such that 1 corresponds to the largest cluster etc.
    Colors[Colors < 0] = 0
    UnlabeledExist = np.sum(Colors == 0) > 0
    NumLabs = Colors + 1
    Sizes = table(NumLabs) ### think about
    Sizes.index=np.arange(len(Sizes))

    if UnlabeledExist:
        if len(Sizes) > 1:
            SizeRank = np.append(1, rankdata(-Sizes[1:len(Sizes)], method="ordinal")+1)
        else:
            SizeRank = 1
        OrdNumLabs = SizeRank[NumLabs - 1]
    else:
        SizeRank = rankdata(-Sizes[np.arange(len(Sizes))], method="ordinal")
        OrdNumLabs = SizeRank[NumLabs - 2]
    ordCoreLabels = OrdNumLabs - UnlabeledExist
    ordCoreLabels[coreLabels == 0] = 0

    if verbose > 0: print( "..done.")

    results = dict(labels = OrdNumLabs-UnlabeledExist,
                   cores = ordCoreLabels,
                   smallLabels = SmallLabels,
                   onBranch = onBranch,
                   mergeDiagnostics = mergeDiagnostics if nExternalSplits==0 else pd.DataFrame({'x':mergeDiagnostics, 'y':externalMergeDiags}),
                   mergeCriteria = dict(maxCoreScatter = maxCoreScatter, minGap = minGap, 
                                        maxAbsCoreScatter = maxAbsCoreScatter, minAbsGap = minAbsGap, 
                                        minExternalSplit = minExternalSplit),
                   branches  = dict(nBranches = nBranches, # Branches = Branches, 
                                    IndMergeToBranch = IndMergeToBranch,
                                    RootBranch = RootBranch, isCluster = isCluster, 
                                    nPoints = nMerge+1))

    return(results)












import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import to_tree


def sign(value):
    #python version of R's sign
    
    if value > 0:
        return(1)
    elif value < 0:
        return(-1)
    else:
        return(0)

def paste(string, n, sep=""):
    #python version of R's paste
    
    results = []
    for i in range(n):
        results.append(string + sep + str(i))
    
    return(results)


def get_heights(Z):
    #python verison of R's dendro$height
    #height = np.zeros(len(dendro["dcoord"]))
    
    #for i, d in enumerate(dendro["dcoord"]):
        #height[i] = d[1]
    
    clusternode = to_tree(Z, True)
    #height = np.array([c.dist for c in clusternode[1]])
    height = np.array([c.dist for c in clusternode[1] if c.is_leaf() != True])
        
    #height.sort()
    
    return(height)
    

def get_merges(z):
    #python version of R's dendro$merge
    n = z.shape[0]
    merges = np.zeros((z.shape[0], 2), dtype=int)
        
    for i in range(z.shape[0]):
        for j in range(2):
            if z[i][j] <= n:
                merges[i][j] = -(z[i][j] + 1)
            else:
                cluster = z[i][j] - n
                merges[i][j] = cluster
                
    return(merges)
    
    
def factor(vector):
    return(vector)


def nlevels(vector):
    #python version of R's nlevels
    return(len(np.unique(vector)))


def levels(vector):
    #python version of R's levels
    return(np.unique(vector))


def tapply(vector, index, function): #can add **args, **kwargs
    #python version of R's tapply
    
    factors = np.unique(index)
    
    #results = pd.Series(np.repeat(np.nan, len(factors)))
    results = np.repeat(np.nan, len(factors))
    #results.index = factors
    
    for i, k in enumerate(factors):
        subset = vector[index == k]
        #results.iloc[i] = function(subset)
        results[i] = function(subset)
    
    return(results)


def tapply_df(df, index, function, axis=0): #can add **args, **kwargs
    #python version of R's tapply
    
    factors = np.unique(index)
    
    if axis == 1:
        #results = pd.DataFrame(np.zeros((len(factors), df.shape[1])))
        results = np.zeros((len(factors), df.shape[1]))
    else:
        #results = pd.DataFrame(np.zeros((df.shape[0], len(factors))))
        results = np.zeros((df.shape[0], len(factors)))
    
    #results.index = factors
    
    if axis == 1:
        for j in range(df.shape[1]):
            for i, k in enumerate(factors):
                subset = df[index == k, j]
                #results.iloc[i, j] = function(subset)
                results[i, j] = function(subset)
    else:
        for i in range(df.shape[0]):
            for j, k in enumerate(factors):
                subset = df[i, index == k]
                #results.iloc[i, j] = function(subset)
                results[i, j] = function(subset)
    
    return(results)


def table(vector):
    
    factors = np.unique(vector)
    results = pd.Series(np.zeros(len(factors), dtype=int))
    results.index = factors
    
    for i, k in enumerate(factors):
        results.iloc[i] = np.sum(vector == k)
        
    return(results)









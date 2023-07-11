import pandas as pd
import numpy as np
import anndata as ad

def atac_concat_get_index(adata1,adata2):
    import gc
    
    # 数据预处理
    adata1.var.loc[:,'ATAC'] = adata1.var.index
    adata1.var = pd.DataFrame((x.split('_') for x in adata1.var['ATAC']) , index = adata1.var.index , columns = ['chr','chromStart','chromEnd'])
    adata2.var.loc[:,'ATAC'] = adata2.var.index
    adata2.var = pd.DataFrame((x.split('_') for x in adata2.var['ATAC']) , index = adata2.var.index , columns = ['chr','chromStart','chromEnd'])

    # 判断是否出现 不存在某条染色体的情况，如果存在，我们希望染色体较少的一方为adata1（因为后续的分析都是针对adata1为染色体较少的情况进行分析的）
    # 如果出现adata1的染色体比adata2的染色体多，则调换位置
    if len(set(adata1.var.chr))>len(set(adata2.var.chr)):
        adata_chr = adata1
        adata1 = adata2
        adata2 = adata_chr
        Exchange = True #Exchange用于判断调换位置的特殊情况
    else:
        Exchange = False
 

    # 循环函数确定和拆分
    atac_var = adata1.var
    Promoter = adata2.var
    Promoter.loc[:,'chr']=Promoter.loc[:,'chr']+'_'
    Promoter.loc[:,'index_1'] = Promoter.index
    atac_var.loc[:,'chr']=atac_var.loc[:,'chr']+'_'
    atac_var.loc[:,'index_1'] = atac_var.index
    chr	= set(Promoter.chr)

    # 比对函数开始
    ATAC = {}
    PROMOTER = {}
    for x in chr:
        # 染色体拆分
        test = pd.DataFrame(columns=['Length'])
        ATAC[x] = test
        ATAC[x]=atac_var[atac_var.index.str.contains(x)]
        PROMOTER[x] = test
        test = pd.DataFrame(columns=['Length'])
        PROMOTER[x]=Promoter.loc[Promoter['chr'] == x]

        # 输出循环染色体
        print('start %s' % (x))
    
        # 特殊情况处理-判断该条染色体是否为空值,并进行赋值
        if len(ATAC[x].index)==0:
            ATAC[x] = PROMOTER[x]
            ATAC_INDEX = ATAC[x].columns.to_list() + ['overleaf_1','overleaf_2','chromStart_New','chromEnd_New','index_2']
            ATAC[x] = ATAC[x].reindex(columns=ATAC_INDEX, fill_value='retain')
            ATAC[x].loc[:,'index_2'] = ATAC[x].loc[:,'index_1']
            continue
        

        # index赋值
        ATAC_INDEX = ATAC[x].columns.to_list() + ['overleaf_1','overleaf_2','chromStart_New','chromEnd_New','index_2']
        ATAC[x] = ATAC[x].reindex(columns=ATAC_INDEX, fill_value='')
        
        # 函数主循环体
        for i in range(0,len(PROMOTER[x].chromEnd)):
            left = 0
            right = len(ATAC[x].chromEnd)-1
            target_Start = int(PROMOTER[x].chromStart.iloc[i])
            target_End = int(PROMOTER[x].chromEnd.iloc[i])


            while(left<=right):
                # 中位数表示 #
                midIndex = round(int((left+right)/2))
                midValue_Start = int(ATAC[x].chromStart.iloc[midIndex])
                midValue_End = int(ATAC[x].chromEnd.iloc[midIndex])
                # 三次condition判断 #

                if (target_Start > midValue_End) :
                    left = midIndex+1

                elif (target_End < midValue_Start):
                    right = midIndex-1
                else:
                    Length = target_End - target_Start
                    ATAC[x].loc[:,'overleaf_1'].iloc[midIndex] = (target_Start - midValue_End)/Length
                    ATAC[x].loc[:,'overleaf_2'].iloc[midIndex] = (midValue_Start - target_End)/Length
                    list = [midValue_Start,midValue_End,target_Start,target_End]
                    ATAC[x].loc[:,'chromStart_New'].iloc[midIndex] = min(list)
                    ATAC[x].loc[:,'chromEnd_New'].iloc[midIndex] = max(list)
                    ATAC[x].loc[:,'index_2'].iloc[midIndex] = PROMOTER[x].index[i]
                    break
 
    # 拼接dataframe
    ATAC_Promoter = pd.DataFrame()
    for i in chr:
        ATAC_Promoter = pd.concat([ATAC_Promoter,ATAC[i],],axis=0)
        
    # 删除未比对上序列
    ATAC_Promoter = ATAC_Promoter[ATAC_Promoter['overleaf_1']!='']

    # 定义新序列拼接函数
    def indeX(series):
        chr = series["chr"]
        chromStart_New = series["chromStart_New"]
        chromEnd_New = series["chromEnd_New"]
        indeX = chr + str(chromStart_New) +'_' +str(chromEnd_New)
        return indeX

    # 拼接新序列
    ATAC_Promoter.loc[:,'index_new'] = ''
    ATAC_Promoter.loc[:,'index_new'] = ATAC_Promoter.apply(indeX,axis=1)

    # 提取新序列
    pair_index = pd.DataFrame()
    pair_index.loc[:,'overleaf_1'] = ATAC_Promoter.loc[:,'overleaf_1']
    pair_index.loc[:,'overleaf_2'] = ATAC_Promoter.loc[:,'overleaf_2']
    pair_index.loc[:,'index_new'] = ATAC_Promoter.loc[:,'index_new']
    # 若出现adata1调换adata2的特殊情况，则在这里把adata1和adata2再调换回来
    if Exchange==True:
        pair_index.loc[:,'index_1'] = ATAC_Promoter.loc[:,'index_2']
        pair_index.loc[:,'index_2'] = ATAC_Promoter.loc[:,'index_1']
    else:
        pair_index.loc[:,'index_1'] = ATAC_Promoter.loc[:,'index_1']
        pair_index.loc[:,'index_2'] = ATAC_Promoter.loc[:,'index_2']
    # 特殊情况处理-空值染色体
    pair_index.loc[pair_index['overleaf_1']=='retain','index_new'] = pair_index.loc[pair_index['overleaf_1']=='retain','index_1']
    gc.collect()
    return pair_index


def atac_concat_inner(adata1,adata2,pair_index):
    import gc
    
    # 设置基本参数
    adata1.var.loc[:,'ATAC'] = adata1.var.index
    df1 = pd.DataFrame((x.split('_') for x in adata1.var['ATAC']) , index = adata1.var.index , columns = ['chr','chromStart','chromEnd'])
    adata1.var = df1
    adata2.var.loc[:,'ATAC'] = adata2.var.index
    df2 = pd.DataFrame((x.split('_') for x in adata2.var['ATAC']) , index = adata2.var.index , columns = ['chr','chromStart','chromEnd'])
    adata2.var = df2

    # 判断是否存在需要调换位置的情况
    if len(set(adata1.var.chr))>len(set(adata2.var.chr)): 
        Exchange = True #Exchange用于判断调换位置的特殊情况
    else:
        Exchange = False
    
    # 通过overleaf_1的赋值来判断是否存在缺失染色体的情况
    if Exchange==False:
        atac1_new = adata1[:,pair_index[pair_index['overleaf_1']!='retain'].index_1]
        atac1_new.var = pair_index[pair_index['overleaf_1']!='retain']
        atac1_new.var.index = atac1_new.var.index_new
        atac2_new = adata2[:,pair_index.index_2]
        atac2_new.var = pair_index
        atac2_new.var.index = atac2_new.var.index_new

    if Exchange==True:
        atac1_new = adata1[:,pair_index.index_1]
        atac1_new.var = pair_index
        atac1_new.var.index = atac1_new.var.index_new
        atac2_new = adata2[:,pair_index[pair_index['overleaf_1']!='retain'].index_2]
        atac2_new.var = pair_index[pair_index['overleaf_1']!='retain']
        atac2_new.var.index = atac2_new.var.index_new
    
    # 配对
    adata_pair = ad.concat([atac1_new,atac2_new],axis=0,join='outer',fill_value=0)
    adata_pair.obs_names_make_unique()
    adata_pair
    gc.collect()
    return adata_pair

def atac_concat_outer(adata1,adata2,pair_index):
    import gc
    
        # 提取关键参数
    adata1.var.loc[:,'ATAC'] = adata1.var.index
    adata1.var = pd.DataFrame((x.split('_') for x in adata1.var['ATAC']) , index = adata1.var.index , columns = ['chr','chromStart','chromEnd'])
    adata2.var.loc[:,'ATAC'] = adata2.var.index
    adata2.var = pd.DataFrame((x.split('_') for x in adata2.var['ATAC']) , index = adata2.var.index , columns = ['chr','chromStart','chromEnd'])

        # 判断是否存在需要调换位置的情况
    if len(set(adata1.var.chr))>len(set(adata2.var.chr)): 
        Exchange = True #Exchange用于判断调换位置的特殊情况
    else:
        Exchange = False
    
        # 通过overleaf_1的赋值来判断是否存在缺失染色体的情况
    if Exchange==False:
        atac1_new = adata1[:,pair_index[pair_index['overleaf_1']!='retain'].index_1]
        atac1_new.var = pair_index[pair_index['overleaf_1']!='retain']
        atac1_new.var.index = atac1_new.var.index_new
        atac2_new = adata2[:,pair_index.index_2]
        atac2_new.var = pair_index
        atac2_new.var.index = atac2_new.var.index_new

    if Exchange==True:
        atac1_new = adata1[:,pair_index.index_1]
        atac1_new.var = pair_index
        atac1_new.var.index = atac1_new.var.index_new
        atac2_new = adata2[:,pair_index[pair_index['overleaf_1']!='retain'].index_2]
        atac2_new.var = pair_index[pair_index['overleaf_1']!='retain']
        atac2_new.var.index = atac2_new.var.index_new

    # 计算前置数据
    index_list = atac1_new.var
    index_list.index = index_list.index_1
    df_month = pd.DataFrame(columns=['name'])
    df_month.name = adata1.var.index.values
    df_month.index = df_month.name.values
    df_month.drop(['name'],axis=1,inplace=True)
    var_1 = pd.concat([index_list,df_month],join='outer',axis=1).copy()
    var_1 = var_1.fillna('*')
    var_1['name']=var_1.index
    # 替换index
    var_1.loc[var_1['index_1']!='*','name'] = var_1[var_1['index_1']!='*'].index_new.values
    var_1.index = var_1.name
    adata1.var =var_1
    # 输出结果
    adata1.var_name = var_1.index
    
     # 同上
    index_list = atac2_new.var
    index_list.index = index_list.index_2
    df_month = pd.DataFrame(columns=['name'])
    df_month.name = adata2.var.index.values
    df_month.index = df_month.name.values
    df_month.drop(['name'],axis=1,inplace=True)
    var_2 = pd.concat([index_list,df_month],join='outer',axis=1).copy()
    var_2 = var_2.fillna('*')
    var_2['name']=var_2.index
    var_2.loc[var_2['index_1']!='*','name'] = var_2[var_2['index_1']!='*'].index_new.values
    var_2.index = var_2.name
    adata2.var =var_2
    adata2.var_name = var_2.index
    adata1.var_names_make_unique()
    adata2.var_names_make_unique()
    adata_pair = ad.concat([adata1,adata2],axis=0,join='outer',fill_value=0)
    adata_pair.obs_names_make_unique()
    adata_pair=adata_pair[:,[i for i in adata_pair.var_name if '-' not in i]]


    gc.collect()
    return adata_pair
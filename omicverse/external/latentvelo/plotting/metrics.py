import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def integration_metrics(models, model_names, x = 'score', hue='Model', edgecolor='#474747ff', height=5, aspect=1.):
    
    batch_correction = pd.DataFrame(pd.concat([pd.DataFrame(m).T for m in models], axis=0))
    model_names_ = []
    for mn in model_names:
        model_names_ = model_names_ + pd.DataFrame(models[0]).shape[1]*[mn]
    batch_correction['Model'] = model_names_ 
    batch_correction['score'] = batch_correction[0]
    batch_correction = batch_correction.drop(0, 1)
    batch_correction['Metric'] = batch_correction.index.values
    
    ax = sns.catplot(
        data=batch_correction, 
        x='Metric', y=x, kind='bar', hue = hue,
        alpha=1, orient='v', height=height, aspect=aspect, palette = 'Set1', edgecolor=edgecolor
    )
    
    ax.set(ylim=(0, 1))
    
    return ax

def box_plot(df, x, y, hue, ax, orient='v', palette ="Set1"):

    g = sns.boxplot(
        data=df, 
        y=y, x=x, hue=hue,ax=ax,
        orient=orient, palette = palette, fliersize=0, showmeans=True,notch=True,
        meanprops={"marker":"o",
                   "markerfacecolor":"white", 
                   "markeredgecolor":"black",
                   "markersize":"8"}
    )
    return g
    
def integration_velocity_metrics(dfs, model_names, basis, figsize=(10,5), palette = "Set1", orient = 'h'):

    
    df_list = []
    for i in range(len(dfs)):
        df_1 = dfs[i][['velocity_cosine_'+basis[i]]]
        df_1['Metric'] = 'Velocity cosine'
        df_1['Score'] = dfs[i]['velocity_cosine_'+basis[i]]

        df_2 = dfs[i][['CBDir_'+basis[i]]]
        df_2['Metric'] = 'CBDir'
        df_2['Score'] = dfs[i]['CBDir_'+basis[i]]

        df_3 = dfs[i][['nn_velo_'+basis[i]]]
        df_3['Metric'] = 'NN velocity cosine'
        df_3['Score'] = dfs[i]['nn_velo_'+basis[i]]

        df_4 = dfs[i][['ICCoh_'+basis[i]]]
        df_4['Metric'] = 'ICCoh'
        df_4['Score'] = dfs[i]['ICCoh_'+basis[i]]

        df_ = pd.concat((df_1, df_2, df_3, df_4), axis=0)
        df_['Model'] = model_names[i]
        df_list.append(df_)

    df = pd.concat(df_list, axis=0)
    
    if orient == 'h':
        fig,ax=plt.subplots(4,1,sharex=False, figsize=figsize)
        ax=ax.flatten()
        box_plot(df.loc[df['Metric'].isin(['Velocity cosine'])], y='Metric', 
                 x='Score', orient=orient, hue='Model',ax=ax[0], palette = palette)
        box_plot(df.loc[df['Metric'].isin(['CBDir'])], y='Metric', 
                 x='Score', orient=orient, hue='Model',ax=ax[1], palette = palette)
        box_plot(df.loc[df['Metric'].isin(['NN velocity cosine'])], y='Metric', 
                 x='Score', orient=orient, hue='Model',ax=ax[2], palette = palette)
        box_plot(df.loc[df['Metric'].isin(['ICCoh'])], y='Metric', 
                 x='Score', orient=orient, hue='Model',ax=ax[3], palette = palette)
        
        for i in range(ax.shape[0]):
            ax[i].legend([],[], frameon=False)
            ax[i].spines.right.set_visible(False)
            ax[i].spines.top.set_visible(False)
            ax[i].set_ylabel('')
            ax[i].set_xlabel('')
            plt.setp(ax[i].get_yticklabels(), fontsize=14)
            plt.setp(ax[i].get_xticklabels(), fontsize=13)
        ax[0].set_xlim(-1.025, 1.025)
        ax[1].set_xlim(-0.025, 1.025)
        ax[2].set_xlim(-1.025, 1.025)
        ax[3].set_xlim(-0.025, 1.025)
        #ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.9), fontsize=8)
        ax[1].legend(loc='center left', bbox_to_anchor=(1, -.5), fontsize=14)
        
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.55) #wspace=None,

    elif orient == 'v':
        fig,ax=plt.subplots(2,2,sharex=False, figsize=figsize)
        ax=ax.flatten()
        box_plot(df.loc[df['Metric'].isin(['Velocity cosine'])], x='Metric', 
                 y='Score', orient=orient, hue='Model',ax=ax[0], palette = palette)
        box_plot(df.loc[df['Metric'].isin(['CBDir'])], x='Metric', 
                 y='Score', orient=orient, hue='Model',ax=ax[1], palette = palette)
        box_plot(df.loc[df['Metric'].isin(['NN velocity cosine'])], x='Metric', 
                 y='Score', orient=orient, hue='Model',ax=ax[2], palette = palette)
        box_plot(df.loc[df['Metric'].isin(['ICCoh'])], x='Metric', 
                 y='Score', orient=orient, hue='Model',ax=ax[3], palette = palette)
        
        for i in range(ax.shape[0]):
            ax[i].legend([],[], frameon=False)
            ax[i].spines.right.set_visible(False)
            ax[i].spines.top.set_visible(False)
            ax[i].set_ylabel('')
            ax[i].set_xlabel('')
            plt.setp(ax[i].get_yticklabels(), fontsize=19)
            plt.setp(ax[i].get_xticklabels(), fontsize=20)
        ax[0].set_ylim(-1.025, 1.025)
        ax[1].set_ylim(-0.025, 1.025)
        ax[2].set_ylim(-1.025, 1.025)
        ax[3].set_ylim(-0.025, 1.025)
        #ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.9), fontsize=8)
        ax[1].legend(loc='center left', bbox_to_anchor=(1, -.5), fontsize=20,frameon=False)
        
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.25,wspace=0.3) #wspace=None,
    
    return fig, ax

def transition_scores(transition_scores_raw, model_names=None, model_label = 'Model', raw=False, custom_colors = None, height=4, aspect=1):

    if model_names == None:

        if raw == False:
            
            transition_scores_ = pd.DataFrame({" ": [t[0] + r'$\rightarrow$' + t[1] for t in transition_scores_raw.keys()], "CBDir score": transition_scores_raw.values()})
            
            ax = sns.catplot(
                data=transition_scores_, 
                y=" ", x="CBDir score", kind='bar', color='k',
                ci="sd", alpha=.6, height=height, aspect=aspect, orient='h'
            ) #color='k', 
            #ax.set_xticklabels(rotation=90)
            ax.set(xlim=(-1, 1))
        
            #ax.fig.suptitle('Model c)')
            return ax

        else:
            
            score = []
            transition = []
            embedding=[]
            
            for key in transition_scores_raw.keys():
                for item in transition_scores_raw[key]:
                    score.append(item)
                    transition.append(key[0] + r'$\rightarrow$' + key[1])
            
            transition_scores_ = pd.DataFrame({'CBDir score':score, ' ':transition})
            
            PROPS = {
            'boxprops':{'facecolor':'darkgrey', 'edgecolor':'black'},
            'medianprops':{'color':'black'},
            'whiskerprops':{'color':'black'},
                'capprops':{'color':'black'}
            }
            
            ax = sns.catplot(
                data=transition_scores_, 
                y=' ', x="CBDir score", kind='box', 
                height=height, aspect=aspect, orient='h', fliersize=0, **PROPS, showmeans=True, notch=True,
                            meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"}
            ) #color='k', 
            #ax.set_xticklabels(rotation=90)
            ax.set(xlim=(-1, 1))
            return ax
        
    else:

        if raw == False:
            
            transition_scores_ = [pd.DataFrame({" ": [t[0] + r'$\rightarrow$' + t[1] for t in tscores.keys()], "CBDir score": tscores.values()}) for tscores in transition_scores_raw]
            
            for i in range(len(model_names)):
                transition_scores_[i][model_label] = model_names[i]
                
            transition_scores_ = pd.DataFrame(pd.concat(transition_scores_, axis=0))
            
            ax = sns.catplot(
                data=transition_scores_, 
                y=" ", x="CBDir score", kind='bar', hue=model_label,
                ci="sd", alpha=1, height=height, aspect=aspect, orient='h', palette = 'Set1', showmeans=True,
                            meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"}
            )
            ax.set(xlim=(-1, 1))
            
            return ax    

        else:

            score = []
            transition = []
            embedding=[]
            
            for items, name in zip(transition_scores_raw, model_names):
                for key in items.keys():
                    for item in items[key]:
                        score.append(item)
                        transition.append(key[0] + r'$\rightarrow$' + key[1])
                        embedding.append(name)

            transition_scores_ = pd.DataFrame({'CBDir score':score, ' ':transition, model_label:embedding})
            
            if np.any(custom_colors != None):
                ax = sns.catplot(
                    data=transition_scores_, 
                    y=' ', x="CBDir score", kind='box', hue=model_label,
                    height=height, aspect=aspect, orient='h', palette = custom_colors, fliersize=0
                ) 
            else:
                ax = sns.catplot(
                    data=transition_scores_, 
                    y=' ', x="CBDir score", kind='box', hue=model_label,
                    height=height, aspect=aspect, orient='h', palette = 'Set1', fliersize=0, showmeans=True,
                            meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"}
                )
            ax.set(xlim=(-1, 1))
            
            return ax    

def coherence_scores(transition_scores_raw, model_names=None, model_label = 'Model', raw=False, custom_colors = None, height=4, aspect=1):

    if model_names == None:

        if raw == False:
            
            transition_scores_ = pd.DataFrame({" ": [t for t in transition_scores_raw.keys()], "ICCoh score": transition_scores_raw.values()})
            
            ax = sns.catplot(
                data=transition_scores_, 
                y=" ", x="ICCoh score", kind='bar', color='k',
                ci="sd", alpha=.6, height=height, aspect=aspect, orient='h'
            ) #color='k', 
            #ax.set_xticklabels(rotation=90)
            ax.set(xlim=(-1, 1))
            
            return ax

        else:
            
            score = []
            transition = []
            embedding=[]
            
            for key in transition_scores_raw.keys():
                for item in transition_scores_raw[key]:
                    score.append(item)
                    transition.append(key)
            
            transition_scores_ = pd.DataFrame({'ICCoh score':score, ' ':transition})
            
            PROPS = {
            'boxprops':{'facecolor':'darkgrey', 'edgecolor':'black'},
            'medianprops':{'color':'black'},
            'whiskerprops':{'color':'black'},
                'capprops':{'color':'black'}
            }
            
            ax = sns.catplot(
                data=transition_scores_, 
                y=' ', x="ICCoh score", kind='box', 
                height=height, aspect=aspect, orient='h', fliersize=0, **PROPS, showmeans=True,
                            meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"}
            )
            ax.set(xlim=(-1, 1))
            return ax
        
    else:

        if raw == False:
            
            transition_scores_ = [pd.DataFrame({" ": [t for t in tscores.keys()], "ICCoh score": tscores.values()}) for tscores in transition_scores_raw]
            
            for i in range(len(model_names)):
                transition_scores_[i][model_label] = model_names[i]
                
            transition_scores_ = pd.DataFrame(pd.concat(transition_scores_, axis=0))
            
            ax = sns.catplot(
                data=transition_scores_, 
                y=" ", x="ICCoh score", kind='bar', hue=model_label,
                ci="sd", alpha=1, height=height, aspect=aspect, orient='h', palette = 'Set1', showmeans=True,
                            meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"}
            ) #color='k', 
            #ax.set_xticklabels(rotation=90)
            ax.set(xlim=(-1, 1))
            
            return ax    

        else:

            score = []
            transition = []
            embedding=[]
            
            for items, name in zip(transition_scores_raw, model_names):
                for key in items.keys():
                    for item in items[key]:
                        score.append(item)
                        transition.append(key)
                        embedding.append(name)

            transition_scores_ = pd.DataFrame({'ICCoh score':score, ' ':transition, model_label:embedding})
            
            if np.any(custom_colors != None):
                ax = sns.catplot(
                    data=transition_scores_, 
                    y=' ', x="ICCoh score", kind='box', hue=model_label,
                    height=height, aspect=aspect, orient='h', palette = custom_colors, fliersize=0
                )
            else:
                ax = sns.catplot(
                    data=transition_scores_, 
                    y=' ', x="ICCoh score", kind='box', hue=model_label,
                    height=height, aspect=aspect, orient='h', palette = 'Set1', fliersize=0, showmeans=True,
                            meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"}
                ) #color='k',
            #ax.set_xticklabels(rotation=90)
            ax.set(xlim=(-1, 1))
            
            #ax.fig.suptitle('Model c)')
            return ax    
        

def cosine_scores(cosine_scores_raw, model_names=None, model_label = 'Model', raw=False, custom_colors = None):

    if model_names == None:

        if raw == False:
            
            cosine_scores_ = pd.DataFrame({" ": [t[0] + r'$\rightarrow$' + t[1] for t in cosine_scores_raw.keys()], "Cosine similarity": cosine_scores_raw.values()})
            
            ax = sns.catplot(
                data=transition_scores_, 
                y=" ", x="Cosine similarity", kind='bar', color='k',
                ci="sd", alpha=.6, height=4, orient='h'
            ) #color='k', 
            #ax.set_xticklabels(rotation=90)
            ax.set(xlim=(-1, 1))
        
            return ax

        else:
            
            score = []
            transition = []
            embedding=[]
            
            for key in cosine_scores_raw.keys():
                for item in cosine_scores_raw[key]:
                    score.append(item)
                    transition.append(key[0] + r'$\rightarrow$' + key[1])
            
            transition_scores_ = pd.DataFrame({'Cosine similarity':score, ' ':transition})
            
            
            PROPS = {
            'boxprops':{'facecolor':'darkgrey', 'edgecolor':'black'},
            'medianprops':{'color':'black'},
            'whiskerprops':{'color':'black'},
            'capprops':{'color':'black'}
            }
            
            ax = sns.catplot(
                data=transition_scores_, 
                y=' ', x="Cosine similarity", kind='box', 
                height=4, orient='h', fliersize=0, **PROPS, showmeans=True,
                            meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"10"}
            ) #color='k', 
            #ax.set_xticklabels(rotation=90)
            ax.set(xlim=(-1, 1))
            return ax
        
    else:

        if raw == False:
            
            transition_scores_ = [pd.DataFrame({" ": [t[0] + r'$\rightarrow$' + t[1] for t in tscores.keys()], "Cosine similarity": tscores.values()}) for tscores in transition_scores_raw]
            
            for i in range(len(model_names)):
                transition_scores_[i][model_label] = model_names[i]
                
            transition_scores_ = pd.DataFrame(pd.concat(transition_scores_, axis=0))
            
            
            ax = sns.catplot(
                data=transition_scores_, 
                y=" ", x="Cosine similarity", kind='bar', hue=model_label,
                ci="sd", alpha=1, height=4, orient='h', palette = 'Set1'
            ) #color='k', 
            #ax.set_xticklabels(rotation=90)
            ax.set(xlim=(-1, 1))
            
            return ax    

        else:

            score = []
            transition = []
            embedding=[]
            
            for items, name in zip(cosine_scores_raw, model_names):
                for key in items.keys():
                    for item in items[key]:
                        score.append(item)
                        transition.append(key[0] + r'$\rightarrow$' + key[1])
                        embedding.append(name)

            transition_scores_ = pd.DataFrame({'Cosine similarity':score, ' ':transition, model_label:embedding})
            
            if np.any(custom_colors != None):
                ax = sns.catplot(
                    data=transition_scores_, 
                    y=' ', x="Cosine similarity", kind='box', hue=model_label,
                    height=4, orient='h', palette = custom_colors, fliersize=0
                ) #color='k',
            else:
                ax = sns.catplot(
                    data=transition_scores_, 
                    y=' ', x="Cosine similarity", kind='box', hue=model_label,
                    height=4, orient='h', palette = 'Set1', fliersize=0
                ) #color='k',
            #ax.set_xticklabels(rotation=90)
            ax.set(xlim=(-1, 1))
            
            #ax.fig.suptitle('Model c)')
            return ax    

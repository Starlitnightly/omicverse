import requests
import os
import numpy as np
import pandas as pd

def gptcelltype(input, tissuename=None, speciename='human',
                provider='qwen',model='qwen-plus', topgenenumber=10,
                base_url=None):
    """
    Annotation of cell types using AGI model.

    Arguments:
        input: dict, input dictionary with clusters as keys and gene markers as values.
        tissuename: str, tissue name.
        provider: str, provider of the model. Default: 'qwen', you can select from ['openai','kimi','qwen'] now.

    """
    from openai import OpenAI
    import os
    import numpy as np
    import pandas as pd
    if base_url is None:
        if provider == 'openai':
            base_url = "https://api.openai.com/v1/"
        elif provider == 'kimi':
            base_url = "https://api.moonshot.cn/v1"
        elif provider == 'qwen':
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_API_KEY = os.getenv("AGI_API_KEY")
    if QWEN_API_KEY == "":
        print("Note: AGI API key not found: returning the prompt itself.")
        API_flag = False
    else:
        API_flag = True
    client = OpenAI(
        api_key=QWEN_API_KEY, # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url=base_url,
    )
    
    if isinstance(input, dict):
        input = {k: 'unknown' if not v else ','.join(v[:topgenenumber]) for k, v in input.items()}
    elif isinstance(input, pd.DataFrame):
        # Filter genes with positive log fold change and group by cluster, selecting top genes
        input = input[input['logfoldchanges'] > 0]
        input = input.groupby('cluster')['names'].apply(lambda x: ','.join(x.iloc[:topgenenumber]))
    else:
        raise ValueError("Input must be either a dictionary of lists or a pandas DataFrame.")
    
    if not API_flag:
        message = f'Identify cell types of {tissuename} cells in {speciename} using the following markers separately for each row. Only provide the cell type name. Do not show numbers before the name. Do not show numbers before the name. Some can be a mixture of multiple cell types.\n' + '\n'.join([f'{k}: {v}' for k, v in input.items()])
        return message
    else:
        print("Note: AGI API key found: returning the cell type annotations.")
        cutnum = int(np.ceil(len(input) / 30))
        if cutnum > 1:
            cid = np.digitize(range(1, len(input) + 1), bins=np.linspace(1, len(input), cutnum + 1))
        else:
            cid = np.ones(len(input), dtype=int)
        
        allres = {}
        for i in range(1, cutnum + 1):
            id_list = [j for j, x in enumerate(cid) if x == i]
            flag = False
            while not flag:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", 
                               "content": f'Identify cell types of {tissuename} cells using the following markers separately for each row. Only provide the cell type name. Do not show numbers before the name. Some can be a mixture of multiple cell types.\n' + '\n'.join([input[list(input.keys())[j]] for j in id_list if input[list(input.keys())[j]] != 'unknown'])}]
                )
                #return response
                res = response.choices[0].message.content.split('\n')
                if len(res) == len(id_list):
                    flag = True
            for idx, cell_type in zip(id_list, res):
                key = list(input.keys())[idx]
                allres[key] = 'unknown' if input[key] == 'unknown' else cell_type.strip(',')
        
        
        print('Note: It is always recommended to check the results returned by GPT-4 in case of AI hallucination, before going to downstream analysis.')
        return allres
    


def gpt4celltype(input_data, tissuename=None, speciename='human',
                provider='qwen', model='qwen-plus', topgenenumber=10,
                base_url=None):
    """
    Annotation of cell types using AGI model.

    Arguments:
        input: dict, input dictionary with clusters as keys and gene markers as values.
        tissuename: str, tissue name.
        provider: str, provider of the model. Default: 'qwen', you can select from ['openai','kimi','qwen'] now.

    """
    input=input_data.copy()
    input_data=input
    del_k=[]
    for k in input_data:
        if len(input_data[k])==0:
            del_k.append(k)
    for k in del_k:
        del input[k]
    
    if base_url is None:
        if provider == 'openai':
            base_url = "https://api.openai.com/v1"
        elif provider == 'kimi':
            base_url = "https://api.moonshot.cn/v1"
        elif provider == 'qwen':
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    QWEN_API_KEY = os.getenv("AGI_API_KEY")
    if QWEN_API_KEY == "":
        print("Note: AGI API key not found: returning the prompt itself.")
        API_flag = False
    else:
        API_flag = True

    if isinstance(input, dict):
        input = {k: 'unknown' if not v else ','.join(v[:topgenenumber]) for k, v in input.items()}
    elif isinstance(input, pd.DataFrame):
        # Filter genes with positive log fold change and group by cluster, selecting top genes
        input = input[input['logfoldchanges'] > 0]
        input = input.groupby('cluster')['names'].apply(lambda x: ','.join(x.iloc[:topgenenumber]))
    else:
        raise ValueError("Input must be either a dictionary of lists or a pandas DataFrame.")

    
    if not API_flag:
        message = f'Identify cell types of {tissuename} cells in {speciename} using the following markers separately for each row. Only provide the cell type name. Do not show numbers before the name. Some can be a mixture of multiple cell types.\n' + '\n'.join([f'{k}: {v}' for k, v in input.items()])
        return message
    else:
        print("Note: AGI API key found: returning the cell type annotations.")
        
        headers = {
            "Authorization": f"Bearer {QWEN_API_KEY}",
        }
        
        cutnum = int(np.ceil(len(input) / 30))
        if cutnum > 1:
            cid = np.digitize(range(1, len(input) + 1), bins=np.linspace(1, len(input), cutnum + 1))
        else:
            cid = np.ones(len(input), dtype=int)
        
        allres = {}
        from tqdm import tqdm
        for i in tqdm(range(1, cutnum + 1)):
            id_list = [j for j, x in enumerate(cid) if x == i]
            flag = False
            while not flag:
                messages = [{"role": "user", 
                             "content": f'Identify cell types of {tissuename} cells using the following markers separately for each row. Only provide the cell type name. Do not show numbers before the name. Some can be a mixture of multiple cell types.\n' + '\n'.join([input[list(input.keys())[j]] for j in id_list if input[list(input.keys())[j]] != 'unknown'])}]
                
                params = {
                    "model": model,
                    "messages": messages
                }
                
                
                response = requests.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=params,
                    stream=False
                )
                
                res = response.json()
                
                if 'choices' in res and len(res['choices']) > 0:
                    res_content = res['choices'][0]['message']['content'].split('\n')
                    #print(res_content)
                    if len(res_content) == len(id_list):
                        flag = True
                        for idx, cell_type in zip(id_list, res_content):
                            key = list(input.keys())[idx]
                            allres[key] = 'unknown' if input[key] == 'unknown' else cell_type.strip(',')
        
        print('Note: It is always recommended to check the results returned by GPT-4 in case of AI hallucination, before going to downstream analysis.')
        for k in del_k:
            allres[k]='Unknown'
        return allres
    


def get_cluster_celltype(cluster_celltypes, cluster_markers, species, organization,
                        model,base_url,provider):
    import os
    import numpy as np
    import pandas as pd
    import requests
    if base_url is None:
        if provider == 'openai':
            base_url = "https://api.openai.com/v1"
        elif provider == 'kimi':
            base_url = "https://api.moonshot.cn/v1"
        elif provider == 'qwen':
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_API_KEY = os.getenv("AGI_API_KEY")
    
    
    # 在这里配置您在本站的API_KEY
    api_key = QWEN_API_KEY
    
    headers = {
        "Authorization": 'Bearer ' + api_key,
    }
    cluster_celltype = {}
    from tqdm import tqdm
    for cluster_id, celltypes in tqdm(cluster_celltypes.items()):
        markers = cluster_markers.get(cluster_id, [])
        question = (
                    f"Given the species: {species} and organization: {organization}, "
                    f"determine the most suitable cell type for cluster {cluster_id}. "
                    f"The possible cell types are: {', '.join(celltypes)}. "
                    f"The gene markers for this cluster are: {', '.join(markers)}. "
                    f"Which cell type best represents this cluster? "
                    f"Only provide the cell type name. Do not show numbers before the name. Some can be a mixture of multiple cell types."
                    f"Do not provide the plural form of celltype."
                )
        #print(question)
        
        params = {
            "messages": [
        
                {
                    "role": 'user',
                    "content": question
                }
            ],
            # 如果需要切换模型，在这里修改
            "model": model
        }
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=params,
            stream=False
        )
        res = response.json()
        answer = res['choices'][0]['message']['content'].split('\n')
        # 将回答加入结果字典
        cluster_celltype[cluster_id] = answer[0].lower()
    
    return cluster_celltype


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
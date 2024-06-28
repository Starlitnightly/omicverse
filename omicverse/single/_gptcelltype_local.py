

def gptcelltype_local(input, tissuename=None, speciename='human',
                model_name='Qwen/Qwen2-7B-Instruct', topgenenumber=10):
    """
    Annotation of cell types using a local LLM model.

    Arguments:
        input: dict, input dictionary with clusters as keys and gene markers as values. \
            e.g. {'cluster1': ['gene1', 'gene2'], 'cluster2': ['gene3']}
        tissuename: str, tissue name.
        speciename: str, species name. Default: 'human'.
        model_name: str, the name or path of the local model to be used.
        topgenenumber: int, the number of top genes to consider for each cluster. Default: 10.
    """
    import re
    import numpy as np
    import pandas as pd
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    
    # Load the model and tokenizer from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map='cuda', 
        torch_dtype='auto', 
        trust_remote_code=True
    )
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer
    )
    
    if isinstance(input, dict):
        input = {k: 'unknown' if not v else ','.join(v[:topgenenumber]) for k, v in input.items()}
    elif isinstance(input, pd.DataFrame):
        # Filter genes with positive log fold change and group by cluster, selecting top genes
        input = input[input['logfoldchanges'] > 0]
        input = input.groupby('cluster')['names'].apply(lambda x: ','.join(x.iloc[:topgenenumber]))
    else:
        raise ValueError("Input must be either a dictionary of lists or a pandas DataFrame.")
    
    message_template = (
        f"Identify cell types of {tissuename} cells in {speciename} using the above markers separately for each row.\n"
        "Provide the cell type name, followed by the reason, which should be enclosed in square brackets.\n"
        "Some can be a mixture of multiple cell types. If so, seperate them with semicolon.\n\n"
        "Output format:\n"
        "cluster: cell type [marker(s)]\n"
        "Output example:\n"
        "0: T cells [CD3D, IL7R]\n"
        "1: Cytotoxic T cells [CCL5, NKG7, GZMA]; Natural Killer (NK) cells [GNLY, KLRD1]"
    )
    
    allres = {}
    cutnum = int(np.ceil(len(input) / 30))
    if cutnum > 1:
        cid = np.digitize(range(1, len(input) + 1), bins=np.linspace(1, len(input), cutnum + 1))
    else:
        cid = np.ones(len(input), dtype=int)
    
    for i in range(1, cutnum + 1):
        id_list = [j for j, x in enumerate(cid) if x == i]
        message = '\n'.join(
            [f"{k}: {v}" for k, v in input.items() if list(input.keys()).index(k) in id_list]
        ) + '\n\n' + message_template

        messages = [
            {"role": "system", "content": "You are an experienced biologist with particular expertise in molecular biology, cell biology, and bioinformatics."},
            {"role": "user", "content": message},
        ]

        generation_args = {
            "max_new_tokens": 5000,
            "return_full_text": False,
            "temperature": 0.3,
            "do_sample": False,
        }
        generated = pipe(messages, **generation_args)
        print(generated[0]['generated_text'])

        pattern = r'\d+:\s+(.+?)\s+\[.*?\]'
        res = re.findall(pattern, generated[0]['generated_text'])
        for idx, cell_type in zip(id_list, res):
            key = list(input.keys())[idx]
            allres[key] = 'unknown' if input[key] == 'unknown' else cell_type

    print('Note: It is always recommended to check the results returned by the LLM in case of AI hallucination, before going to downstream analysis.')
    return allres

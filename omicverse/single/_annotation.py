from anndata import AnnData
import scanpy as sc


class Annotation(object):

    def __init__(self, adata: AnnData):
        self.adata = adata
        self.cellxgene_desc_df=None

    def add_reference_sc(self, reference: AnnData):
        self.adata_ref=reference

    def add_reference_pkl(self, reference: str):
        self.pkl_ref=reference

        from celltypist import models
        self.model = models.Model.load(model = self.pkl_ref)

    def query_reference(
        self,
        source='cellxgene',
        data_desc:str=None,
        llm_model='gpt-4o-mini',
        llm_api_key='sk*',
        llm_provider='openai',
        llm_base_url='https://api.openai.com/v1',
        llm_extra_params={},
    ):
        if self.cellxgene_desc_df is None and source=='cellxgene':
            self.cellxgene_desc_df=_cellxgene_scrape_with_api()
            print(f"CellxGene description dataframe saved to self.cellxgene_desc_df")

        #use llm to query the appropriate cellxgene collection datasets from cellxgene_desc_df's description column
        #find the most relevant cellxgene collection datasets from cellxgene_desc_df's description column by the data_desc
        

    def annotate(
        self,
        method='celltypist',
    ):
        if method=='celltypist':
            import celltypist
            predictions = celltypist.annotate(
                self.adata, model = self.pkl_ref,
                majority_voting = True
            )
            self.adata.obs['celltypist_prediction'] = predictions.predicted_labels
            print(f"Celltypist prediction saved to adata.obs['celltypist_prediction']")



import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json

# 方案3: 直接访问 API (如果可用)
def _cellxgene_scrape_with_api():
    """尝试直接访问 CellxGene API"""
    api_url = "https://api.cellxgene.cziscience.com/curation/v1/collections"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    print(f"正在访问 API: {api_url}")
    response = requests.get(api_url, headers=headers, timeout=30)

    if response.status_code == 200:
        print(f"✓ API 访问成功 (状态码: {response.status_code})")
        data = response.json()

        # 解析 JSON 数据为 DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'collections' in data:
            df = pd.DataFrame(data['collections'])
        else:
            df = pd.json_normalize(data)

        return df
    else:
        print(f"✗ API 访问失败 (状态码: {response.status_code})")
        return None
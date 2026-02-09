from __future__ import annotations
import json
import logging
import os
import pickle
import re
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm
from .._registry import register_function

@register_function(
    aliases=["细胞本体映射", "CellOntologyMapper", "cellontology_mapper", "细胞本体注释", "本体映射器"],
    category="single",
    description="Comprehensive cell ontology mapping using Cell Ontology and Cell Taxonomy with AI-powered abbreviation expansion",
    examples=[
        "# Basic Cell Ontology mapping",
        "mapper = ov.single.CellOntologyMapper(",
        "    cl_obo_file='cl.json',",
        "    model_name='sentence-transformers/all-MiniLM-L6-v2')",
        "# Map cell types to ontology",
        "results = mapper.map_adata(adata, cell_name_col='celltype')",
        "# Setup LLM for abbreviation expansion",
        "mapper.setup_llm_expansion(api_type='openai', api_key='sk-***',",
        "                           tissue_context='Brain', species='mouse')",
        "# Enhanced mapping with LLM expansion",
        "results = mapper.map_adata_with_expansion(adata, cell_name_col='celltype',",
        "                                         expand_abbreviations=True)",
        "# Load Cell Taxonomy for enhanced annotations",
        "mapper.load_cell_taxonomy_resource('Cell_Taxonomy_resource.txt')",
        "results = mapper.map_adata_with_taxonomy(adata, cell_name_col='celltype',",
        "                                        use_taxonomy=True, species='Homo sapiens')",
        "# Query specific cell types",
        "similar = mapper.find_similar_cells('T helper cell', top_k=5)",
        "info = mapper.get_cell_info('regulatory T cell')"
    ],
    related=["single.download_cl", "single.pySCSA", "single.gptcelltype"]
)
class CellOntologyMapper:
    """
    🧬 Cell ontology mapping class using NLP
    """
    
    def __init__(self, cl_obo_file=None, embeddings_path=None, model_name="all-mpnet-base-v2", local_model_dir=None):
        """
        🚀 Initialize CellOntologyMapper
        
        Parameters
        ----------
        cl_obo_file : str, optional
            📄 Cell Ontology OBO file path
        embeddings_path : str, optional
            💾 Pre-computed embeddings file path
        model_name : str
            🤖 Sentence Transformer model name
        local_model_dir : str, optional
            📁 Local directory to save downloaded models (avoid default cache)
        """
        self.model_name = model_name
        self.model = None
        self.local_model_dir = local_model_dir
        self.ontology_embeddings = None
        self.ontology_labels = None
        self.popv_dict = None
        self.output_path = None
        
        # LLM expansion functionality related
        self.llm_client = None
        self.llm_config = None
        self.abbreviation_cache = {}
        self.cache_file = None
        
        # Context information
        self.tissue_context = None
        self.species = "human"
        self.study_context = None
        
        # Cell taxonomy resource (treated as additional ontology)
        self.taxonomy_resource = None
        self.taxonomy_embeddings = None
        self.taxonomy_labels = None
        self.taxonomy_info_dict = None  # For quick lookup of detailed info
        # Initialize based on provided parameters
        if embeddings_path and os.path.exists(embeddings_path):
            print("📥 Loading existing ontology embeddings...")
            self.load_embeddings(embeddings_path)
        elif cl_obo_file and os.path.exists(cl_obo_file):
            print("🔨 Creating ontology resources from OBO file...")
            self.create_ontology_resources(cl_obo_file)
        else:
            print("?  Initialized empty mapper, please use load_embeddings() or create_ontology_resources()")
    
    def setup_llm_expansion(self, api_type="openai", api_key=None, model="gpt-3.5-turbo", 
                           base_url=None, cache_file="abbreviation_cache.json", 
                           tissue_context=None, species="human", study_context=None,
                           extra_params=None):
        """
        🤖 Setup LLM API for abbreviation expansion
        
        Parameters
        ----------
        api_type : str
            🔌 API type: "openai", "anthropic", "ollama", "qwen", "ernie", "glm", "spark", "doubao", "custom_openai"
        api_key : str, optional
            🔐 API key
        model : str
            🧠 Model name
        base_url : str, optional
            🌐 Custom API base URL (required for custom_openai and some domestic models)
        cache_file : str
            💽 Cache file path
        tissue_context : str or list, optional
            🧬 Tissue context information, e.g. "immune system", "brain", "liver" or list of tissues
        species : str
            🐭 Species, default "human", can also be "mouse", "rat", etc.
        study_context : str, optional
            🔬 Study context, e.g. "cancer", "development", "aging", "disease", etc.
        extra_params : dict, optional
            🔧 Additional parameters for specific APIs
        """
        self.llm_config = {
            'api_type': api_type,
            'api_key': api_key,
            'model': model,
            'base_url': base_url,
            'extra_params': extra_params or {}
        }
        
        # New context information
        self.tissue_context = tissue_context
        self.species = species
        self.study_context = study_context
        
        self.cache_file = cache_file
        self._load_abbreviation_cache()
        
        # Initialize client
        try:
            if api_type == "openai":
                import openai
                if base_url:
                    self.llm_client = openai.OpenAI(api_key=api_key, base_url=base_url)
                else:
                    self.llm_client = openai.OpenAI(api_key=api_key)
                    
            elif api_type == "custom_openai":
                # Generic OpenAI-compatible API with custom base_url
                import openai
                if not base_url:
                    raise ValueError("🌐 base_url is required for custom_openai API type")
                self.llm_client = openai.OpenAI(api_key=api_key, base_url=base_url)
                
            elif api_type == "anthropic":
                import anthropic
                self.llm_client = anthropic.Anthropic(api_key=api_key)
                
            elif api_type == "ollama":
                import requests
                self.llm_client = "ollama"  # Mark as ollama, use requests
                self.base_url = base_url or "http://localhost:11434"
                
            elif api_type == "qwen":
                # 阿里云通义千问 (DashScope)
                try:
                    import dashscope
                    if api_key:
                        dashscope.api_key = api_key
                    elif 'DASHSCOPE_API_KEY' not in os.environ:
                        raise ValueError("🔑 API key required for Qwen. Set api_key parameter or DASHSCOPE_API_KEY environment variable")
                    self.llm_client = "qwen"
                    print("🤖 Using DashScope API (通义千问)")
                except ImportError:
                    print("📦 Installing DashScope SDK: pip install dashscope")
                    raise ImportError("Please install: pip install dashscope")
                    
            elif api_type == "ernie":
                # 百度文心一言
                try:
                    import ernie
                    if api_key:
                        # Assume api_key is in format "access_key:secret_key"
                        if ':' in api_key:
                            access_key, secret_key = api_key.split(':', 1)
                            ernie.api_type = 'aistudio'
                            ernie.access_token = access_key
                        else:
                            ernie.api_type = 'aistudio'  
                            ernie.access_token = api_key
                    self.llm_client = "ernie"
                    print("🤖 Using ERNIE API (文心一言)")
                except ImportError:
                    print("📦 Installing ERNIE SDK: pip install ernie-bot-sdk")
                    raise ImportError("Please install: pip install ernie-bot-sdk")
                    
            elif api_type == "glm":
                # 智谱AI GLM
                try:
                    import zhipuai
                    if api_key:
                        zhipuai.api_key = api_key
                    elif 'ZHIPUAI_API_KEY' not in os.environ:
                        raise ValueError("🔑 API key required for GLM. Set api_key parameter or ZHIPUAI_API_KEY environment variable")
                    self.llm_client = "glm"
                    print("🤖 Using ZhipuAI API (智谱GLM)")
                except ImportError:
                    print("📦 Installing ZhipuAI SDK: pip install zhipuai")
                    raise ImportError("Please install: pip install zhipuai")
                    
            elif api_type == "spark":
                # 讯飞星火
                import requests
                if not api_key:
                    raise ValueError("🔑 API key required for Spark API")
                # Assume api_key format: "app_id:api_key:api_secret"
                if api_key.count(':') != 2:
                    raise ValueError("🔑 Spark API key should be in format: 'app_id:api_key:api_secret'")
                app_id, spark_api_key, api_secret = api_key.split(':')
                self.llm_config['app_id'] = app_id
                self.llm_config['spark_api_key'] = spark_api_key
                self.llm_config['api_secret'] = api_secret
                self.llm_client = "spark"
                print("🤖 Using iFlytek Spark API (讯飞星火)")
                
            elif api_type == "doubao":
                # 字节跳动豆包 (使用通用OpenAI兼容格式)
                import openai
                default_base_url = "https://ark.cn-beijing.volces.com/api/v3"
                actual_base_url = base_url or default_base_url
                self.llm_client = openai.OpenAI(api_key=api_key, base_url=actual_base_url)
                print(f"🤖 Using Doubao API (豆包) - Base URL: {actual_base_url}")
                
            else:
                supported_types = ["openai", "custom_openai", "anthropic", "ollama", "qwen", "ernie", "glm", "spark", "doubao"]
                print(f"✗ Unsupported API type: {api_type}")
                print(f"💡 Supported types: {', '.join(supported_types)}")
                return False
            
            print(f"✓ LLM expansion functionality setup complete (Type: {api_type}, Model: {model})")
            if base_url and api_type not in ["doubao"]:  # doubao already prints base_url
                print(f"🌐 Custom Base URL: {base_url}")
            if tissue_context:
                print(f"🧬 Tissue context: {tissue_context}")
            if study_context:
                print(f"🔬 Study context: {study_context}")
            print(f"🐭 Species: {species}")
            return True
            
        except ImportError as e:
            print(f"✗ Missing required library: {e}")
            print("📦 Install required packages based on your API type:")
            print("   - OpenAI: pip install openai")
            print("   - Anthropic: pip install anthropic")
            print("   - 通义千问: pip install dashscope")
            print("   - 文心一言: pip install ernie-bot-sdk")
            print("   - 智谱GLM: pip install zhipuai")
            print("   - Ollama/Spark/Doubao: pip install requests")
            return False
        except Exception as e:
            print(f"✗ LLM setup failed: {e}")
            return False
    
    def _load_abbreviation_cache(self):
        """📥 Load abbreviation cache"""
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.abbreviation_cache = json.load(f)
                print(f"✓ Loaded {len(self.abbreviation_cache)} cached abbreviation expansions")
            except:
                self.abbreviation_cache = {}
        else:
            self.abbreviation_cache = {}
    
    def _save_abbreviation_cache(self):
        """💾 Save abbreviation cache"""
        if self.cache_file:
            try:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.abbreviation_cache, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"✗ Failed to save cache: {e}")
    
    def _is_likely_abbreviation(self, cell_name):
        """🔍 Determine if it's likely an abbreviation (improved pattern matching)"""
        cell_name = cell_name.strip()
        
        # 如果为空或过长，不太可能是缩写
        if not cell_name or len(cell_name) > 20:
            return False
        
        # 明确不是缩写的常见细胞类型（添加这个检查）
        non_abbreviations = [
            'T cell', 'B cell', 'T cells', 'B cells',
            'NK cell', 'NK cells', 'dendritic cell', 'dendritic cells',
            'memory cell', 'naive cell', 'plasma cell', 'stem cell',
            'killer cell', 'helper cell', 'regulatory cell',
            'cytotoxic cell', 'effector cell', 'progenitor cell'
        ]
        
        if cell_name.lower() in [x.lower() for x in non_abbreviations]:
            return False
        
        # 包含 "cell" 或 "cells" 的长短语通常不是缩写
        if 'cell' in cell_name.lower() and len(cell_name) > 6:
            return False
        
        # 常见的明确缩写模式
        explicit_patterns = [
            r'^[A-Z]{2,5}$',                    # 全大写字母，2-5个字符：NK, DC, CTL
            r'^[A-Z][a-z]?[A-Z]+$',            # 混合大小写：Th1, Tc, NK
            r'.*\+$',                           # 以+结尾: CD4+, CD8+
            r'.*-$',                            # 以-结尾: CD8-, Memory-
            r'^CD\d+[\+\-]?$',                 # CD开头+数字: CD4, CD8+, CD25-
            r'^[A-Z]+\d+[\+\-]?$',             # 字母+数字: Th1, Tc17, NK22+
            r'^[A-Z]{1,3}\d+[A-Za-z]*[\+\-]?$', # 更复杂的字母数字组合
        ]
        
        # 复杂缩写模式（新增）
        complex_patterns = [
            r'^[A-Z]{1,3}\.[A-Za-z]+$',        # 点分隔：TA.Early, NK.dim, DC.mature
            r'^[A-Z]{1,4}_[A-Za-z]+$',         # 下划线分隔：T_reg, NK_bright, DC_plasmacytoid
            r'^[A-Z]{1,3}[0-9]*\.[A-Z]{1,3}[0-9]*$', # 点分隔字母数字：CD8.CM, Th1.Mem
            r'^[A-Za-z]{2,4}\.[0-9]+$',        # 字母点数字：Th.1, Tc.17
            r'^[A-Z]+/[A-Z]+$',                # 斜杠分隔：CD4/CD8, NK/NKT
            r'^[A-Z]{1,3}[0-9]+[a-z]+$',       # 数字后小写：CD4lo, CD8hi, CD25dim
        ]
        
        # 生物学特异性模式
        bio_specific_patterns = [
            r'^[A-Z]{2,4}[\+\-]{1,3}$',        # 多个+/-：CD4++, CD8--, TCR+/-
            r'^[A-Z]{1,3}[0-9]*[a-z]{2,4}$',   # 后缀模式：CD4bright, CD8dim, NKbright
            r'^[A-Z]{1,4}[0-9]*[A-Z][a-z]*$',  # 混合大小写：CD4Mem, CD8Eff, NKDim
            r'^p[A-Z]{1,3}[0-9]*[\+\-]?$',     # p开头：pDC, pTreg, pNK+
            r'^[A-Z]{1,3}SP$',                 # SP结尾：CD4SP, CD8SP (Single Positive)
            r'^[A-Z]{1,3}DP$',                 # DP结尾：CD4DP (Double Positive)
            r'^[A-Z]{1,3}DN$',                 # DN结尾：CD4DN (Double Negative)
        ]
        
        # 组织特异性缩写
        tissue_specific_patterns = [
            r'^[A-Z]{2,3}C$',                  # 以C结尾的细胞：HSC, MSC, NSC
            r'^[A-Z]{1,3}[0-9]*[A-Z]$',       # 字母数字字母：AT1, AT2, PT, DT
            r'^[A-Z]{2,4}[0-9]*$',             # 短字母数字组合：OPC, OL, MG, AC
        ]
        
        # 检查所有模式
        all_patterns = (explicit_patterns + complex_patterns + 
                       bio_specific_patterns + tissue_specific_patterns)
        
        for pattern in all_patterns:
            if re.match(pattern, cell_name):
                return True
        
        # 长度和字符组合判断（改进）
        # 短且包含大写字母的可能是缩写，但排除常见词组
        if len(cell_name) <= 6:
            # 包含大写字母
            if any(c.isupper() for c in cell_name):
                # 但不是常见的非缩写词组
                if ' ' not in cell_name:  # 单个词更可能是缩写
                    return True
            # 全小写但很短且常见
            if cell_name.lower() in ['nk', 'dc', 'th', 'tc', 'treg', 'ctl']:
                return True
        
        # 包含特殊符号通常是缩写
        special_chars = ['+', '-', '.', '_', '/', ':']
        if any(char in cell_name for char in special_chars):
            return True
        
        # 数字混合模式
        if re.search(r'[A-Za-z]+[0-9]+', cell_name) or re.search(r'[0-9]+[A-Za-z]+', cell_name):
            return True
        
        # 多个连续大写字母（但不是常见词组）
        if re.search(r'[A-Z]{2,}', cell_name) and ' ' not in cell_name:
            return True
        
        # 常见细胞类型缩写词典检查
        common_abbreviations = {
            # 免疫细胞
            'nk', 'dc', 'th', 'tc', 'treg', 'ctl', 'nkt', 'mait', 'ilc',
            'pdc', 'cdc', 'mdc', 'tam', 'tac', 'til', 'caf', 'msc',
            # 神经细胞
            'opc', 'ol', 'mg', 'ac', 'pv', 'sst', 'vip', 'cck',
            # 其他组织
            'hsc', 'lsc', 'kc', 'pt', 'dt', 'ic', 'pc', 'pod',
            'at1', 'at2', 'am', 'club', 'hep', 'lsec',
            # 干细胞和祖细胞
            'esc', 'ipsc', 'npc', 'gpc', 'rpc', 'cpc',
            # 癌细胞相关
            'csc', 'ctc', 'caf', 'tam', 'tex', 'tn'
        }
        
        if cell_name.lower() in common_abbreviations:
            return True
        
        # 检查是否包含常见缩写作为子串（但不是包含 cell 的长词组）
        for abbr in common_abbreviations:
            if abbr in cell_name.lower() and len(cell_name) <= 10 and 'cell' not in cell_name.lower():
                return True
        
        return False
    
    def test_abbreviation_detection(self, test_cases=None):
        """
        🧪 Test abbreviation detection with various examples
        
        Parameters
        ----------
        test_cases : dict, optional
            📝 Custom test cases {cell_name: expected_result}
        """
        if test_cases is None:
            # 预设测试用例
            test_cases = {
                # 应该被识别为缩写的
                'NK': True,
                'DC': True, 
                'TA.Early': True,
                'CD4+': True,
                'CD8-': True,
                'Th1': True,
                'Treg': True,
                'pDC': True,
                'NK.dim': True,
                'T_reg': True,
                'CD8.CM': True,
                'Th.1': True,
                'CD4/CD8': True,
                'CD4lo': True,
                'CD8hi': True,
                'CD4++': True,
                'CD8SP': True,
                'HSC': True,
                'OPC': True,
                'AT1': True,
                'TAM': True,
                'CTL': True,
                'NKT': True,
                'pTreg': True,
                # 不应该被识别为缩写的
                'T cell': False,
                'Natural killer cell': False,
                'Dendritic cell': False,
                'Regulatory T cell': False,
                'Memory T cell': False,
                'Naive B cell': False,
                'Activated macrophage': False,
                'Cytotoxic T lymphocyte': False,
                'Helper T cell': False,
                'Plasma cell': False,
            }
        
        print("Testing abbreviation detection...")
        print("=" * 60)
        
        correct = 0
        total = 0
        errors = []
        
        for cell_name, expected in test_cases.items():
            result = self._is_likely_abbreviation(cell_name)
            total += 1
            
            if result == expected:
                correct += 1
                status = "✓"
            else:
                errors.append((cell_name, expected, result))
                status = "✗"
            
            print(f"{status} {cell_name:<20} Expected: {expected}, Got: {result}")
        
        print("\n" + "=" * 60)
        print(f"Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
        
        if errors:
            print(f"\n✗ Errors ({len(errors)}):")
            for cell_name, expected, got in errors:
                print(f"  - {cell_name}: expected {expected}, got {got}")
        else:
            print("All tests passed!")
        
        return correct / total
    
    def _call_llm_for_expansion(self, cell_name):
        """🤖 Call LLM for abbreviation expansion"""
        if self.llm_client is None:
            return None
        
        # Build context information
        context_parts = []
        
        if self.species and self.species != "human":
            context_parts.append(f"Species: {self.species}")
        
        if self.tissue_context:
            if isinstance(self.tissue_context, list):
                tissue_info = ", ".join(self.tissue_context)
            else:
                tissue_info = self.tissue_context
            context_parts.append(f"Tissue/Organ context: {tissue_info}")
        
        if self.study_context:
            context_parts.append(f"Study context: {self.study_context}")
        
        context_str = "\n".join(context_parts) if context_parts else ""
        
        # Get context-specific examples
        examples = self._get_context_specific_examples()
        
        prompt = f"""You are an expert in cell biology and immunology. Your task is to expand cell type abbreviations to their full, standard names.

{context_str}

Given the cell type abbreviation: "{cell_name}"

Please provide:
1. The most likely full name for this cell type
2. Alternative possible full names if there are multiple interpretations
3. Your confidence level (high/medium/low)

{examples}

Please respond in JSON format:
{{
    "primary_expansion": "most likely full name",
    "alternatives": ["alternative1", "alternative2"],
    "confidence": "high/medium/low",
    "reasoning": "brief explanation"
}}

Cell type abbreviation: {cell_name}"""

        # Initialize content variable
        content = None
        api_type = self.llm_config.get('api_type', 'unknown')
        
        try:
            if api_type in ["openai", "custom_openai", "doubao"]:
                # OpenAI API and compatible APIs
                response = self.llm_client.chat.completions.create(
                    model=self.llm_config['model'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=300,
                    **self.llm_config.get('extra_params', {})
                )
                content = response.choices[0].message.content
                
            elif api_type == "anthropic":
                response = self.llm_client.messages.create(
                    model=self.llm_config['model'],
                    max_tokens=300,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}],
                    **self.llm_config.get('extra_params', {})
                )
                content = response.content[0].text
                
            elif api_type == "ollama":
                import requests
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.llm_config['model'],
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1, **self.llm_config.get('extra_params', {})}
                    },
                    timeout=30
                )
                response.raise_for_status()
                content = response.json().get('response', '')
                
            elif api_type == "qwen":
                # 阿里云通义千问
                import dashscope
                from dashscope import Generation
                response = dashscope.Generation.call(
                    model=self.llm_config['model'] or 'qwen-turbo',
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.1,
                    max_tokens=300,
                    **self.llm_config.get('extra_params', {})
                )
                if response.status_code == 200:
                    content = response.output.text
                else:
                    raise Exception(f"Qwen API error: {response.message}")
                    
            elif api_type == "ernie":
                # 百度文心一言
                import ernie
                response = ernie.ChatCompletion.create(
                    model=self.llm_config['model'] or 'ernie-bot',
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.1,
                    **self.llm_config.get('extra_params', {})
                )
                content = response.get_result()
                
            elif api_type == "glm":
                # 智谱AI GLM
                import zhipuai
                response = zhipuai.model_api.invoke(
                    model=self.llm_config['model'] or 'chatglm_turbo',
                    prompt=[{'role': 'user', 'content': prompt}],
                    temperature=0.1,
                    **self.llm_config.get('extra_params', {})
                )
                if response['code'] == 200:
                    content = response['data']['choices'][0]['content']
                else:
                    raise Exception(f"GLM API error: {response.get('msg', 'Unknown error')}")
                    
            elif api_type == "spark":
                # 讯飞星火 (WebSocket API)
                content = self._call_spark_api(prompt)
                
            else:
                raise ValueError(f"Unsupported API type: {api_type}")
            
        except Exception as e:
            print(f"✗ LLM call failed ({api_type}): {e}")
            return None
        
        # Check if content was successfully retrieved
        if content is None:
            print(f"✗ No content received from {api_type} API")
            return None
        
        # Parse JSON response
        try:
            # Extract JSON part
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                # If no JSON format, try parsing text
                return {"primary_expansion": content.strip(), "confidence": "low"}
                
        except json.JSONDecodeError:
            return {"primary_expansion": content.strip(), "confidence": "low"}
    
    def _call_spark_api(self, prompt):
        """🔥 Call iFlytek Spark API using WebSocket"""
        import requests
        import json
        import time
        import hashlib
        import hmac
        import base64
        from urllib.parse import urlencode
        
        # Spark API endpoint (using HTTP API instead of WebSocket for simplicity)
        url = "https://spark-api.xf-yun.com/v1.1/chat/completions"
        
        app_id = self.llm_config['app_id']
        api_key = self.llm_config['spark_api_key'] 
        api_secret = self.llm_config['api_secret']
        
        # Generate authentication
        timestamp = str(int(time.time()))
        signature_string = f"host: spark-api.xf-yun.com\ndate: {timestamp}\nGET /v1.1/chat/completions HTTP/1.1"
        signature = base64.b64encode(
            hmac.new(api_secret.encode(), signature_string.encode(), hashlib.sha256).digest()
        ).decode()
        
        authorization = f'api_key="{api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature}"'
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': authorization,
            'Date': timestamp,
            'Host': 'spark-api.xf-yun.com'
        }
        
        data = {
            "model": self.llm_config['model'] or "generalv3",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.1,
            "max_tokens": 300
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            # Fallback: return a simple HTTP API call
            print(f"⚠️  Spark WebSocket API failed, falling back to simple format: {e}")
            return f"Failed to expand {prompt.split('abbreviation: ')[-1]} via Spark API"
    
    def _get_context_specific_examples(self):
        """🎯 Generate context-specific examples"""
        base_examples = [
            "NK → Natural killer cell",
            "DC → Dendritic cell", 
            "Treg → Regulatory T cell",
            "CD4+ → CD4-positive T cell",
            "Th1 → T helper 1 cell",
            "CTL → Cytotoxic T lymphocyte"
        ]
        
        # Add tissue-specific examples
        tissue_examples = {
            "immune": [
                "NK → Natural killer cell",
                "DC → Dendritic cell",
                "Treg → Regulatory T cell", 
                "pDC → Plasmacytoid dendritic cell",
                "Th17 → T helper 17 cell",
                "Tfh → T follicular helper cell"
            ],
            "brain": [
                "OPC → Oligodendrocyte precursor cell",
                "OL → Oligodendrocyte",
                "AC → Astrocyte",
                "MG → Microglia",
                "PV+ → Parvalbumin-positive interneuron"
            ],
            "liver": [
                "HSC → Hepatic stellate cell",
                "KC → Kupffer cell",
                "LSEC → Liver sinusoidal endothelial cell",
                "Hep → Hepatocyte"
            ],
            "kidney": [
                "PT → Proximal tubule cell",
                "DT → Distal tubule cell",
                "PC → Principal cell",
                "IC → Intercalated cell",
                "Pod → Podocyte"
            ],
            "lung": [
                "AT1 → Alveolar type 1 cell",
                "AT2 → Alveolar type 2 cell",
                "AM → Alveolar macrophage",
                "Club → Club cell"
            ]
        }
        
        # Species-specific examples
        species_examples = {
            "mouse": [
                "mDC → mouse Dendritic cell",
                "mTreg → mouse Regulatory T cell"
            ],
            "rat": [
                "rNK → rat Natural killer cell"
            ]
        }
        
        # Study context examples
        study_examples = {
            "cancer": [
                "CAF → Cancer-associated fibroblast",
                "TAM → Tumor-associated macrophage",
                "Tex → Exhausted T cell",
                "TIL → Tumor-infiltrating lymphocyte"
            ],
            "development": [
                "PSC → Pluripotent stem cell",
                "NPC → Neural progenitor cell",
                "HSC → Hematopoietic stem cell"
            ]
        }
        
        # Select most relevant examples
        selected_examples = base_examples.copy()
        
        if self.tissue_context:
            for tissue_key in tissue_examples:
                if tissue_key.lower() in str(self.tissue_context).lower():
                    selected_examples.extend(tissue_examples[tissue_key][:3])
                    break
        
        if self.species and self.species in species_examples:
            selected_examples.extend(species_examples[self.species])
        
        if self.study_context:
            for study_key in study_examples:
                if study_key.lower() in self.study_context.lower():
                    selected_examples.extend(study_examples[study_key][:3])
                    break
        
        # Remove duplicates and limit quantity
        unique_examples = list(dict.fromkeys(selected_examples))[:8]
        
        return "Common examples:\n" + "\n".join(f"- {ex}" for ex in unique_examples)
    
    def expand_abbreviations(self, cell_names, force_expand=False, save_cache=True, 
                           tissue_context=None, species=None, study_context=None):
        """
        🔄 Expand cell type abbreviations
        
        Parameters
        ----------
        cell_names : list
            📝 List of cell names
        force_expand : bool
            🔒 Whether to force expand all names (not just abbreviations)
        save_cache : bool
            💾 Whether to save to cache
        tissue_context : str or list, optional
            🧬 Temporary override tissue context information
        species : str, optional
            🐭 Temporary override species information
        study_context : str, optional
            🔬 Temporary override study context information
        
        Returns
        -------
        expanded_names : dict
            📋 Mapping from original names to expanded names
        """
        if self.llm_client is None:
            print("✗ Please setup LLM API first using setup_llm_expansion()")
            return {name: name for name in cell_names}
        
        # Temporarily save original context
        original_tissue = self.tissue_context
        original_species = self.species
        original_study = self.study_context
        
        # Update with temporary context if provided
        if tissue_context is not None:
            self.tissue_context = tissue_context
        if species is not None:
            self.species = species
        if study_context is not None:
            self.study_context = study_context
        
        try:
            expanded_names = {}
            to_expand = []
            
            print("🔍 Analyzing cell names...")
            if self.tissue_context:
                print(f"🧬 Using tissue context: {self.tissue_context}")
            if self.study_context:
                print(f"🔬 Using study context: {self.study_context}")
            print(f"🐭 Species: {self.species}")
            
            for cell_name in cell_names:
                # Check cache (with context-aware cache key)
                cache_key = self._get_cache_key(cell_name)
                if cache_key in self.abbreviation_cache:
                    expanded_names[cell_name] = self.abbreviation_cache[cache_key]['primary_expansion']
                    continue
                
                # Determine if expansion is needed
                if force_expand or self._is_likely_abbreviation(cell_name):
                    to_expand.append(cell_name)
                    print(f"  🔤 Identified potential abbreviation: {cell_name}")
                else:
                    expanded_names[cell_name] = cell_name
            
            if not to_expand:
                print("✓ No abbreviations found to expand")
                return expanded_names
            
            print(f"\n🤖 Expanding {len(to_expand)} abbreviations using LLM...")
            
            for i, cell_name in enumerate(to_expand):
                print(f"  📝 [{i+1}/{len(to_expand)}] Expanding: {cell_name}")
                
                result = self._call_llm_for_expansion(cell_name)
                
                if result and 'primary_expansion' in result:
                    expansion = result['primary_expansion']
                    expanded_names[cell_name] = expansion
                    
                    # Save to cache (using context-aware cache key)
                    cache_key = self._get_cache_key(cell_name)
                    self.abbreviation_cache[cache_key] = result
                    
                    print(f"    ✓ → {expansion} (Confidence: {result.get('confidence', 'unknown')})")
                    
                    if result.get('alternatives'):
                        print(f"    💡 Alternatives: {', '.join(result['alternatives'])}")
                else:
                    expanded_names[cell_name] = cell_name
                    print(f"    ✗ → Expansion failed, keeping original")
            
            if save_cache:
                self._save_abbreviation_cache()
            
            return expanded_names
            
        finally:
            # Restore original context
            self.tissue_context = original_tissue
            self.species = original_species
            self.study_context = original_study
    
    def _get_cache_key(self, cell_name):
        """🔑 Generate cache key with context information"""
        context_parts = [cell_name]
        
        if self.tissue_context:
            if isinstance(self.tissue_context, list):
                context_parts.append(f"tissue:{','.join(self.tissue_context)}")
            else:
                context_parts.append(f"tissue:{self.tissue_context}")
        
        if self.species and self.species != "human":
            context_parts.append(f"species:{self.species}")
        
        if self.study_context:
            context_parts.append(f"study:{self.study_context}")
        
        return "|".join(context_parts)
    
    def map_cells_with_expansion(self, cell_names, threshold=0.5, expand_abbreviations=True,
                           tissue_context=None, species=None, study_context=None,
                           use_llm_selection=True, llm_candidates_count=10):
        """
        🔄 First expand abbreviations, then perform ontology mapping with optional LLM selection
        
        Parameters
        ----------
        cell_names : list
            📝 List of cell names to map
        threshold : float
            📊 Similarity threshold
        expand_abbreviations : bool
            🔄 Whether to enable abbreviation expansion
        tissue_context : str or list, optional
            🧬 Tissue context information
        species : str, optional
            🐭 Species information
        study_context : str, optional
            🔬 Study context information
        use_llm_selection : bool
            🤖 Whether to use LLM for selecting from top candidates (when LLM is available)
        llm_candidates_count : int
            📊 Number of top candidates to send to LLM for selection
        
        Returns
        -------
        mapping_results : dict
            📋 Mapping results (including original and expanded name information + LLM selection)
        """
        if expand_abbreviations and self.llm_client is not None:
            print("📝 Step 1: Expanding abbreviations")
            expanded_names = self.expand_abbreviations(
                cell_names, 
                tissue_context=tissue_context,
                species=species, 
                study_context=study_context
            )
            
            print("\n🎯 Step 2: Performing ontology mapping with LLM-enhanced selection")
            expanded_cell_names = list(expanded_names.values())
            base_results = self.map_cells(expanded_cell_names, threshold, 
                                        use_llm_selection=use_llm_selection,
                                        llm_candidates_count=llm_candidates_count)
            
            # Reorganize results with original name information and enhanced with LLM selection
            mapping_results = {}
            for original_name in cell_names:
                expanded_name = expanded_names[original_name]
                if expanded_name in base_results:
                    result = base_results[expanded_name].copy()
                    result['original_name'] = original_name
                    result['expanded_name'] = expanded_name
                    result['was_expanded'] = (original_name != expanded_name)
                    
                    # For LLM selection, also pass the expanded name for better context
                    if use_llm_selection and result.get('selection_method') == 'llm_selection':
                        # Re-call LLM with both original and expanded names for better context
                        if self.llm_client is not None and original_name != expanded_name:
                            # Get top candidates again for this specific call
                            if self.ontology_embeddings is not None:
                                self._load_model()
                                # Encode the expanded name
                                cell_embedding = self.model.encode([expanded_name])
                                ontology_emb_matrix = np.array([
                                    self.ontology_embeddings[label] for label in self.ontology_labels
                                ])
                                similarities = cosine_similarity(cell_embedding, ontology_emb_matrix)[0]
                                
                                # Get top candidates
                                top_indices = np.argsort(similarities)[-llm_candidates_count:][::-1]
                                top_candidates = []
                                for idx in top_indices:
                                    match_label = self.ontology_labels[idx]
                                    match_similarity = similarities[idx]
                                    top_candidates.append({
                                        'label': match_label,
                                        'similarity': match_similarity,
                                        'index': idx
                                    })
                                
                                # Call LLM with enhanced context (original + expanded)
                                llm_result = self._call_llm_for_selection(original_name, top_candidates, expanded_name)
                                
                                if llm_result and 'selected_rank' in llm_result:
                                    selected_rank = llm_result['selected_rank']
                                    selected_idx = selected_rank - 1
                                    
                                    if 0 <= selected_idx < len(top_candidates):
                                        # Update with new LLM selection that considered both names
                                        selected_candidate = top_candidates[selected_idx]
                                        result['best_match'] = selected_candidate['label']
                                        result['similarity'] = selected_candidate['similarity']
                                        result['llm_reasoning'] = llm_result.get('reasoning', '')
                                        result['llm_confidence'] = llm_result.get('confidence', 'unknown')
                                        result['selection_method'] = 'llm_selection_with_expansion'
                                        
                                        # Update ontology info
                                        ontology_info = self._get_ontology_id(selected_candidate['label'])
                                        result['ontology_id'] = ontology_info['ontology_id']
                                        result['cl_id'] = ontology_info['cl_id']
                    
                    mapping_results[original_name] = result
                else:
                    # This shouldn't happen, but as backup
                    mapping_results[original_name] = {
                        'best_match': 'Unknown',
                        'similarity': 0.0,
                        'confidence': 'low',
                        'original_name': original_name,
                        'expanded_name': expanded_name,
                        'was_expanded': (original_name != expanded_name),
                        'top3_matches': [],
                        'selection_method': 'failed',
                        'llm_reasoning': None,
                        'llm_confidence': None
                    }
        else:
            if expand_abbreviations and self.llm_client is None:
                print("⚠️  Abbreviation expansion requested but LLM client not configured")
                print("💡 To enable LLM expansion, use: mapper.setup_llm_expansion()")
                print("🎯 Performing direct ontology mapping with LLM selection")
            elif not expand_abbreviations:
                print("🎯 Performing direct ontology mapping with LLM selection (abbreviation expansion disabled by parameter)")
            else:
                print("🎯 Performing direct ontology mapping with LLM selection")
                
            mapping_results = self.map_cells(cell_names, threshold, 
                                        use_llm_selection=use_llm_selection,
                                        llm_candidates_count=llm_candidates_count)
            
            # Add expansion information
            for cell_name in mapping_results:
                mapping_results[cell_name]['original_name'] = cell_name
                mapping_results[cell_name]['expanded_name'] = cell_name
                mapping_results[cell_name]['was_expanded'] = False
        
        return mapping_results 
    
    def show_expansion_summary(self, mapping_results):
        """📊 Show abbreviation expansion summary"""
        expanded_items = [
            (name, result) for name, result in mapping_results.items() 
            if result.get('was_expanded', False)
        ]
        
        if not expanded_items:
            print("ℹ️  No abbreviation expansions performed")
            return
        
        print(f"\n📋 Abbreviation Expansion Summary ({len(expanded_items)} items)")
        print("=" * 60)
        for name, result in expanded_items:
            print(f"🔤 {name} → {result['expanded_name']}")
            print(f"  🎯 Mapped to: {result['best_match']} (Similarity: {result['similarity']:.3f})")
            if name in self.abbreviation_cache:
                cache_info = self.abbreviation_cache[name]
                if cache_info.get('confidence'):
                    print(f"  📊 Expansion confidence: {cache_info['confidence']}")
            print()
    
    def clear_abbreviation_cache(self):
        """🗑️  Clear abbreviation cache"""
        self.abbreviation_cache = {}
        if self.cache_file and os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print("✓ Abbreviation cache cleared")
    
    def _check_network_connection(self, timeout=5):
        """
        🌐 Check network connectivity
        
        Parameters
        ----------
        timeout : int
            ⏱️ Connection timeout in seconds
            
        Returns
        -------
        bool
            ✓ True if network is available, False otherwise
        """
        import socket
        try:
            # Try to connect to Google DNS
            socket.create_connection(("8.8.8.8", 53), timeout)
            return True
        except OSError:
            try:
                # Try to connect to Baidu (for China users)
                socket.create_connection(("baidu.com", 80), timeout)
                return True
            except OSError:
                return False
    
    def _setup_hf_mirror(self):
        """
        🪞 Setup HF-Mirror environment
        """
        import os
        # Set HF-Mirror endpoint
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print("🇨🇳 Using HF-Mirror (hf-mirror.com) for faster downloads in China")
    
    def set_model(self, model_name, local_model_dir=None):
        """
        🎯 Set model name and local save directory
        
        Parameters
        ----------
        model_name : str
            🤖 Model name (e.g., 'all-mpnet-base-v2', 'sentence-transformers/all-MiniLM-L6-v2')
        local_model_dir : str, optional
            📁 Local directory to save the model (avoid default cache)
        """
        self.model_name = model_name
        self.local_model_dir = local_model_dir
        self.model = None  # Reset model to trigger reload
        
        print(f"🎯 Model set to: {model_name}")
        if local_model_dir:
            print(f"📁 Local save directory: {local_model_dir}")
        print("💡 Model will be downloaded when first used")
    
    def set_local_model(self, model_path):
        """
        🏠 Set local model path
        
        Parameters
        ----------
        model_path : str
            📁 Local model directory path
        """
        if not os.path.exists(model_path):
            raise ValueError(f"✗ Model path does not exist: {model_path}")
        
        self.model_name = model_path
        self.model = None  # Reset model to trigger reload
        print(f"✓ Local model path set to: {model_path}")
        print("💡 Model will be loaded when first used")
    
    def _load_model(self):
        """🤖 Lazy load sentence transformer model with network detection and HF-Mirror support"""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            import os
            
            print(f"🔄 Loading model {self.model_name}...")
            
            try:
                # 1. 检查是否是本地路径
                if os.path.exists(self.model_name):
                    print(f"📁 Loading local model from: {self.model_name}")
                    self.model = SentenceTransformer(self.model_name)
                    print(f"✓ Local model loaded successfully!")
                    return
                
                # 2. 检查网络连接
                print("🌐 Checking network connectivity...")
                has_network = self._check_network_connection()
                
                if not has_network:
                    raise ConnectionError("✗ No network connection available")
                
                print("✓ Network connection available")
                
                # 3. 设置 HF-Mirror 加速下载
                self._setup_hf_mirror()
                
                # 4. 准备下载参数
                download_kwargs = {}
                
                # 如果指定了本地保存目录，设置 cache_folder
                if self.local_model_dir:
                    # 确保目录存在
                    os.makedirs(self.local_model_dir, exist_ok=True)
                    download_kwargs['cache_folder'] = self.local_model_dir
                    print(f"📁 Models will be saved to: {self.local_model_dir}")
                
                # 5. 尝试从 HF-Mirror 下载模型
                print(f"🪞 Downloading model from HF-Mirror: {self.model_name}")
                
                self.model = SentenceTransformer(
                    self.model_name, 
                    **download_kwargs
                )
                
                print(f"✓ Model loaded successfully from HF-Mirror!")
                
                # 如果指定了本地目录，显示实际保存位置
                if self.local_model_dir:
                    model_path = os.path.join(self.local_model_dir, f"models--sentence-transformers--{self.model_name.replace('/', '--')}")
                    if os.path.exists(model_path):
                        print(f"💾 Model cached at: {model_path}")
                
            except Exception as e:
                print(f"✗ Failed to load model from HF-Mirror: {e}")
                
                # 6. 回退到官方 HuggingFace Hub
                print("🔄 Falling back to official HuggingFace Hub...")
                try:
                    # 移除 HF-Mirror 设置
                    if 'HF_ENDPOINT' in os.environ:
                        del os.environ['HF_ENDPOINT']
                    
                    download_kwargs = {}
                    if self.local_model_dir:
                        download_kwargs['cache_folder'] = self.local_model_dir
                    
                    self.model = SentenceTransformer(
                        self.model_name,
                        **download_kwargs
                    )
                    
                    print(f"✓ Model loaded successfully from official HuggingFace!")
                    
                except Exception as e2:
                    print(f"✗ Failed to load model from official source: {e2}")
                    print(f"💡 Please check:")
                    print(f"   - Model name is correct: {self.model_name}")
                    print(f"   - Network connection is stable")
                    print(f"   - Sufficient disk space available")
                    if self.local_model_dir:
                        print(f"   - Directory permissions for: {self.local_model_dir}")
                    raise
    
    def create_ontology_resources(self, cl_obo_file, save_embeddings=True):
        """
        🔨 Create ontology resources from OBO file
        
        Parameters
        ----------
        cl_obo_file : str
            📄 Cell Ontology OBO file path
        save_embeddings : bool
            💾 Whether to save embeddings to file
        """
        self.output_path = Path(cl_obo_file).parent
        
        print("📖 Parsing ontology file...")
        with open(cl_obo_file, encoding='utf-8') as f:
            graph = json.load(f)["graphs"][0]
        
        # Build ontology dictionary
        self.popv_dict = {}
        self.popv_dict["nodes"] = [
            entry for entry in graph["nodes"] 
            if entry["type"] == "CLASS" and entry.get("lbl", False)
        ]
        
        self.popv_dict["lbl_sentence"] = {
            entry["lbl"]: f"{entry['lbl']}: {entry.get('meta', {}).get('definition', {}).get('val', '')} {' '.join(entry.get('meta', {}).get('comments', []))}"
            for entry in self.popv_dict["nodes"]
        }
        
        self.popv_dict["id_2_lbl"] = {entry["id"]: entry["lbl"] for entry in self.popv_dict["nodes"]}
        self.popv_dict["lbl_2_id"] = {entry["lbl"]: entry["id"] for entry in self.popv_dict["nodes"]}
        
        self.popv_dict["edges"] = [
            i for i in graph["edges"]
            if i["sub"].split("/")[-1][0:2] == "CL" and i["obj"].split("/")[-1][0:2] == "CL" and i["pred"] == "is_a"
        ]
        
        self.popv_dict["ct_edges"] = [
            [self.popv_dict["id_2_lbl"][i["sub"]], self.popv_dict["id_2_lbl"][i["obj"]]] 
            for i in self.popv_dict["edges"]
        ]
        
        # Create embeddings
        print("🧠 Creating NLP embeddings...")
        self._create_embeddings()
        
        # Save resources
        if save_embeddings:
            self.save_embeddings()
        
        self._save_ontology_files()
        print("✓ Ontology resources creation completed!")
    
    def _create_embeddings(self):
        """🧠 Create ontology embeddings"""
        self._load_model()
        
        sentences = list(self.popv_dict["lbl_sentence"].values())
        labels = list(self.popv_dict["lbl_sentence"].keys())
        
        print(f"🔄 Encoding {len(sentences)} ontology labels...")
        sentence_embeddings = self.model.encode(sentences, show_progress_bar=True)
        
        self.ontology_embeddings = {}
        for label, embedding in zip(labels, sentence_embeddings):
            self.ontology_embeddings[label] = embedding
        
        self.ontology_labels = labels
    
    def save_embeddings(self, output_path=None):
        """💾 Save embeddings to file"""
        if output_path is None:
            output_path = self.output_path
        
        save_data = {
            'embeddings': self.ontology_embeddings,
            'labels': self.ontology_labels,
            'model_name': self.model_name,
            'popv_dict': getattr(self, 'popv_dict', None)  # Include popv_dict for ontology IDs
        }
        
        embeddings_file = os.path.join(output_path, "ontology_embeddings.pkl")
        with open(embeddings_file, "wb") as f:
            pickle.dump(save_data, f)
        
        print(f"💾 Embeddings saved to: {embeddings_file}")
        if save_data['popv_dict'] is not None:
            print(f"📋 Ontology mappings included: {len(save_data['popv_dict'].get('lbl_2_id', {}))} cell types")
    
    def load_embeddings(self, embeddings_path):
        """📥 Load embeddings from file"""
        with open(embeddings_path, "rb") as f:
            save_data = pickle.load(f)
        
        self.ontology_embeddings = save_data['embeddings']
        self.ontology_labels = save_data['labels']
        self.model_name = save_data.get('model_name', self.model_name)
        
        # Load popv_dict if available (for ontology IDs)
        if 'popv_dict' in save_data and save_data['popv_dict'] is not None:
            self.popv_dict = save_data['popv_dict']
            print(f"📥 Loaded embeddings for {len(self.ontology_labels)} ontology labels")
            print(f"📋 Ontology mappings loaded: {len(self.popv_dict.get('lbl_2_id', {}))} cell types")
        else:
            self.popv_dict = None
            print(f"📥 Loaded embeddings for {len(self.ontology_labels)} ontology labels")
            print("⚠️  No ontology ID mappings found in file (ontology_id will be None)")
            print("💡 Use create_ontology_resources() to generate complete ontology data with IDs")
    
    def load_ontology_mappings(self, popv_json_path):
        """
        📋 Load ontology ID mappings from cl_popv.json file
        
        Parameters
        ----------
        popv_json_path : str
            📄 Path to cl_popv.json file
        
        Returns
        -------
        success : bool
            ✓ True if loaded successfully
        """
        try:
            with open(popv_json_path, 'r', encoding='utf-8') as f:
                self.popv_dict = json.load(f)
            
            mapping_count = len(self.popv_dict.get('lbl_2_id', {}))
            print(f"✓ Loaded ontology mappings: {mapping_count} cell types")
            print("📋 Ontology IDs will now be available in mapping results")
            return True
            
        except FileNotFoundError:
            print(f"✗ File not found: {popv_json_path}")
            print("💡 Make sure the cl_popv.json file exists")
            return False
        except json.JSONDecodeError as e:
            print(f"✗ JSON decode error: {e}")
            return False
        except Exception as e:
            print(f"✗ Failed to load ontology mappings: {e}")
            return False
    
    def check_ontology_status(self):
        """
        🔍 Check ontology data status and provide diagnostic information
        
        Returns
        -------
        status : dict
            📊 Status information
        """
        status = {
            'embeddings_loaded': self.ontology_embeddings is not None,
            'labels_count': len(self.ontology_labels) if self.ontology_labels else 0,
            'popv_dict_loaded': self.popv_dict is not None,
            'ontology_mappings_count': 0,
            'can_provide_ontology_ids': False
        }
        
        if self.popv_dict and 'lbl_2_id' in self.popv_dict:
            status['ontology_mappings_count'] = len(self.popv_dict['lbl_2_id'])
            status['can_provide_ontology_ids'] = True
        
        print("🔍 === Ontology Status Diagnostic ===")
        print(f"📊 Embeddings loaded: {'✓' if status['embeddings_loaded'] else '✗'}")
        print(f"📝 Ontology labels: {status['labels_count']}")
        print(f"📋 Ontology mappings loaded: {'✓' if status['popv_dict_loaded'] else '✗'}")
        print(f"🆔 Ontology ID mappings: {status['ontology_mappings_count']}")
        print(f"🎯 Can provide ontology IDs: {'✓' if status['can_provide_ontology_ids'] else '✗'}")
        
        if not status['can_provide_ontology_ids']:
            print("\n💡 === Solutions to get Ontology IDs ===")
            print("1. 🔨 Create complete ontology resources:")
            print("   mapper.create_ontology_resources('cl.json')")
            print("2. 📋 Load ontology mappings separately:")
            print("   mapper.load_ontology_mappings('cl_popv.json')")
            print("3. 🔄 Re-save embeddings to include mappings:")
            print("   # After loading mappings, re-save embeddings")
            print("   mapper.save_embeddings()")
        
        return status
    
    def _save_ontology_files(self):
        """💾 Save other ontology files"""
        if self.output_path is None:
            return
        
        # Save JSON file
        with open(f"{self.output_path}/cl_popv.json", "w", encoding='utf-8') as f:
            json.dump(self.popv_dict, f, indent=4, ensure_ascii=False)
        
        # Save edge information
        children_edge_celltype_df = pd.DataFrame(self.popv_dict["ct_edges"])
        children_edge_celltype_df.to_csv(
            f"{self.output_path}/cl.ontology", 
            sep="\t", header=False, index=False
        )
        
        # Save text format embeddings
        output_file = os.path.join(self.output_path, "cl.ontology.nlp.emb")
        with open(output_file, "w", encoding='utf-8') as fout:
            for label, vec in self.ontology_embeddings.items():
                fout.write(label + "\t" + "\t".join(map(str, vec)) + "\n")
    
    def _call_llm_for_selection(self, cell_name, top_candidates, expanded_name=None):
        """
        🤖 Call LLM to select the best cell type from top candidates
        
        Parameters
        ----------
        cell_name : str
            📝 Original cell name
        top_candidates : list
            📋 List of candidate dictionaries with 'label' and 'similarity' keys
        expanded_name : str, optional
            🔤 Expanded cell name if abbreviation expansion was performed
        
        Returns
        -------
        selection_result : dict or None
            📋 LLM selection result with selected candidate and reasoning
        """
        if self.llm_client is None:
            return None
        
        # Build context information
        context_parts = []
        
        if self.species and self.species != "human":
            context_parts.append(f"Species: {self.species}")
        
        if self.tissue_context:
            if isinstance(self.tissue_context, list):
                tissue_info = ", ".join(self.tissue_context)
            else:
                tissue_info = self.tissue_context
            context_parts.append(f"Tissue/Organ context: {tissue_info}")
        
        if self.study_context:
            context_parts.append(f"Study context: {self.study_context}")
        
        context_str = "\n".join(context_parts) if context_parts else ""
        
        # Prepare candidate list for the prompt
        candidates_text = []
        for i, candidate in enumerate(top_candidates, 1):
            candidates_text.append(f"{i}. {candidate['label']} (similarity: {candidate['similarity']:.3f})")
        candidates_str = "\n".join(candidates_text)
        
        # Build the prompt
        cell_info = f"Original cell name: '{cell_name}'"
        if expanded_name and expanded_name != cell_name:
            cell_info += f"\nExpanded cell name: '{expanded_name}'"
        
        prompt = f"""You are an expert in cell biology and immunology. Your task is to select the most appropriate cell type from a list of candidates based on biological context.

{context_str}

{cell_info}

Here are the top candidate cell types ranked by semantic similarity:

{candidates_str}

Based on your expertise in cell biology and the provided context information, please select the most biologically appropriate cell type from the candidates above.

Consider the following factors:
1. Biological accuracy and cell type definitions
2. Tissue/organ context compatibility 
3. Species-specific cell type nomenclature
4. Study context relevance (e.g., cancer, development, aging)
5. Standard cell ontology terminology

Please respond in JSON format:
{{
    "selected_rank": 1,
    "selected_label": "selected cell type name",
    "confidence": "high/medium/low",
    "reasoning": "brief biological explanation for your selection"
}}

Please provide only the JSON response."""

        # Initialize content variable
        content = None
        api_type = self.llm_config.get('api_type', 'unknown')
        
        try:
            if api_type in ["openai", "custom_openai", "doubao"]:
                # OpenAI API and compatible APIs
                response = self.llm_client.chat.completions.create(
                    model=self.llm_config['model'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=400,
                    **self.llm_config.get('extra_params', {})
                )
                content = response.choices[0].message.content
                
            elif api_type == "anthropic":
                response = self.llm_client.messages.create(
                    model=self.llm_config['model'],
                    max_tokens=400,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}],
                    **self.llm_config.get('extra_params', {})
                )
                content = response.content[0].text
                
            elif api_type == "ollama":
                import requests
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.llm_config['model'],
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1, **self.llm_config.get('extra_params', {})}
                    },
                    timeout=30
                )
                response.raise_for_status()
                content = response.json().get('response', '')
                
            elif api_type == "qwen":
                # 阿里云通义千问
                import dashscope
                from dashscope import Generation
                response = dashscope.Generation.call(
                    model=self.llm_config['model'] or 'qwen-turbo',
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.1,
                    max_tokens=400,
                    **self.llm_config.get('extra_params', {})
                )
                if response.status_code == 200:
                    content = response.output.text
                else:
                    raise Exception(f"Qwen API error: {response.message}")
                    
            elif api_type == "ernie":
                # 百度文心一言
                import ernie
                response = ernie.ChatCompletion.create(
                    model=self.llm_config['model'] or 'ernie-bot',
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.1,
                    **self.llm_config.get('extra_params', {})
                )
                content = response.get_result()
                
            elif api_type == "glm":
                # 智谱AI GLM
                import zhipuai
                response = zhipuai.model_api.invoke(
                    model=self.llm_config['model'] or 'chatglm_turbo',
                    prompt=[{'role': 'user', 'content': prompt}],
                    temperature=0.1,
                    **self.llm_config.get('extra_params', {})
                )
                if response['code'] == 200:
                    content = response['data']['choices'][0]['content']
                else:
                    raise Exception(f"GLM API error: {response.get('msg', 'Unknown error')}")
                    
            elif api_type == "spark":
                # 讯飞星火 (WebSocket API)
                content = self._call_spark_api(prompt)
                
            else:
                raise ValueError(f"Unsupported API type: {api_type}")
            
        except Exception as e:
            print(f"✗ LLM selection call failed ({api_type}): {e}")
            return None
        
        # Check if content was successfully retrieved
        if content is None:
            print(f"✗ No content received from {api_type} API for selection")
            return None
        
        # Parse JSON response
        try:
            # Extract JSON part
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # Validate the result
                if 'selected_rank' in result and 'selected_label' in result:
                    # Ensure selected_rank is within valid range
                    selected_rank = int(result['selected_rank'])
                    if 1 <= selected_rank <= len(top_candidates):
                        return result
                    else:
                        print(f"⚠️  LLM selected invalid rank {selected_rank}, falling back to top similarity")
                        return None
                else:
                    print(f"⚠️  LLM response missing required fields, falling back to top similarity")
                    return None
            else:
                print(f"⚠️  LLM response not in JSON format, falling back to top similarity")
                return None
                
        except json.JSONDecodeError:
            print(f"⚠️  Failed to parse LLM selection response, falling back to top similarity")
            return None
        except ValueError:
            print(f"⚠️  Invalid selected_rank in LLM response, falling back to top similarity")
            return None

    def map_cells(self, cell_names, threshold=0.5, use_llm_selection=False, llm_candidates_count=10):
        """
        🎯 Map cell names to ontology with optional LLM-enhanced selection
        
        Parameters
        ----------
        cell_names : list
            📝 List of cell names to map
        threshold : float
            📊 Similarity threshold
        use_llm_selection : bool
            🤖 Whether to use LLM for selecting from top candidates (when LLM is available)
        llm_candidates_count : int
            📊 Number of top candidates to send to LLM for selection
        
        Returns
        -------
        mapping_results : dict
            📋 Mapping results (now includes ontology IDs and LLM selection info)
        """
        if self.ontology_embeddings is None:
            raise ValueError("✗ Please load or create ontology embeddings first")
        
        self._load_model()
        
        print(f"🎯 Mapping {len(cell_names)} cell names...")
        if use_llm_selection and self.llm_client is not None:
            print(f"🤖 LLM-enhanced selection enabled (from top {llm_candidates_count} candidates)")
        
        # Encode cell names
        cell_embeddings = self.model.encode(cell_names, show_progress_bar=True)
        
        # Get ontology embedding matrix
        ontology_emb_matrix = np.array([
            self.ontology_embeddings[label] for label in self.ontology_labels
        ])
        
        # Calculate similarities
        similarities = cosine_similarity(cell_embeddings, ontology_emb_matrix)
        
        mapping_results = {}
        llm_selection_count = 0
        
        for i, cell_name in enumerate(cell_names):
            # Get top candidates for potential LLM selection
            top_indices = np.argsort(similarities[i])[-llm_candidates_count:][::-1]
            top_candidates = []
            for idx in top_indices:
                match_label = self.ontology_labels[idx]
                match_similarity = similarities[i][idx]
                top_candidates.append({
                    'label': match_label,
                    'similarity': match_similarity,
                    'index': idx
                })
            
            # Default to top similarity match
            best_match_idx = top_indices[0]  # Highest similarity
            best_similarity = similarities[i][best_match_idx]
            best_match_label = self.ontology_labels[best_match_idx]
            selection_method = "cosine_similarity"
            llm_reasoning = None
            llm_confidence = None
            
            # Try LLM selection if enabled and available
            if use_llm_selection and self.llm_client is not None:
                print(f"  🤖 [{i+1}/{len(cell_names)}] LLM selecting for: {cell_name}")
                
                # Call LLM for selection
                llm_result = self._call_llm_for_selection(cell_name, top_candidates)
                
                if llm_result and 'selected_rank' in llm_result:
                    selected_rank = llm_result['selected_rank']
                    # Convert from 1-indexed to 0-indexed
                    selected_idx = selected_rank - 1
                    
                    if 0 <= selected_idx < len(top_candidates):
                        # Use LLM selection
                        selected_candidate = top_candidates[selected_idx]
                        best_match_idx = selected_candidate['index']
                        best_similarity = selected_candidate['similarity']
                        best_match_label = selected_candidate['label']
                        selection_method = "llm_selection"
                        llm_reasoning = llm_result.get('reasoning', '')
                        llm_confidence = llm_result.get('confidence', 'unknown')
                        llm_selection_count += 1
                        
                        print(f"    ✓ LLM selected: {best_match_label} (rank {selected_rank}, sim: {best_similarity:.3f})")
                        if llm_reasoning:
                            print(f"    💡 Reasoning: {llm_reasoning[:100]}...")
                    else:
                        print(f"    ⚠️  LLM selected invalid rank, using top similarity")
                else:
                    print(f"    ✗ LLM selection failed, using top similarity")
            
            # Get ontology ID information for best match
            ontology_info = self._get_ontology_id(best_match_label)
            
            # Get top 3 best matches with their IDs (for compatibility)
            top3_indices = np.argsort(similarities[i])[-3:][::-1]
            top3_matches = []
            for idx in top3_indices:
                match_label = self.ontology_labels[idx]
                match_similarity = similarities[i][idx]
                match_ontology_info = self._get_ontology_id(match_label)
                top3_matches.append({
                    'label': match_label,
                    'similarity': match_similarity,
                    'ontology_id': match_ontology_info['ontology_id'],
                    'cl_id': match_ontology_info['cl_id']
                })
            
            mapping_results[cell_name] = {
                'best_match': best_match_label,
                'similarity': best_similarity,
                'confidence': 'high' if best_similarity > threshold else 'low',
                'ontology_id': ontology_info['ontology_id'],
                'cl_id': ontology_info['cl_id'],
                'top3_matches': top3_matches,
                'selection_method': selection_method,
                'llm_reasoning': llm_reasoning,
                'llm_confidence': llm_confidence,
                'top_candidates_count': len(top_candidates)
            }
        
        if use_llm_selection and self.llm_client is not None:
            print(f"✓ LLM selection summary: {llm_selection_count}/{len(cell_names)} cells selected by LLM")
        
        return mapping_results 
    
    def map_adata(self, adata, cell_name_col=None, threshold=0.5, new_col_name='cell_ontology'):
        """
        🧬 Map cell names in AnnData object to ontology
        
        Parameters
        ----------
        adata : AnnData
            📊 Single-cell data object
        cell_name_col : str, optional
            📝 Column name containing cell names, use index if None
        threshold : float
            📊 Similarity threshold
        new_col_name : str
            🏷️  New column name
        
        Returns
        -------
        mapping_results : dict
            📋 Mapping results
        """
        # Get cell names
        if cell_name_col is None:
            cell_names = adata.obs.index.unique().tolist()
            cell_names_series = adata.obs.index.to_series()
            print(f"📊 Using {len(cell_names)} unique cell names from index")
        else:
            cell_names = adata.obs[cell_name_col].unique().tolist()
            cell_names_series = adata.obs[cell_name_col]
            print(f"📊 Using {len(cell_names)} unique cell names from column '{cell_name_col}'")
        
        # Perform mapping
        mapping_results = self.map_cells(cell_names, threshold)
        
        # Apply to adata
        print("📝 Applying mapping results to AnnData...")
        
        def get_best_match(cell_name):
            return mapping_results.get(cell_name, {}).get('best_match', 'Unknown')
        
        def get_similarity(cell_name):
            return mapping_results.get(cell_name, {}).get('similarity', 0.0)
        
        def get_confidence(cell_name):
            return mapping_results.get(cell_name, {}).get('confidence', 'low')
        
        def get_ontology_id(cell_name):
            return mapping_results.get(cell_name, {}).get('ontology_id', None)
        
        def get_cl_id(cell_name):
            return mapping_results.get(cell_name, {}).get('cl_id', None)
        
        def get_expanded_name(cell_name):
            return mapping_results.get(cell_name, {}).get('expanded_name', cell_name)
        
        def was_expanded(cell_name):
            return mapping_results.get(cell_name, {}).get('was_expanded', False)
        
        adata.obs[new_col_name] = cell_names_series.apply(get_best_match)
        adata.obs[f'{new_col_name}_similarity'] = cell_names_series.apply(get_similarity)
        adata.obs[f'{new_col_name}_confidence'] = cell_names_series.apply(get_confidence)
        adata.obs[f'{new_col_name}_ontology_id'] = cell_names_series.apply(get_ontology_id)
        adata.obs[f'{new_col_name}_cl_id'] = cell_names_series.apply(get_cl_id)
        adata.obs[f'{new_col_name}_expanded'] = cell_names_series.apply(get_expanded_name)
        adata.obs[f'{new_col_name}_was_expanded'] = cell_names_series.apply(was_expanded)
        
        # Statistics
        high_conf_count = sum(1 for r in mapping_results.values() if r['confidence'] == 'high')
        print(f"✓ Mapping completed: {high_conf_count}/{len(mapping_results)} cell names have high confidence mapping")
        
        return mapping_results
    
    def get_statistics(self, mapping_results):
        """📊 Get mapping statistics"""
        total = len(mapping_results)
        high_conf = sum(1 for r in mapping_results.values() if r['confidence'] == 'high')
        low_conf = total - high_conf
        
        similarities = [r['similarity'] for r in mapping_results.values()]
        
        stats = {
            'total_mappings': total,
            'high_confidence': high_conf,
            'low_confidence': low_conf,
            'high_confidence_ratio': high_conf / total if total > 0 else 0,
            'mean_similarity': np.mean(similarities),
            'median_similarity': np.median(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities)
        }
        
        return stats
    
    def print_mapping_summary_with_ids(self, mapping_results, top_n=10):
        """📋 Print mapping summary with ontology IDs"""
        stats = self.get_statistics(mapping_results)
        
        print("\n" + "="*60)
        print("MAPPING STATISTICS SUMMARY")
        print("="*60)
        print(f"Total mappings:\t\t{stats['total_mappings']}")
        print(f"High confidence:\t{stats['high_confidence']} ({stats['high_confidence_ratio']:.2%})")
        print(f"Low confidence:\t\t{stats['low_confidence']}")
        print(f"Average similarity:\t{stats['mean_similarity']:.3f}")
        print(f"Median similarity:\t{stats['median_similarity']:.3f}")
        
        print(f"\nTOP {top_n} MAPPING RESULTS")
        print("-" * 60)
        sorted_results = sorted(
            mapping_results.items(), 
            key=lambda x: x[1]['similarity'], 
            reverse=True
        )
        
        for i, (cell_name, result) in enumerate(sorted_results[:top_n]):
            conf_mark = "✓" if result['confidence'] == 'high' else "?"
            cl_id = result.get('cl_id', 'N/A')
            print(f"\n{i+1:2d}. [{conf_mark}] {cell_name}")
            print(f"     → {result['best_match']}")
            print(f"     Similarity: {result['similarity']:.3f}")
            print(f"     CL ID: {cl_id}")
            if result.get('was_expanded', False):
                print(f"     Expanded from: {result.get('expanded_name', cell_name)}")
            print()
    
    def print_mapping_summary(self, mapping_results, top_n=10):
        """📋 Print mapping summary"""
        stats = self.get_statistics(mapping_results)
        
        print("\n" + "="*60)
        print("MAPPING STATISTICS SUMMARY")
        print("="*60)
        print(f"Total mappings:\t\t{stats['total_mappings']}")
        print(f"High confidence:\t{stats['high_confidence']} ({stats['high_confidence_ratio']:.2%})")
        print(f"Low confidence:\t\t{stats['low_confidence']}")
        print(f"Average similarity:\t{stats['mean_similarity']:.3f}")
        print(f"Median similarity:\t{stats['median_similarity']:.3f}")
        
        print(f"\nTOP {top_n} MAPPING RESULTS")
        print("-" * 60)
        sorted_results = sorted(
            mapping_results.items(), 
            key=lambda x: x[1]['similarity'], 
            reverse=True
        )
        
        for i, (cell_name, result) in enumerate(sorted_results[:top_n]):
            conf_mark = "✓" if result['confidence'] == 'high' else "?"
            print(f"{conf_mark} {cell_name} -> {result['best_match']} (Similarity: {result['similarity']:.3f})")
    
    def save_mapping_results(self, mapping_results, output_file):
        """💾 Save mapping results to file"""
        results_data = []
        
        for cell_name, result in mapping_results.items():
            row_data = {
                'cell_name': cell_name,
                'best_match': result['best_match'],
                'similarity': result['similarity'],
                'confidence': result['confidence'],
                'ontology_id': result.get('ontology_id', ''),
                'cl_id': result.get('cl_id', ''),
            }
            
            # Handle top3_matches (new structure with dictionaries)
            top3_matches = result.get('top3_matches', [])
            for i, match in enumerate(top3_matches[:3], 1):
                if isinstance(match, dict):
                    # New structure
                    row_data[f'top{i}_match'] = match.get('label', '')
                    row_data[f'top{i}_similarity'] = match.get('similarity', 0)
                    row_data[f'top{i}_ontology_id'] = match.get('ontology_id', '')
                    row_data[f'top{i}_cl_id'] = match.get('cl_id', '')
                else:
                    # Old structure (tuple)
                    row_data[f'top{i}_match'] = match[0] if len(match) > 0 else ''
                    row_data[f'top{i}_similarity'] = match[1] if len(match) > 1 else 0
                    row_data[f'top{i}_ontology_id'] = ''
                    row_data[f'top{i}_cl_id'] = ''
            
            # Fill missing top matches
            for i in range(len(top3_matches) + 1, 4):
                row_data[f'top{i}_match'] = ''
                row_data[f'top{i}_similarity'] = 0
                row_data[f'top{i}_ontology_id'] = ''
                row_data[f'top{i}_cl_id'] = ''
            
            results_data.append(row_data)
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
        print(f"Columns: {list(results_df.columns)}")
    
    def list_ontology_cells(self, max_display=50, return_all=False):
        """
        📋 List all cell types in the ontology
        
        Parameters
        ----------
        max_display : int
            📊 Maximum number to display
        return_all : bool
            📝 Whether to return complete list
        
        Returns
        -------
        cell_types : list
            📋 List of cell types
        """
        if self.ontology_labels is None:
            raise ValueError("✗ Please load or create ontology embeddings first")
        
        total_count = len(self.ontology_labels)
        print(f"📊 Total {total_count} cell types in ontology")
        
        if max_display > 0:
            print(f"\n📋 First {min(max_display, total_count)} cell types:")
            for i, cell_type in enumerate(self.ontology_labels[:max_display]):
                print(f"{i+1:3d}. {cell_type}")
            
            if total_count > max_display:
                print(f"... {total_count - max_display} more cell types")
                print("💡 Use return_all=True to get complete list")
        
        if return_all:
            return self.ontology_labels.copy()
        else:
            return self.ontology_labels[:max_display]
    
    def search_ontology_cells(self, keyword, case_sensitive=False, max_results=20):
        """
        🔍 Search cell types containing specific keywords in the ontology
        
        Parameters
        ----------
        keyword : str
            🔤 Search keyword
        case_sensitive : bool
            📝 Whether case sensitive
        max_results : int
            📊 Maximum number of results to return
        
        Returns
        -------
        matches : list
            📋 List of matching cell types
        """
        if self.ontology_labels is None:
            raise ValueError("✗ Please load or create ontology embeddings first")
        
        if not case_sensitive:
            keyword = keyword.lower()
            search_labels = [label.lower() for label in self.ontology_labels]
        else:
            search_labels = self.ontology_labels
        
        matches = []
        original_matches = []
        
        for i, label in enumerate(search_labels):
            if keyword in label:
                matches.append(label)
                original_matches.append(self.ontology_labels[i])
        
        print(f"🔍 Found {len(matches)} cell types containing '{keyword}':")
        
        for i, match in enumerate(original_matches[:max_results]):
            print(f"{i+1:3d}. {match}")
        
        if len(matches) > max_results:
            print(f"... {len(matches) - max_results} more results")
        
        return original_matches[:max_results]
    
    def get_cell_info(self, cell_name):
        """
        ℹ️  Get detailed information for specific cell type
        
        Parameters
        ----------
        cell_name : str
            📝 Cell type name
        
        Returns
        -------
        info : dict
            ℹ️  Cell information dictionary
        """
        if self.ontology_labels is None:
            raise ValueError("✗ Please load or create ontology embeddings first")
        
        if cell_name not in self.ontology_labels:
            print(f"✗ Cell type not found: {cell_name}")
            # Try fuzzy matching
            similar = self.search_ontology_cells(cell_name, max_results=5)
            if similar:
                print("💡 Did you mean one of these:")
                for s in similar:
                    print(f"  - {s}")
            return None
        
        info = {'name': cell_name}
        
        # Add more information if popv_dict exists
        if self.popv_dict and 'lbl_sentence' in self.popv_dict:
            if cell_name in self.popv_dict['lbl_sentence']:
                info['description'] = self.popv_dict['lbl_sentence'][cell_name]
            
            if 'lbl_2_id' in self.popv_dict and cell_name in self.popv_dict['lbl_2_id']:
                info['ontology_id'] = self.popv_dict['lbl_2_id'][cell_name]
        
        # Display information
        print(f"\nℹ️  === {cell_name} ===")
        if 'ontology_id' in info:
            print(f"🆔 Ontology ID: {info['ontology_id']}")
        if 'description' in info:
            print(f"📝 Description: {info['description']}")
        
        return info
    
    def browse_ontology_by_category(self, categories=None, max_per_category=10):
        """
        📂 Browse ontology cell types by category
        
        Parameters
        ----------
        categories : list, optional
            📝 List of category keywords to view
        max_per_category : int
            📊 Maximum number to display per category
        """
        if self.ontology_labels is None:
            raise ValueError("✗ Please load or create ontology embeddings first")
        
        if categories is None:
            categories = [
                'T cell', 'B cell', 'NK cell', 'dendritic cell', 'macrophage',
                'neutrophil', 'eosinophil', 'basophil', 'monocyte', 'lymphocyte',
                'epithelial cell', 'endothelial cell', 'fibroblast', 'neuron',
                'stem cell', 'progenitor cell', 'cancer cell', 'tumor cell'
            ]
        
        print("📂 === Browse Ontology Cell Types by Category ===\n")
        
        for category in categories:
            matches = self.search_ontology_cells(category, max_results=max_per_category)
            if matches:
                print(f"\n🏷️  【{category} related】 (Showing top {len(matches)}):")
                for i, match in enumerate(matches):
                    print(f"  {i+1}. {match}")
            print("-" * 50)
    
    def get_ontology_statistics(self):
        """📊 Get ontology statistics"""
        if self.ontology_labels is None:
            raise ValueError("✗ Please load or create ontology embeddings first")
        
        total_cells = len(self.ontology_labels)
        
        # Analyze cell type name length distribution
        lengths = [len(label) for label in self.ontology_labels]
        
        # Count common words
        all_words = []
        for label in self.ontology_labels:
            words = label.lower().split()
            all_words.extend(words)
        
        from collections import Counter
        word_counts = Counter(all_words)
        common_words = word_counts.most_common(10)
        
        stats = {
            'total_cell_types': total_cells,
            'avg_name_length': np.mean(lengths),
            'min_name_length': np.min(lengths),
            'max_name_length': np.max(lengths),
            'common_words': common_words
        }
        
        print("📊 === Ontology Statistics ===")
        print(f"📝 Total cell types: {stats['total_cell_types']}")
        print(f"📏 Average name length: {stats['avg_name_length']:.1f} characters")
        print(f"📏 Shortest name length: {stats['min_name_length']} characters")
        print(f"📏 Longest name length: {stats['max_name_length']} characters")
        print(f"\n🔤 Most common words:")
        for word, count in common_words:
            print(f"  {word}: {count} times")
        
        return stats
    
    def find_similar_cells(self, cell_name, top_k=10):
        """
        🔍 Find ontology cell types most similar to given cell name
        
        Parameters
        ----------
        cell_name : str
            📝 Input cell name
        top_k : int
            📊 Return top k most similar results
        
        Returns
        -------
        similar_cells : list
            📋 Similar cell types and their similarities
        """
        if self.ontology_embeddings is None:
            raise ValueError("✗ Please load or create ontology embeddings first")
        
        self._load_model()
        
        # Encode input cell name
        cell_embedding = self.model.encode([cell_name])
        
        # Get ontology embedding matrix
        ontology_emb_matrix = np.array([
            self.ontology_embeddings[label] for label in self.ontology_labels
        ])
        
        # Calculate similarities
        similarities = cosine_similarity(cell_embedding, ontology_emb_matrix)[0]
        
        # Get top-k most similar
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        similar_cells = [
            (self.ontology_labels[idx], similarities[idx]) 
            for idx in top_indices
        ]
        
        print(f"\n🎯 Ontology cell types most similar to '{cell_name}':")
        for i, (label, sim) in enumerate(similar_cells):
            print(f"{i+1:2d}. {label:<40} (Similarity: {sim:.3f})")
        
        return similar_cells
    
    def download_model(self):
        """
        📥 Manually download and load the model
        
        Returns
        -------
        bool
            ✓ True if successful, False otherwise
        """
        try:
            self._load_model()
            return True
        except Exception as e:
            print(f"✗ Model download failed: {e}")
            return False
    
    def _extract_cl_id(self, ontology_id):
        """
        🆔 Extract CL number from ontology ID
        
        Parameters
        ----------
        ontology_id : str
            📝 Full ontology ID like "http://purl.obolibrary.org/obo/CL_0000084"
            
        Returns
        -------
        cl_id : str
            🔢 CL ID like "CL:0000084" or None if not found
        """
        if not ontology_id:
            return None
            
        try:
            # Extract CL number from URL format
            if "CL_" in ontology_id:
                cl_number = ontology_id.split("CL_")[-1]
                return f"CL:{cl_number}"
            # Handle other formats if needed
            elif "CL:" in ontology_id:
                return ontology_id.split("/")[-1]
            else:
                return None
        except:
            return None
    
    def _get_ontology_id(self, cell_label):
        """
        🔗 Get ontology ID for a cell label
        
        Parameters
        ----------
        cell_label : str
            📝 Cell type label
            
        Returns
        -------
        ontology_info : dict
            📋 Dictionary with ontology_id and cl_id
        """
        if not self.popv_dict or 'lbl_2_id' not in self.popv_dict:
            return {'ontology_id': None, 'cl_id': None}
        
        ontology_id = self.popv_dict['lbl_2_id'].get(cell_label)
        cl_id = self._extract_cl_id(ontology_id)
        
        return {
            'ontology_id': ontology_id,
            'cl_id': cl_id
        }

    def load_cell_taxonomy_resource(self, taxonomy_file, species_filter=None):
        """
        📊 Load Cell Taxonomy resource as additional ontology
        
        This method treats Cell Taxonomy as an additional ontology source,
        reusing the same NLP embedding and matching mechanisms as Cell Ontology.
        
        Parameters
        ----------
        taxonomy_file : str
            📄 Path to Cell_Taxonomy_resource.txt file
        species_filter : str or list, optional
            🐭 Filter by species (e.g., "Homo sapiens", "Mus musculus")
            
        Returns
        -------
        success : bool
            ✓ True if loaded successfully
        """
        try:
            import pandas as pd
            
            print(f"📊 Loading Cell Taxonomy resource from: {taxonomy_file}")
            
            # Load the taxonomy file
            df = pd.read_csv(taxonomy_file, sep='\t', dtype=str, na_values=['NA'])
            print(f"✓ Loaded {len(df)} taxonomy entries")
            
            # Filter by species if specified
            if species_filter:
                if isinstance(species_filter, str):
                    species_filter = [species_filter]
                original_count = len(df)
                df = df[df['Species'].isin(species_filter)]
                print(f"🐭 Filtered by species {species_filter}: {len(df)}/{original_count} entries")
            
            self.taxonomy_resource = df
            
            # Create taxonomy embeddings using the same approach as ontology
            self._create_taxonomy_embeddings()
            
            # Show statistics
            species_counts = df['Species'].value_counts()
            print(f"📈 Species distribution:")
            for species, count in species_counts.head(5).items():
                print(f"  🐭 {species}: {count} entries")
            
            cell_type_count = df['Cell_standard'].nunique()
            print(f"🧬 Unique cell types: {cell_type_count}")
            
            marker_count = df['Cell_Marker'].dropna().nunique()
            print(f"🎯 Unique markers: {marker_count}")
            
            return True
            
        except FileNotFoundError:
            print(f"✗ File not found: {taxonomy_file}")
            return False
        except Exception as e:
            print(f"✗ Failed to load taxonomy resource: {e}")
            return False
    
    def _create_taxonomy_embeddings(self):
        """🧠 Create taxonomy embeddings using the same approach as ontology"""
        if self.taxonomy_resource is None:
            return
        
        self._load_model()
        
        # Get unique cell types and their detailed descriptions
        unique_cell_types = self.taxonomy_resource['Cell_standard'].dropna().unique().tolist()
        
        if not unique_cell_types:
            print("⚠️  No valid cell types found in taxonomy resource")
            return
        
        print(f"🧠 Creating embeddings for {len(unique_cell_types)} taxonomy cell types...")
        
        # Create embeddings for cell type names (same as ontology approach)
        cell_embeddings = self.model.encode(unique_cell_types, show_progress_bar=True)
        
        # Store embeddings
        self.taxonomy_embeddings = {}
        for cell_type, embedding in zip(unique_cell_types, cell_embeddings):
            self.taxonomy_embeddings[cell_type] = embedding
        
        self.taxonomy_labels = unique_cell_types
        
        # Create detailed info dictionary for quick lookup
        self.taxonomy_info_dict = {}
        for _, row in self.taxonomy_resource.iterrows():
            cell_type = row['Cell_standard']
            if pd.notna(cell_type):
                if cell_type not in self.taxonomy_info_dict:
                    self.taxonomy_info_dict[cell_type] = []
                self.taxonomy_info_dict[cell_type].append(row.to_dict())
        
        print(f"✓ Created taxonomy embeddings for {len(self.taxonomy_embeddings)} cell types")
    
    def map_cells_with_taxonomy(self, cell_names, threshold=0.5, expand_abbreviations=True,
                               use_taxonomy=True, species=None, tissue_context=None, study_context=None,
                               use_llm_selection=True, llm_candidates_count=10):
        """
        🔄 Enhanced cell mapping using both ontology and taxonomy
        
        This method combines Cell Ontology and Cell Taxonomy using the same NLP approach,
        providing comprehensive mapping results with additional marker gene information.
        
        Parameters
        ----------
        cell_names : list
            📝 List of cell names to map
        threshold : float
            📊 Similarity threshold
        expand_abbreviations : bool
            🔄 Whether to enable abbreviation expansion
        use_taxonomy : bool
            📊 Whether to include taxonomy resource in mapping
        species : str, optional
            🐭 Species information for taxonomy filtering
        tissue_context : str or list, optional
            🧬 Tissue context information
        study_context : str, optional
            🔬 Study context information
        
        Returns
        -------
        mapping_results : dict
            📋 Combined mapping results from ontology and taxonomy
        """
        # First perform standard ontology mapping with expansion
        mapping_results = self.map_cells_with_expansion(
            cell_names, threshold, expand_abbreviations,
            tissue_context=tissue_context, species=species, study_context=study_context,
            use_llm_selection=use_llm_selection,llm_candidates_count=llm_candidates_count
        )
        
        # Enhance with taxonomy if available and requested
        if use_taxonomy and self.taxonomy_embeddings is not None:
            print("\n📊 Enhancing with taxonomy resource using NLP similarity...")
            
            self._enhance_with_taxonomy(mapping_results, cell_names, threshold, species,
                                        use_llm_selection=use_llm_selection,llm_candidates_count=llm_candidates_count)
        
        return mapping_results
    
    def _enhance_with_taxonomy(self, mapping_results, cell_names, threshold, species,
                               use_llm_selection=True, llm_candidates_count=10):
        """🧬 Enhance mapping results with taxonomy information using NLP similarity and LLM selection"""
        self._load_model()
        
        # Get taxonomy embedding matrix
        taxonomy_emb_matrix = np.array([
            self.taxonomy_embeddings[label] for label in self.taxonomy_labels
        ])
        
        # Determine LLM candidates count for taxonomy (similar to ontology)
        #llm_candidates_count = 10  # Default number of taxonomy candidates for LLM selection
        
        print(f"🧬 Taxonomy enhancement: Processing {len(cell_names)} cell names...")
        if self.llm_client is not None:
            print(f"🤖 LLM-enhanced taxonomy selection enabled (from top {llm_candidates_count} candidates)")
        
        llm_taxonomy_selection_count = 0
        
        for cell_name in cell_names:
            if cell_name not in mapping_results:
                continue
                
            result = mapping_results[cell_name]
            
            # PRIORITIZE expanded name for taxonomy matching
            names_to_check = []
            expanded_name = result.get('expanded_name', cell_name)
            was_expanded = result.get('was_expanded', False)
            
            # First priority: Use expanded name if available
            if was_expanded and expanded_name != cell_name:
                names_to_check.append(expanded_name)
                print(f"  🔤 Using expanded name '{expanded_name}' (from '{cell_name}') for taxonomy matching")
            else:
                names_to_check.append(cell_name)
             
            # Secondary options: original name and best ontology match  
            if cell_name not in names_to_check:
                names_to_check.append(cell_name)
                
            best_match = result.get('best_match', '')
            if best_match and best_match not in names_to_check:
                names_to_check.append(best_match)
            
            # Find best taxonomy matches using NLP similarity (same approach as ontology)
            best_taxonomy_match = None
            best_taxonomy_similarity = 0
            best_matched_from = None
            taxonomy_selection_method = "cosine_similarity"
            taxonomy_llm_reasoning = None
            taxonomy_llm_confidence = None
            
            for name_to_check in names_to_check:
                # Encode the name
                name_embedding = self.model.encode([name_to_check])
                
                # Calculate similarities with taxonomy
                similarities = cosine_similarity(name_embedding, taxonomy_emb_matrix)[0]
                
                # Get top matches above threshold
                valid_indices = np.where(similarities > threshold)[0]
                if len(valid_indices) == 0:
                    continue
                
                # Get top candidates for potential LLM selection
                top_indices = np.argsort(similarities)[-llm_candidates_count:][::-1]
                top_taxonomy_candidates = []
                
                for idx in top_indices:
                    if similarities[idx] <= threshold:
                        continue
                        
                    taxonomy_label = self.taxonomy_labels[idx]
                    taxonomy_similarity = similarities[idx]
                    
                    # Get detailed taxonomy info
                    taxonomy_entries = self.taxonomy_info_dict.get(taxonomy_label, [])
                    
                    # Filter by species if specified
                    if species:
                        species_entries = [
                            entry for entry in taxonomy_entries 
                            if entry.get('Species') == species
                        ]
                        if species_entries:
                            target_entries = species_entries
                        else:
                            target_entries = taxonomy_entries
                    else:
                        target_entries = taxonomy_entries
                    
                    if target_entries:
                        top_taxonomy_candidates.append({
                            'cell_type': taxonomy_label,
                            'similarity': taxonomy_similarity,
                            'info': target_entries[0],  # Use first matching entry
                            'matched_from': name_to_check
                        })
                
                if not top_taxonomy_candidates:
                    continue
                
                # Sort candidates by similarity (descending)
                top_taxonomy_candidates.sort(key=lambda x: x['similarity'], reverse=True)
                
                # Try LLM selection if enabled and available
                selected_candidate = None
                if self.llm_client is not None and len(top_taxonomy_candidates) > 1:
                    print(f"    🤖 LLM selecting taxonomy match for: {name_to_check}")
                    
                    # Call LLM for taxonomy selection
                    llm_result = self._call_llm_for_taxonomy_selection(
                        cell_name, top_taxonomy_candidates, expanded_name if was_expanded else None
                    )
                    
                    if llm_result and 'selected_rank' in llm_result:
                        selected_rank = llm_result['selected_rank']
                        selected_idx = selected_rank - 1
                        
                        if 0 <= selected_idx < len(top_taxonomy_candidates):
                            # Use LLM selection
                            selected_candidate = top_taxonomy_candidates[selected_idx]
                            taxonomy_selection_method = "llm_selection"
                            taxonomy_llm_reasoning = llm_result.get('reasoning', '')
                            taxonomy_llm_confidence = llm_result.get('confidence', 'unknown')
                            llm_taxonomy_selection_count += 1
                            
                            print(f"      ✓ LLM selected: {selected_candidate['cell_type']} (rank {selected_rank}, sim: {selected_candidate['similarity']:.3f})")
                            if taxonomy_llm_reasoning:
                                print(f"      💡 Reasoning: {taxonomy_llm_reasoning[:100]}...")
                        else:
                            print(f"      ⚠️  LLM selected invalid taxonomy rank, using top similarity")
                    else:
                        print(f"      ✗ LLM taxonomy selection failed, using top similarity")
                
                # Fallback to top similarity if LLM didn't select or failed
                if selected_candidate is None:
                    selected_candidate = top_taxonomy_candidates[0]  # Top similarity
                
                # Update best match if this is better
                if selected_candidate['similarity'] > best_taxonomy_similarity:
                    best_taxonomy_similarity = selected_candidate['similarity']
                    best_taxonomy_match = selected_candidate
                    best_matched_from = selected_candidate['matched_from']
                    
                    # Show successful taxonomy match with selection method
                    species_note = f" ({selected_candidate['info'].get('Species', 'Unknown')})" if species else ""
                    selection_note = " [LLM SELECTED]" if taxonomy_selection_method == "llm_selection" else ""
                    
                    if was_expanded and best_matched_from == expanded_name:
                        print(f"      ✓ Taxonomy match: '{best_matched_from}' → '{selected_candidate['cell_type']}'{species_note} (sim: {selected_candidate['similarity']:.3f}){selection_note} [USED EXPANDED NAME]")
                    else:
                        print(f"      ✓ Taxonomy match: '{best_matched_from}' → '{selected_candidate['cell_type']}'{species_note} (sim: {selected_candidate['similarity']:.3f}){selection_note}")
                    
                    # If we found a species-specific match with good confidence, we can break
                    if not species or selected_candidate['info'].get('Species') == species:
                        if taxonomy_selection_method == "llm_selection" or selected_candidate['similarity'] > 0.8:
                            break
                
                # If we found a good match for this name, we can stop trying other names
                if best_taxonomy_match and (not species or best_taxonomy_match['info'].get('Species') == species):
                    if taxonomy_selection_method == "llm_selection" or best_taxonomy_match['similarity'] > 0.8:
                        break
            
            # Add taxonomy information to results
            if best_taxonomy_match:
                info = best_taxonomy_match['info']
                result['taxonomy_match'] = best_taxonomy_match['cell_type']
                result['taxonomy_similarity'] = best_taxonomy_match['similarity']
                result['taxonomy_matched_from'] = best_taxonomy_match['matched_from']
                result['taxonomy_selection_method'] = taxonomy_selection_method
                result['taxonomy_llm_reasoning'] = taxonomy_llm_reasoning
                result['taxonomy_llm_confidence'] = taxonomy_llm_confidence
                
                # Add detailed taxonomy information
                result['ct_id'] = info.get('CT_ID')
                result['cell_marker'] = info.get('Cell_Marker')
                result['specific_cell_ontology_id'] = info.get('Specific_Cell_Ontology_ID')
                result['gene_info'] = {
                    'entrez_id': info.get('Gene_ENTREZID'),
                    'gene_alias': info.get('Gene_Alias'),
                    'ensembl_id': info.get('Gene_Ensembl_ID'),
                    'uniprot': info.get('Uniprot'),
                    'pfam': info.get('PFAM'),
                    'go_terms': info.get('GO2')
                }
                result['tissue_ontology_id'] = info.get('Tissue_UberonOntology_ID')
                result['pmid'] = info.get('PMID')
                result['source'] = info.get('Source')
                
                # Boost confidence if taxonomy provides strong support
                if best_taxonomy_match['similarity'] > 0.8 or taxonomy_selection_method == "llm_selection":
                    if result['confidence'] == 'low':
                        result['confidence'] = 'medium'
                        result['confidence_reason'] = f'Enhanced by taxonomy resource ({taxonomy_selection_method})'
                    elif result['confidence'] == 'medium':
                        result['confidence'] = 'high'  
                        result['confidence_reason'] = f'Strong taxonomy confirmation ({taxonomy_selection_method})'
            else:
                # Show when no taxonomy match found
                if was_expanded:
                    print(f"      ✗ No taxonomy match found for '{expanded_name}' (expanded from '{cell_name}')")
                else:
                    print(f"      ✗ No taxonomy match found for '{cell_name}'")
        
        if self.llm_client is not None:
            print(f"✓ LLM taxonomy selection summary: {llm_taxonomy_selection_count} selections made by LLM")
    
    def search_by_marker(self, markers, species=None, top_k=10):
        """
        🎯 Search cell types by gene markers using taxonomy resource
        
        Parameters
        ----------
        markers : str or list
            🧬 Gene marker(s) to search for
        species : str, optional
            🐭 Species to filter by
        top_k : int
            📊 Maximum number of results to return
            
        Returns
        -------
        matches : list
            📊 List of matching cell types with detailed information
        """
        if self.taxonomy_resource is None:
            print("✗ No taxonomy resource loaded")
            return []
        
        if isinstance(markers, str):
            markers = [markers]
        
        df = self.taxonomy_resource
        if species:
            df = df[df['Species'] == species]
        
        matches = []
        
        for marker in markers:
            # Search in Cell_Marker column
            marker_matches = df[df['Cell_Marker'].str.contains(marker, na=False, case=False)]
            
            # Search in Gene_Alias column
            alias_matches = df[df['Gene_Alias'].str.contains(marker, na=False, case=False)]
            
            # Combine matches
            combined_matches = pd.concat([marker_matches, alias_matches]).drop_duplicates()
            
            for _, row in combined_matches.iterrows():
                match_info = {
                    'cell_type': row['Cell_standard'],
                    'species': row['Species'],
                    'marker': row['Cell_Marker'],
                    'ct_id': row['CT_ID'],
                    'gene_alias': row['Gene_Alias'],
                    'entrez_id': row['Gene_ENTREZID'],
                    'matched_marker': marker,
                    'source': row.get('Source', 'Unknown')
                }
                matches.append(match_info)
        
        # Remove duplicates and limit results
        unique_matches = []
        seen = set()
        for match in matches:
            key = f"{match['cell_type']}_{match['species']}_{match['ct_id']}"
            if key not in seen and len(unique_matches) < top_k:
                seen.add(key)
                unique_matches.append(match)
        
        print(f"🎯 Found {len(unique_matches)} cell types with markers {markers}")
        for match in unique_matches[:5]:  # Show top 5
            print(f"  🧬 {match['cell_type']} ({match['species']}) - Marker: {match['marker']}")
        
        return unique_matches

    def map_adata_with_taxonomy(self, adata, cell_name_col=None, threshold=0.5, 
                                new_col_name='cell_ontology', expand_abbreviations=True,
                                use_taxonomy=True, species=None, tissue_context=None, study_context=None):
        """
        🧬 Apply taxonomy-enhanced mapping to AnnData object
        
        This method applies the unified ontology + taxonomy mapping to AnnData,
        providing comprehensive cell type annotation with gene marker information.
        
        Parameters
        ----------
        adata : AnnData
            📊 Single-cell data object
        cell_name_col : str, optional
            📝 Column name containing cell names, use index if None
        threshold : float
            📊 Similarity threshold
        new_col_name : str
            🏷️  Base name for new columns
        expand_abbreviations : bool
            🔄 Whether to enable LLM abbreviation expansion
        use_taxonomy : bool
            📊 Whether to include taxonomy resource enhancement
        species : str, optional
            🐭 Species information for taxonomy filtering
        tissue_context : str or list, optional
            🧬 Tissue context information
        study_context : str, optional
            🔬 Study context information
        
        Returns
        -------
        mapping_results : dict
            📋 Enhanced mapping results with both ontology and taxonomy information
        """
        # Get cell names
        if cell_name_col is None:
            cell_names = adata.obs.index.unique().tolist()
            cell_names_series = adata.obs.index.to_series()
            print(f"📊 Using {len(cell_names)} unique cell names from index")
        else:
            cell_names = adata.obs[cell_name_col].unique().tolist()
            cell_names_series = adata.obs[cell_name_col]
            print(f"📊 Using {len(cell_names)} unique cell names from column '{cell_name_col}'")
        
        # Perform taxonomy-enhanced mapping
        mapping_results = self.map_cells_with_taxonomy(
            cell_names, threshold, expand_abbreviations, use_taxonomy,
            species=species, tissue_context=tissue_context, study_context=study_context
        )
        
        # Apply to adata
        print("\n📝 Applying enhanced mapping results to AnnData...")
        
        # Helper functions for extracting information
        def get_best_match(cell_name):
            return mapping_results.get(cell_name, {}).get('best_match', 'Unknown')
        
        def get_similarity(cell_name):
            return mapping_results.get(cell_name, {}).get('similarity', 0.0)
        
        def get_confidence(cell_name):
            return mapping_results.get(cell_name, {}).get('confidence', 'low')
        
        def get_ontology_id(cell_name):
            return mapping_results.get(cell_name, {}).get('ontology_id', None)
        
        def get_cl_id(cell_name):
            return mapping_results.get(cell_name, {}).get('cl_id', None)
        
        def get_expanded_name(cell_name):
            return mapping_results.get(cell_name, {}).get('expanded_name', cell_name)
        
        def was_expanded(cell_name):
            return mapping_results.get(cell_name, {}).get('was_expanded', False)
        
        # Taxonomy-specific helper functions
        def get_taxonomy_match(cell_name):
            return mapping_results.get(cell_name, {}).get('taxonomy_match', None)
        
        def get_taxonomy_similarity(cell_name):
            return mapping_results.get(cell_name, {}).get('taxonomy_similarity', 0.0)
        
        def get_ct_id(cell_name):
            return mapping_results.get(cell_name, {}).get('ct_id', None)
        
        def get_cell_marker(cell_name):
            return mapping_results.get(cell_name, {}).get('cell_marker', None)
        
        def get_gene_entrez_id(cell_name):
            gene_info = mapping_results.get(cell_name, {}).get('gene_info', {})
            return gene_info.get('entrez_id', None)
        
        def get_gene_alias(cell_name):
            gene_info = mapping_results.get(cell_name, {}).get('gene_info', {})
            return gene_info.get('gene_alias', None)
        
        def get_pmid(cell_name):
            return mapping_results.get(cell_name, {}).get('pmid', None)
        
        # Apply standard ontology columns
        adata.obs[new_col_name] = cell_names_series.apply(get_best_match)
        adata.obs[f'{new_col_name}_similarity'] = cell_names_series.apply(get_similarity)
        adata.obs[f'{new_col_name}_confidence'] = cell_names_series.apply(get_confidence)
        adata.obs[f'{new_col_name}_ontology_id'] = cell_names_series.apply(get_ontology_id)
        adata.obs[f'{new_col_name}_cl_id'] = cell_names_series.apply(get_cl_id)
        adata.obs[f'{new_col_name}_expanded'] = cell_names_series.apply(get_expanded_name)
        adata.obs[f'{new_col_name}_was_expanded'] = cell_names_series.apply(was_expanded)
        
        # Apply taxonomy enhancement columns if taxonomy is used
        if use_taxonomy and self.taxonomy_embeddings is not None:
            adata.obs[f'{new_col_name}_taxonomy_match'] = cell_names_series.apply(get_taxonomy_match)
            adata.obs[f'{new_col_name}_taxonomy_similarity'] = cell_names_series.apply(get_taxonomy_similarity)
            adata.obs[f'{new_col_name}_ct_id'] = cell_names_series.apply(get_ct_id)
            adata.obs[f'{new_col_name}_cell_marker'] = cell_names_series.apply(get_cell_marker)
            adata.obs[f'{new_col_name}_gene_entrez_id'] = cell_names_series.apply(get_gene_entrez_id)
            adata.obs[f'{new_col_name}_gene_alias'] = cell_names_series.apply(get_gene_alias)
            adata.obs[f'{new_col_name}_pmid'] = cell_names_series.apply(get_pmid)
        
        # Statistics
        high_conf_count = sum(1 for r in mapping_results.values() if r['confidence'] == 'high')
        medium_conf_count = sum(1 for r in mapping_results.values() if r['confidence'] == 'medium')
        low_conf_count = sum(1 for r in mapping_results.values() if r['confidence'] == 'low')
        expanded_count = sum(1 for r in mapping_results.values() if r.get('was_expanded', False))
        
        print(f"✓ Enhanced mapping completed:")
        print(f"  📊 {high_conf_count}/{len(mapping_results)} high confidence mappings")
        print(f"  📊 {medium_conf_count}/{len(mapping_results)} medium confidence mappings")
        print(f"  📊 {low_conf_count}/{len(mapping_results)} low confidence mappings")
        print(f"  🔄 {expanded_count}/{len(mapping_results)} abbreviation expansions")
        
        if use_taxonomy and self.taxonomy_embeddings is not None:
            taxonomy_enhanced_count = sum(1 for r in mapping_results.values() if 'taxonomy_match' in r)
            print(f"  📊 {taxonomy_enhanced_count}/{len(mapping_results)} enhanced with taxonomy resource")
        
        return mapping_results

    def print_mapping_summary_taxonomy(self, mapping_results, top_n=10):
        """📋 Print comprehensive mapping summary with taxonomy information"""
        stats = self.get_statistics(mapping_results)
        
        print("\n" + "="*80)
        print("ENHANCED MAPPING SUMMARY (ONTOLOGY + TAXONOMY)")
        print("="*80)
        print(f"Total mappings:\t\t{stats['total_mappings']}")
        print(f"High confidence:\t{stats['high_confidence']} ({stats['high_confidence_ratio']:.2%})")
        print(f"Low confidence:\t\t{stats['low_confidence']}")
        print(f"Average similarity:\t{stats['mean_similarity']:.3f}")
        
        # Count taxonomy enhancements
        taxonomy_enhanced = sum(1 for r in mapping_results.values() if 'taxonomy_match' in r and r['taxonomy_match'])
        llm_expanded = sum(1 for r in mapping_results.values() if r.get('was_expanded', False))
        
        print(f"LLM expansions:\t\t{llm_expanded}")
        print(f"Taxonomy enhanced:\t{taxonomy_enhanced}")
        
        print(f"\nDETAILED MAPPING RESULTS (Top {top_n})")
        print("-" * 80)
        
        sorted_results = sorted(
            mapping_results.items(), 
            key=lambda x: x[1]['similarity'], 
            reverse=True
        )
        
        for i, (cell_name, result) in enumerate(sorted_results[:top_n]):
            conf_mark = "✓" if result['confidence'] == 'high' else "?" if result['confidence'] == 'medium' else "✗"
            
            print(f"\n{i+1:2d}. [{conf_mark}] {cell_name}")
            
            # Show expansion info
            if result.get('was_expanded', False):
                expanded_name = result.get('expanded_name', cell_name)
                print(f"     🔤 Expanded: {cell_name} → {expanded_name}")
            
            # Ontology mapping
            print(f"     🎯 Ontology: {result['best_match']}")
            print(f"        Similarity: {result['similarity']:.3f}")
            cl_id = result.get('cl_id', 'N/A')
            print(f"        CL ID: {cl_id}")
            
            # Taxonomy enhancement (if available)
            taxonomy_match = result.get('taxonomy_match')
            if taxonomy_match:
                taxonomy_sim = result.get('taxonomy_similarity', 0)
                matched_from = result.get('taxonomy_matched_from', 'Unknown')
                print(f"     🧬 Taxonomy: {taxonomy_match}")
                print(f"        Similarity: {taxonomy_sim:.3f}")
                print(f"        Matched from: {matched_from}")
                
                # Show additional taxonomy info
                ct_id = result.get('ct_id')
                if ct_id:
                    print(f"        CT ID: {ct_id}")
                
                cell_marker = result.get('cell_marker')
                if cell_marker:
                    print(f"        🎯 Marker: {cell_marker}")
                
                gene_info = result.get('gene_info', {})
                gene_alias = gene_info.get('gene_alias')
                if gene_alias:
                    print(f"        🧬 Gene: {gene_alias}")
                
                entrez_id = gene_info.get('entrez_id')
                if entrez_id:
                    print(f"        🆔 ENTREZ: {entrez_id}")
            else:
                print(f"     🧬 Taxonomy: No match found")
            
            print()
        
        # Show summary of issues
        no_taxonomy_matches = [
            name for name, result in mapping_results.items() 
            if not result.get('taxonomy_match')
        ]
        
        if no_taxonomy_matches:
            print(f"\n⚠️  CELLS WITHOUT TAXONOMY MATCHES ({len(no_taxonomy_matches)}):")
            for name in no_taxonomy_matches:
                result = mapping_results[name]
                ontology_match = result.get('best_match', 'Unknown')
                similarity = result.get('similarity', 0)
                print(f"  - {name} → {ontology_match} (sim: {similarity:.3f})")
        
        print("\n" + "="*80)
    
    def find_similar_cells_taxonomy(self, cell_name, species=None, top_k=10):
        """
        🧬 Find taxonomy cell types most similar to given cell name
        
        Parameters
        ----------
        cell_name : str
            📝 Input cell name
        species : str, optional
            🐭 Filter by species (e.g., "Homo sapiens", "Mus musculus")
        top_k : int
            📊 Return top k most similar results
        
        Returns
        -------
        similar_cells : list
            📋 Similar cell types with similarities and taxonomy info
        """
        if self.taxonomy_embeddings is None:
            print("✗ Please load taxonomy resource first using load_cell_taxonomy_resource()")
            return []
        
        self._load_model()
        
        # Encode input cell name
        cell_embedding = self.model.encode([cell_name])
        
        # Get taxonomy embedding matrix
        taxonomy_emb_matrix = np.array([
            self.taxonomy_embeddings[label] for label in self.taxonomy_labels
        ])
        
        # Calculate similarities
        similarities = cosine_similarity(cell_embedding, taxonomy_emb_matrix)[0]
        
        # Get top-k most similar
        top_indices = np.argsort(similarities)[-top_k*3:][::-1]  # Get more to filter by species
        
        similar_cells = []
        for idx in top_indices:
            if len(similar_cells) >= top_k:
                break
                
            taxonomy_label = self.taxonomy_labels[idx]
            similarity = similarities[idx]
            
            # Get detailed taxonomy info
            taxonomy_entries = self.taxonomy_info_dict.get(taxonomy_label, [])
            
            # Filter by species if specified
            if species:
                taxonomy_entries = [
                    entry for entry in taxonomy_entries 
                    if entry.get('Species') == species
                ]
            
            if taxonomy_entries:
                # Use first matching entry
                entry = taxonomy_entries[0]
                cell_info = {
                    'cell_type': taxonomy_label,
                    'similarity': similarity,
                    'species': entry.get('Species'),
                    'ct_id': entry.get('CT_ID'),
                    'cell_marker': entry.get('Cell_Marker'),
                    'gene_alias': entry.get('Gene_Alias'),
                    'entrez_id': entry.get('Gene_ENTREZID'),
                    'pmid': entry.get('PMID'),
                    'source': entry.get('Source')
                }
                similar_cells.append(cell_info)
        
        print(f"\n🧬 Taxonomy cell types most similar to '{cell_name}':")
        if species:
            print(f"🐭 Filtered by species: {species}")
        
        for i, cell_info in enumerate(similar_cells):
            print(f"{i+1:2d}. {cell_info['cell_type']:<40} (Similarity: {cell_info['similarity']:.3f})")
            print(f"     🐭 Species: {cell_info['species']}")
            if cell_info['cell_marker']:
                print(f"     🎯 Marker: {cell_info['cell_marker']}")
            if cell_info['ct_id']:
                print(f"     🆔 CT ID: {cell_info['ct_id']}")
            print()
        
        return similar_cells
    
    def map_cells_with_expansion(self, cell_names, threshold=0.5, expand_abbreviations=True,
                           tissue_context=None, species=None, study_context=None,
                           use_llm_selection=True, llm_candidates_count=10):
        """
        🔄 First expand abbreviations, then perform ontology mapping with optional LLM selection
        
        Parameters
        ----------
        cell_names : list
            📝 List of cell names to map
        threshold : float
            📊 Similarity threshold
        expand_abbreviations : bool
            🔄 Whether to enable abbreviation expansion
        tissue_context : str or list, optional
            🧬 Tissue context information
        species : str, optional
            🐭 Species information
        study_context : str, optional
            🔬 Study context information
        use_llm_selection : bool
            🤖 Whether to use LLM for selecting from top candidates (when LLM is available)
        llm_candidates_count : int
            📊 Number of top candidates to send to LLM for selection
        
        Returns
        -------
        mapping_results : dict
            📋 Mapping results (including original and expanded name information + LLM selection)
        """
        if expand_abbreviations and self.llm_client is not None:
            print("📝 Step 1: Expanding abbreviations")
            expanded_names = self.expand_abbreviations(
                cell_names, 
                tissue_context=tissue_context,
                species=species, 
                study_context=study_context
            )
            
            print("\n🎯 Step 2: Performing ontology mapping with LLM-enhanced selection")
            expanded_cell_names = list(expanded_names.values())
            base_results = self.map_cells(expanded_cell_names, threshold, 
                                        use_llm_selection=use_llm_selection,
                                        llm_candidates_count=llm_candidates_count)
            
            # Reorganize results with original name information and enhanced with LLM selection
            mapping_results = {}
            for original_name in cell_names:
                expanded_name = expanded_names[original_name]
                if expanded_name in base_results:
                    result = base_results[expanded_name].copy()
                    result['original_name'] = original_name
                    result['expanded_name'] = expanded_name
                    result['was_expanded'] = (original_name != expanded_name)
                    
                    # For LLM selection, also pass the expanded name for better context
                    if use_llm_selection and result.get('selection_method') == 'llm_selection':
                        # Re-call LLM with both original and expanded names for better context
                        if self.llm_client is not None and original_name != expanded_name:
                            # Get top candidates again for this specific call
                            if self.ontology_embeddings is not None:
                                self._load_model()
                                # Encode the expanded name
                                cell_embedding = self.model.encode([expanded_name])
                                ontology_emb_matrix = np.array([
                                    self.ontology_embeddings[label] for label in self.ontology_labels
                                ])
                                similarities = cosine_similarity(cell_embedding, ontology_emb_matrix)[0]
                                
                                # Get top candidates
                                top_indices = np.argsort(similarities)[-llm_candidates_count:][::-1]
                                top_candidates = []
                                for idx in top_indices:
                                    match_label = self.ontology_labels[idx]
                                    match_similarity = similarities[idx]
                                    top_candidates.append({
                                        'label': match_label,
                                        'similarity': match_similarity,
                                        'index': idx
                                    })
                                
                                # Call LLM with enhanced context (original + expanded)
                                llm_result = self._call_llm_for_selection(original_name, top_candidates, expanded_name)
                                
                                if llm_result and 'selected_rank' in llm_result:
                                    selected_rank = llm_result['selected_rank']
                                    selected_idx = selected_rank - 1
                                    
                                    if 0 <= selected_idx < len(top_candidates):
                                        # Update with new LLM selection that considered both names
                                        selected_candidate = top_candidates[selected_idx]
                                        result['best_match'] = selected_candidate['label']
                                        result['similarity'] = selected_candidate['similarity']
                                        result['llm_reasoning'] = llm_result.get('reasoning', '')
                                        result['llm_confidence'] = llm_result.get('confidence', 'unknown')
                                        result['selection_method'] = 'llm_selection_with_expansion'
                                        
                                        # Update ontology info
                                        ontology_info = self._get_ontology_id(selected_candidate['label'])
                                        result['ontology_id'] = ontology_info['ontology_id']
                                        result['cl_id'] = ontology_info['cl_id']
                    
                    mapping_results[original_name] = result
                else:
                    # This shouldn't happen, but as backup
                    mapping_results[original_name] = {
                        'best_match': 'Unknown',
                        'similarity': 0.0,
                        'confidence': 'low',
                        'original_name': original_name,
                        'expanded_name': expanded_name,
                        'was_expanded': (original_name != expanded_name),
                        'top3_matches': [],
                        'selection_method': 'failed',
                        'llm_reasoning': None,
                        'llm_confidence': None
                    }
        else:
            if expand_abbreviations and self.llm_client is None:
                print("⚠️  Abbreviation expansion requested but LLM client not configured")
                print("💡 To enable LLM expansion, use: mapper.setup_llm_expansion()")
                print("🎯 Performing direct ontology mapping with LLM selection")
            elif not expand_abbreviations:
                print("🎯 Performing direct ontology mapping with LLM selection (abbreviation expansion disabled by parameter)")
            else:
                print("🎯 Performing direct ontology mapping with LLM selection")
                
            mapping_results = self.map_cells(cell_names, threshold, 
                                        use_llm_selection=use_llm_selection,
                                        llm_candidates_count=llm_candidates_count)
            
            # Add expansion information
            for cell_name in mapping_results:
                mapping_results[cell_name]['original_name'] = cell_name
                mapping_results[cell_name]['expanded_name'] = cell_name
                mapping_results[cell_name]['was_expanded'] = False
        
        return mapping_results
    
    def download_model(self):
        """
        📥 Manually download and load the model
        
        Returns
        -------
        bool
            ✓ True if successful, False otherwise
        """
        try:
            self._load_model()
            return True
        except Exception as e:
            print(f"✗ Model download failed: {e}")
            return False

    def get_cell_info_taxonomy(self, cell_name, species=None):
        """
        🧬 Get detailed taxonomy information for specific cell type
        
        Parameters
        ----------
        cell_name : str
            📝 Cell type name
        species : str, optional
            🐭 Filter by species (e.g., "Homo sapiens", "Mus musculus")
        
        Returns
        -------
        info_list : list
            📋 List of taxonomy information dictionaries
        """
        if self.taxonomy_resource is None:
            print("✗ Please load taxonomy resource first using load_cell_taxonomy_resource()")
            return []
        
        # Search for exact or partial matches
        df = self.taxonomy_resource
        
        # Try exact match first
        exact_matches = df[df['Cell_standard'].str.lower() == cell_name.lower()]
        
        if exact_matches.empty:
            # Try partial match
            partial_matches = df[df['Cell_standard'].str.contains(cell_name, case=False, na=False)]
            if partial_matches.empty:
                print(f"✗ Cell type not found in taxonomy: {cell_name}")
                
                # Suggest similar cells using NLP
                if self.taxonomy_embeddings:
                    print("💡 Searching for similar cell types...")
                    similar = self.find_similar_cells_taxonomy(cell_name, species=species, top_k=5)
                    if similar:
                        print("💡 Did you mean one of these:")
                        for s in similar[:3]:
                            print(f"  - {s['cell_type']} (sim: {s['similarity']:.3f})")
                return []
            else:
                matches = partial_matches
        else:
            matches = exact_matches
        
        # Filter by species if specified
        if species:
            matches = matches[matches['Species'] == species]
            if matches.empty:
                print(f"✗ Cell type '{cell_name}' not found for species '{species}'")
                return []
        
        print(f"\n🧬 === Cell Taxonomy Information: {cell_name} ===")
        
        info_list = []
        for i, (_, row) in enumerate(matches.iterrows()):
            print(f"\n📊 Entry {i+1}:")
            
            info = {
                'cell_type': row['Cell_standard'],
                'species': row['Species'],
                'ct_id': row.get('CT_ID'),
                'cell_marker': row.get('Cell_Marker'),
                'specific_cell_ontology_id': row.get('Specific_Cell_Ontology_ID'),
                'gene_info': {
                    'entrez_id': row.get('Gene_ENTREZID'),
                    'gene_alias': row.get('Gene_Alias'),
                    'ensembl_id': row.get('Gene_Ensembl_ID'),
                    'uniprot': row.get('Uniprot'),
                    'pfam': row.get('PFAM'),
                    'go_terms': row.get('GO2')
                },
                'tissue_ontology_id': row.get('Tissue_UberonOntology_ID'),
                'pmid': row.get('PMID'),
                'source': row.get('Source')
            }
            
            # Display information
            print(f"🐭 Species: {info['species']}")
            if info['ct_id']:
                print(f"🆔 CT ID: {info['ct_id']}")
            if info['cell_marker']:
                print(f"🎯 Cell Marker: {info['cell_marker']}")
            if info['specific_cell_ontology_id']:
                print(f"🔗 Cell Ontology ID: {info['specific_cell_ontology_id']}")
            
            # Gene information
            gene_info = info['gene_info']
            if any(gene_info.values()):
                print(f"🧬 Gene Information:")
                if gene_info['gene_alias']:
                    print(f"   Gene Alias: {gene_info['gene_alias']}")
                if gene_info['entrez_id']:
                    print(f"   ENTREZ ID: {gene_info['entrez_id']}")
                if gene_info['ensembl_id']:
                    print(f"   Ensembl ID: {gene_info['ensembl_id']}")
                if gene_info['uniprot']:
                    print(f"   UniProt: {gene_info['uniprot']}")
                if gene_info['pfam']:
                    print(f"   PFAM: {gene_info['pfam']}")
                if gene_info['go_terms']:
                    # Truncate GO terms if too long
                    go_terms = str(gene_info['go_terms'])
                    if len(go_terms) > 100:
                        go_terms = go_terms[:100] + "..."
                    print(f"   GO Terms: {go_terms}")
            
            if info['tissue_ontology_id']:
                print(f"🧬 Tissue Ontology ID: {info['tissue_ontology_id']}")
            
            if info['pmid']:
                print(f"📚 PMID: {info['pmid']}")
            
            if info['source']:
                print(f"📄 Source: {info['source']}")
            
            info_list.append(info)
            
            if i >= 4:  # Limit to 5 entries
                remaining = len(matches) - 5
                if remaining > 0:
                    print(f"\n... and {remaining} more entries (use species filter to narrow results)")
                break
        
        return info_list
    
    def _call_llm_for_taxonomy_selection(self, cell_name, top_taxonomy_candidates, expanded_name=None):
        """
        🤖 Call LLM to select the best taxonomy cell type from top candidates
        
        Parameters
        ----------
        cell_name : str
            📝 Original cell name
        top_taxonomy_candidates : list
            📋 List of taxonomy candidate dictionaries with 'cell_type', 'similarity', and 'info' keys
        expanded_name : str, optional
            🔤 Expanded cell name if abbreviation expansion was performed
        
        Returns
        -------
        selection_result : dict or None
            📋 LLM selection result with selected candidate and reasoning
        """
        if self.llm_client is None:
            return None
        
        # Build context information
        context_parts = []
        
        if self.species and self.species != "human":
            context_parts.append(f"Species: {self.species}")
        
        if self.tissue_context:
            if isinstance(self.tissue_context, list):
                tissue_info = ", ".join(self.tissue_context)
            else:
                tissue_info = self.tissue_context
            context_parts.append(f"Tissue/Organ context: {tissue_info}")
        
        if self.study_context:
            context_parts.append(f"Study context: {self.study_context}")
        
        context_str = "\n".join(context_parts) if context_parts else ""
        
        # Prepare candidate list for the prompt
        candidates_text = []
        for i, candidate in enumerate(top_taxonomy_candidates, 1):
            info = candidate['info']
            species_info = info.get('Species', 'Unknown')
            marker_info = info.get('Cell_Marker', 'N/A')
            gene_alias = info.get('Gene_Alias', 'N/A')
            
            candidate_desc = f"{i}. {candidate['cell_type']} (similarity: {candidate['similarity']:.3f})"
            candidate_desc += f"\n   Species: {species_info}"
            candidate_desc += f"\n   Cell Marker: {marker_info}"
            if gene_alias != 'N/A':
                candidate_desc += f"\n   Gene Alias: {gene_alias}"
            
            candidates_text.append(candidate_desc)
        
        candidates_str = "\n".join(candidates_text)
        
        # Build the prompt
        cell_info = f"Original cell name: '{cell_name}'"
        if expanded_name and expanded_name != cell_name:
            cell_info += f"\nExpanded cell name: '{expanded_name}'"
        
        prompt = f"""You are an expert in cell biology and immunology. Your task is to select the most appropriate cell type from a list of taxonomy candidates based on biological context and marker gene information.

{context_str}

{cell_info}

Here are the top candidate cell types from Cell Taxonomy database ranked by semantic similarity:

{candidates_str}

Based on your expertise in cell biology and the provided context information, please select the most biologically appropriate cell type from the taxonomy candidates above.

Consider the following factors:
1. Biological accuracy and cell type definitions
2. Species compatibility (if species context is provided)
3. Tissue/organ context compatibility 
4. Cell marker gene specificity and accuracy
5. Gene expression patterns and cellular functions
6. Study context relevance (e.g., cancer, development, aging)
7. Literature evidence and experimental validation

Please respond in JSON format:
{{
    "selected_rank": 1,
    "selected_cell_type": "selected cell type name",
    "confidence": "high/medium/low",
    "reasoning": "brief biological explanation focusing on marker genes and tissue context"
}}

Please provide only the JSON response."""

        # Initialize content variable
        content = None
        api_type = self.llm_config.get('api_type', 'unknown')
        
        try:
            if api_type in ["openai", "custom_openai", "doubao"]:
                # OpenAI API and compatible APIs
                response = self.llm_client.chat.completions.create(
                    model=self.llm_config['model'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=400,
                    **self.llm_config.get('extra_params', {})
                )
                content = response.choices[0].message.content
                
            elif api_type == "anthropic":
                response = self.llm_client.messages.create(
                    model=self.llm_config['model'],
                    max_tokens=400,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}],
                    **self.llm_config.get('extra_params', {})
                )
                content = response.content[0].text
                
            elif api_type == "ollama":
                import requests
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.llm_config['model'],
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1, **self.llm_config.get('extra_params', {})}
                    },
                    timeout=30
                )
                response.raise_for_status()
                content = response.json().get('response', '')
                
            elif api_type == "qwen":
                # 阿里云通义千问
                import dashscope
                from dashscope import Generation
                response = dashscope.Generation.call(
                    model=self.llm_config['model'] or 'qwen-turbo',
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.1,
                    max_tokens=400,
                    **self.llm_config.get('extra_params', {})
                )
                if response.status_code == 200:
                    content = response.output.text
                else:
                    raise Exception(f"Qwen API error: {response.message}")
                    
            elif api_type == "ernie":
                # 百度文心一言
                import ernie
                response = ernie.ChatCompletion.create(
                    model=self.llm_config['model'] or 'ernie-bot',
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.1,
                    **self.llm_config.get('extra_params', {})
                )
                content = response.get_result()
                
            elif api_type == "glm":
                # 智谱AI GLM
                import zhipuai
                response = zhipuai.model_api.invoke(
                    model=self.llm_config['model'] or 'chatglm_turbo',
                    prompt=[{'role': 'user', 'content': prompt}],
                    temperature=0.1,
                    **self.llm_config.get('extra_params', {})
                )
                if response['code'] == 200:
                    content = response['data']['choices'][0]['content']
                else:
                    raise Exception(f"GLM API error: {response.get('msg', 'Unknown error')}")
                    
            elif api_type == "spark":
                # 讯飞星火 (WebSocket API)
                content = self._call_spark_api(prompt)
                
            else:
                raise ValueError(f"Unsupported API type: {api_type}")
            
        except Exception as e:
            print(f"✗ LLM taxonomy selection call failed ({api_type}): {e}")
            return None
        
        # Check if content was successfully retrieved
        if content is None:
            print(f"✗ No content received from {api_type} API for taxonomy selection")
            return None
        
        # Parse JSON response
        try:
            # Extract JSON part
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # Validate the result
                if 'selected_rank' in result and 'selected_cell_type' in result:
                    # Ensure selected_rank is within valid range
                    selected_rank = int(result['selected_rank'])
                    if 1 <= selected_rank <= len(top_taxonomy_candidates):
                        return result
                    else:
                        print(f"⚠️  LLM selected invalid taxonomy rank {selected_rank}, falling back to top similarity")
                        return None
                else:
                    print(f"⚠️  LLM taxonomy response missing required fields, falling back to top similarity")
                    return None
            else:
                print(f"⚠️  LLM taxonomy response not in JSON format, falling back to top similarity")
                return None
                
        except json.JSONDecodeError:
            print(f"⚠️  Failed to parse LLM taxonomy selection response, falling back to top similarity")
            return None
        except ValueError:
            print(f"⚠️  Invalid selected_rank in LLM taxonomy response, falling back to top similarity")
            return None
    
    def map_adata_with_expansion(self, adata, cell_name_col=None, threshold=0.5, 
                                new_col_name='cell_ontology', expand_abbreviations=True,
                                tissue_context=None, species=None, study_context=None,
                                use_llm_selection=True, llm_candidates_count=10):
        """
        🧬 Perform ontology mapping with abbreviation expansion on AnnData
        
        Parameters
        ----------
        adata : AnnData
            📊 Single-cell data object
        cell_name_col : str, optional
            📝 Column name containing cell names
        threshold : float
            📊 Similarity threshold
        new_col_name : str
            🏷️  New column name
        expand_abbreviations : bool
            🔄 Whether to enable abbreviation expansion
        tissue_context : str or list, optional
            🧬 Tissue context information, e.g. "immune system", "brain", "liver"
        species : str, optional
            🐭 Species information, e.g. "human", "mouse", "rat"
        study_context : str, optional
            🔬 Study context information, e.g. "cancer", "development", "aging"
        
        Returns
        -------
        mapping_results : dict
            📋 Mapping results
        """
        # Get cell names
        if cell_name_col is None:
            cell_names = adata.obs.index.unique().tolist()
            cell_names_series = adata.obs.index.to_series()
            print(f"📊 Using {len(cell_names)} unique cell names from index")
        else:
            cell_names = adata.obs[cell_name_col].unique().tolist()
            cell_names_series = adata.obs[cell_name_col]
            print(f"📊 Using {len(cell_names)} unique cell names from column '{cell_name_col}'")
        
        # Perform mapping with expansion
        mapping_results = self.map_cells_with_expansion(
            cell_names, threshold, expand_abbreviations,
            tissue_context=tissue_context,
            species=species,
            study_context=study_context,
            use_llm_selection=use_llm_selection, llm_candidates_count=llm_candidates_count
        )
        
        # Apply to adata
        print("\n📝 Applying mapping results to AnnData...")
        
        def get_best_match(cell_name):
            return mapping_results.get(cell_name, {}).get('best_match', 'Unknown')
        
        def get_similarity(cell_name):
            return mapping_results.get(cell_name, {}).get('similarity', 0.0)
        
        def get_confidence(cell_name):
            return mapping_results.get(cell_name, {}).get('confidence', 'low')
        
        def get_ontology_id(cell_name):
            return mapping_results.get(cell_name, {}).get('ontology_id', None)
        
        def get_cl_id(cell_name):
            return mapping_results.get(cell_name, {}).get('cl_id', None)
        
        def get_expanded_name(cell_name):
            return mapping_results.get(cell_name, {}).get('expanded_name', cell_name)
        
        def was_expanded(cell_name):
            return mapping_results.get(cell_name, {}).get('was_expanded', False)
        
        adata.obs[new_col_name] = cell_names_series.apply(get_best_match)
        adata.obs[f'{new_col_name}_similarity'] = cell_names_series.apply(get_similarity)
        adata.obs[f'{new_col_name}_confidence'] = cell_names_series.apply(get_confidence)
        adata.obs[f'{new_col_name}_ontology_id'] = cell_names_series.apply(get_ontology_id)
        adata.obs[f'{new_col_name}_cl_id'] = cell_names_series.apply(get_cl_id)
        adata.obs[f'{new_col_name}_expanded'] = cell_names_series.apply(get_expanded_name)
        adata.obs[f'{new_col_name}_was_expanded'] = cell_names_series.apply(was_expanded)
        
        # Statistics
        high_conf_count = sum(1 for r in mapping_results.values() if r['confidence'] == 'high')
        expanded_count = sum(1 for r in mapping_results.values() if r['was_expanded'])
        
        print(f"✓ Mapping completed:")
        print(f"  📊 {high_conf_count}/{len(mapping_results)} cell names have high confidence mapping")
        print(f"  🔄 {expanded_count}/{len(mapping_results)} cell names underwent abbreviation expansion")
        
        return mapping_results
    

    
    def browse_ontology_by_category(self, categories=None, max_per_category=10):
        """
        📂 Browse ontology cell types by category
        
        Parameters
        ----------
        categories : list, optional
            📝 List of category keywords to view
        max_per_category : int
            📊 Maximum number to display per category
        """
        if self.ontology_labels is None:
            raise ValueError("✗ Please load or create ontology embeddings first")
        
        if categories is None:
            categories = [
                'T cell', 'B cell', 'NK cell', 'dendritic cell', 'macrophage',
                'neutrophil', 'eosinophil', 'basophil', 'monocyte', 'lymphocyte',
                'epithelial cell', 'endothelial cell', 'fibroblast', 'neuron',
                'stem cell', 'progenitor cell', 'cancer cell', 'tumor cell'
            ]
        
        print("📂 === Browse Ontology Cell Types by Category ===\n")
        
        for category in categories:
            matches = self.search_ontology_cells(category, max_results=max_per_category)
            if matches:
                print(f"\n🏷️  【{category} related】 (Showing top {len(matches)}):")
                for i, match in enumerate(matches):
                    print(f"  {i+1}. {match}")
            print("-" * 50)

# 🛠️  Utility functions (maintaining backward compatibility)
def get_minified_adata(adata) -> AnnData:
    """📦 Return a minified AnnData."""
    adata = adata.copy()
    if hasattr(adata, 'raw') and adata.raw is not None:
        del adata.raw
    all_zeros = csr_matrix(adata.X.shape)
    X = all_zeros
    layers = {layer: all_zeros.copy() for layer in adata.layers}
    adata.X = X
    adata.layers = layers
    return adata

def majority_vote(x):
    """🗳️  Majority voting function"""
    a, b = np.unique(x, return_counts=True)
    return a[np.argmax(b)]

def majority_count(x):
    """🔢 Majority counting function"""
    a, b = np.unique(x, return_counts=True)
    return np.max(b)

@register_function(
    aliases=["下载细胞本体", "download_cl", "download_cell_ontology", "下载CL", "细胞本体下载"],
    category="single", 
    description="Download Cell Ontology (CL) JSON file from multiple reliable sources with automatic fallback",
    examples=[
        "# Basic download to default directory",
        "ov.single.download_cl()",
        "# Custom directory and filename", 
        "ov.single.download_cl(output_dir='data', filename='cell_ontology.json')",
        "# Use with CellOntologyMapper",
        "ov.single.download_cl(output_dir='ontology_data')",
        "mapper = ov.single.CellOntologyMapper(cl_obo_file='ontology_data/cl.json')",
        "# Alternative download sources available:",
        "# - Official OBO Library (primary)",
        "# - Google Drive (fallback)",
        "# - Lanzou Cloud/蓝奏云 (China fallback)"
    ],
    related=["single.CellOntologyMapper", "single.pySCSA", "single.gptcelltype"]
)
def download_cl(output_dir="new_ontology", filename="cl.json"):
    """
    📥 Download Cell Ontology file from multiple sources with automatic fallback
    
    This is a standalone function that downloads cl.json from multiple sources:
    1. Official OBO Library (direct JSON)
    2. OSS Mirror for Chinese users (ZIP format)
    3. Google Drive backup (ZIP format)
    
    Parameters
    ----------
    output_dir : str, optional
        Directory to save the file (default: "new_ontology")
    filename : str, optional
        Output filename (default: "cl.json")
        
    Returns
    -------
    success : bool
        True if download successful, False otherwise
    file_path : str or None
        Path to downloaded file if successful
    
    Examples
    --------
    >>> success, file_path = download_cl()
    >>> if success:
    ...     print(f"Downloaded to: {file_path}")
    
    >>> success, file_path = download_cl("my_data", "cell_ontology.json")
    """
    import requests
    import zipfile
    import json
    import socket
    import os
    from pathlib import Path
    
    def check_network_connection(timeout=5):
        """Check if network is available"""
        try:
            socket.create_connection(("8.8.8.8", 53), timeout)
            return True
        except OSError:
            try:
                socket.create_connection(("baidu.com", 80), timeout)
                return True
            except OSError:
                return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    # Define download sources
    sources = [
        {
            'name': 'Official OBO Library',
            'url': 'http://purl.obolibrary.org/obo/cl/cl.json',
            'is_zip': False,
            'description': 'Direct download from official Cell Ontology'
        },
        {
            'name': 'OSS Mirror (China)',
            'url': 'https://starlit.oss-cn-beijing.aliyuncs.com/single/cl.json.zip',
            'is_zip': True,
            'description': 'Fast mirror for Chinese users'
        },
        {
            'name': 'Google Drive Backup',
            'url': 'https://drive.google.com/uc?export=download&id=1niokr5INjWFVjiXHfdCoWioh0ZEYCPkv',
            'is_zip': True,
            'description': 'Google Drive backup copy'
        }
    ]
    
    print(f"Downloading Cell Ontology to: {output_path}")
    print("=" * 60)
    
    for i, source in enumerate(sources, 1):
        print(f"\n[{i}/{len(sources)}] Trying {source['name']}...")
        print(f"    URL: {source['url']}")
        print(f"    Description: {source['description']}")
        
        try:
            # Check network connectivity
            if not check_network_connection():
                print("    ✗ No network connection available")
                continue
            
            # Download file
            print("    → Downloading...")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(source['url'], headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # Determine temporary file path
            if source['is_zip']:
                temp_file = os.path.join(output_dir, f"temp_{filename}.zip")
            else:
                temp_file = output_path
            
            # Save downloaded content with tqdm progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            if total_size > 0:
                # Use tqdm progress bar
                with open(temp_file, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                            desc="    Progress", ncols=80, leave=False) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            else:
                # Fallback to simple progress display
                downloaded = 0
                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                print(f"\r    → Progress: {progress:.1f}%", end='', flush=True)
                if total_size == 0:
                    print()  # New line after progress
            
            # Get final file size for display
            actual_size = os.path.getsize(temp_file) if os.path.exists(temp_file) else 0
            print(f"    → Downloaded: {actual_size / (1024*1024):.2f} MB")
            
            # Handle zip files
            if source['is_zip']:
                print("    → Extracting ZIP file...")
                try:
                    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                        # Look for cl.json in the zip
                        json_files = [f for f in zip_ref.namelist() if f.endswith('.json')]
                        if not json_files:
                            raise ValueError("No JSON file found in ZIP archive")
                        
                        # Extract the first JSON file found
                        json_file = json_files[0]
                        print(f"    → Extracting: {json_file}")
                        
                        # Extract with progress if possible
                        # Get file info for progress
                        file_info = zip_ref.getinfo(json_file)
                        extract_size = file_info.file_size
                        
                        with zip_ref.open(json_file) as source_file, \
                             open(output_path, 'wb') as target_file:
                            
                            if extract_size > 0:
                                with tqdm(total=extract_size, unit='B', unit_scale=True,
                                        desc="    Extracting", ncols=80, leave=False) as pbar:
                                    while True:
                                        chunk = source_file.read(8192)
                                        if not chunk:
                                            break
                                        target_file.write(chunk)
                                        pbar.update(len(chunk))
                            else:
                                # No size info, just copy without progress
                                target_file.write(source_file.read())
                        
                        print(f"    → Extracted to: {output_path}")
                    
                    # Remove temporary zip file
                    os.remove(temp_file)
                    
                except Exception as e:
                    print(f"    ✗ ZIP extraction failed: {e}")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    continue
            
            # Verify the downloaded file
            if not os.path.exists(output_path):
                print("    ✗ Output file not found after download")
                continue
            
            file_size = os.path.getsize(output_path)
            if file_size < 1024:  # Less than 1KB is probably an error
                print(f"    ✗ Downloaded file too small: {file_size} bytes")
                os.remove(output_path)
                continue
            
            # Try to validate JSON structure
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Basic validation - check if it looks like an ontology file
                if 'graphs' not in data and 'nodes' not in data:
                    print("    ✗ File doesn't appear to be a valid ontology file")
                    os.remove(output_path)
                    continue
                
                print(f"    ✓ File validation successful")
                
            except Exception as e:
                print(f"    ✗ JSON validation failed: {e}")
                os.remove(output_path)
                continue
            
            print(f"    ✓ Successfully downloaded from {source['name']}")
            print(f"    File saved to: {output_path}")
            print(f"    File size: {file_size / (1024*1024):.2f} MB")
            
            return True, output_path
            
        except requests.exceptions.RequestException as e:
            print(f"    ✗ Network error: {e}")
            continue
            
        except Exception as e:
            print(f"    ✗ Download failed: {e}")
            continue

    
    
    print(f"\n✗ All download sources failed")
    print("Suggestions:")
    print("   - Check your internet connection")
    print("   - Try again later")  
    print("   - Download manually and place in the output directory")
    
    return False, None

# 🚀 ================== Usage Examples ==================
"""
💡 Examples using HF-Mirror, custom local directories, and ontology IDs:

# 1. 📥 Download Cell Ontology file (standalone function)
from omicverse.single._cellmatch import download_cl

# Basic download
success, file_path = download_cl()
if success:
    print(f"Downloaded to: {file_path}")

# Custom directory and filename
success, file_path = download_cl("my_ontology", "cell_ontology.json")

# 2. 🔧 Basic setup with custom model directory
mapper = CellOntologyMapper(
    model_name="all-mpnet-base-v2",
    local_model_dir="./my_models"  # 📁 Custom save directory
)

# 3. 🌐 Network detection and HF-Mirror download
mapper = CellOntologyMapper()
mapper.set_model(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    local_model_dir="/Users/your_username/ai_models"
)

# Manually download model
success = mapper.download_model()
if success:
    print("✓ Model ready for use!")

# 4. 🧬 Create ontology resources from downloaded file
# First download the ontology file
success, cl_file = download_cl("new_ontology")
if success:
    # Then create resources
    mapper.create_ontology_resources(cl_file, save_embeddings=True)

# Or create ontology resources from existing OBO file
mapper.create_ontology_resources("cl.obo.json", save_embeddings=True)

# 5. 🎯 Map cell names with ontology IDs
cell_names = ["NK", "TA.Early", "CD4+", "Dendritic cell"]
mapping_results = mapper.map_cells(cell_names, threshold=0.5)

# Results now include ontology IDs:
for cell_name, result in mapping_results.items():
    print(f"🔍 {cell_name}")
    print(f"  ➤ Best match: {result['best_match']}")
    print(f"  🆔 Ontology ID: {result['ontology_id']}")
    print(f"  🔢 CL ID: {result['cl_id']}")
    print(f"  📊 Similarity: {result['similarity']:.3f}")
    print(f"  🎯 Confidence: {result['confidence']}")
    print(f"  📋 Top 3 matches:")
    for i, match in enumerate(result['top3_matches'], 1):
        print(f"    {i}. {match['label']} (CL: {match['cl_id']}, Sim: {match['similarity']:.3f})")

# 5b. 📊 Load Cell Taxonomy as additional ontology (NEW FEATURE!)
# Cell Taxonomy is treated as an additional ontology source using the same NLP approach
mapper.load_cell_taxonomy_resource("Cell_Taxonomy_resource.txt")

# Load with species filter for better performance
mapper.load_cell_taxonomy_resource("Cell_Taxonomy_resource.txt", 
                                   species_filter=["Homo sapiens", "Mus musculus"])

# 5c. 🎯 Search by gene markers (taxonomy-specific feature)
nk_cells = mapper.search_by_marker("CD56", species="Homo sapiens")
t_cells = mapper.search_by_marker(["CD3", "CD4"], species="Homo sapiens")

# 5d. 🔄 Enhanced mapping with taxonomy (reuses ontology NLP approach)
enhanced_results = mapper.map_cells_with_taxonomy(
    cell_names, 
    threshold=0.5,
    expand_abbreviations=True,  # LLM expansion
    use_taxonomy=True,          # Include taxonomy
    species="Homo sapiens"
)

# Enhanced results include taxonomy information:
for cell_name, result in enhanced_results.items():
    print(f"🔍 {cell_name}")
    print(f"  🎯 Ontology match: {result['best_match']} (sim: {result['similarity']:.3f})")
    
    # NEW: Taxonomy enhancement (if found)
    if 'taxonomy_match' in result:
        print(f"  🧬 Taxonomy match: {result['taxonomy_match']} (sim: {result['taxonomy_similarity']:.3f})")
        print(f"  🆔 CT ID: {result.get('ct_id', 'N/A')}")
        print(f"  🔬 Cell marker: {result.get('cell_marker', 'N/A')}")
        
        gene_info = result.get('gene_info', {})
        if gene_info.get('gene_alias'):
            print(f"  🧬 Gene aliases: {gene_info['gene_alias']}")
        if gene_info.get('go_terms'):
            print(f"  🧬 GO terms: {gene_info['go_terms'][:100]}...")  # Truncate for display

# 6. 🤖 Setup LLM expansion with context
mapper.setup_llm_expansion(
    api_type="openai",
    api_key="your_api_key",
    tissue_context="immune system",
    species="human",
    study_context="cancer"
)

# 7. 📊 Map AnnData with abbreviation expansion and ontology IDs
mapping_results = mapper.map_adata_with_expansion(
    adata, 
    cell_name_col="cell_type",
    new_col_name="cell_ontology",
    expand_abbreviations=True,
    tissue_context="immune system"
)

# AnnData now contains these columns:
# - cell_ontology: Best match cell type
# - cell_ontology_similarity: Similarity score
# - cell_ontology_confidence: Mapping confidence
# - cell_ontology_ontology_id: Full ontology ID
# - cell_ontology_cl_id: CL ID (e.g., "CL:0000084")
# - cell_ontology_expanded: Expanded cell name (if abbreviation)
# - cell_ontology_was_expanded: Whether abbreviation expansion was performed

# 8. 💾 Save detailed results with ontology IDs
mapper.save_mapping_results(mapping_results, "cell_mapping_with_ids.csv")

# CSV will include columns:
# cell_name, best_match, similarity, confidence, ontology_id, cl_id,
# top1_match, top1_similarity, top1_ontology_id, top1_cl_id,
# top2_match, top2_similarity, top2_ontology_id, top2_cl_id,
# top3_match, top3_similarity, top3_ontology_id, top3_cl_id

# 9. 🔍 Get detailed cell information with ontology ID
cell_info = mapper.get_cell_info("Natural killer cell")
print(f"Cell: {cell_info['name']}")
print(f"Ontology ID: {cell_info.get('ontology_id', 'N/A')}")
print(f"Description: {cell_info.get('description', 'N/A')}")

# 🇨🇳 =================== 国产大模型使用示例 ===================

# 10A. 🤖 阿里云通义千问 (DashScope)
# mapper.setup_llm_expansion(
#     api_type="qwen",
#     api_key="your_dashscope_api_key",  # 或设置环境变量 DASHSCOPE_API_KEY
#     model="qwen-turbo",  # 可选: qwen-plus, qwen-max
#     tissue_context="immune system",
#     species="human",
#     study_context="cancer",
#     extra_params={"top_p": 0.8}  # 可选的额外参数
# )

# 10B. 🤖 百度文心一言 (ERNIE)
# mapper.setup_llm_expansion(
#     api_type="ernie",
#     api_key="your_ernie_access_token",  # 或 "access_key:secret_key" 格式
#     model="ernie-bot",  # 可选: ernie-bot-turbo, ernie-bot-4
#     tissue_context="brain",
#     species="mouse",
#     study_context="development"
# )

# 10C. 🤖 智谱AI GLM
# mapper.setup_llm_expansion(
#     api_type="glm",
#     api_key="your_zhipuai_api_key",  # 或设置环境变量 ZHIPUAI_API_KEY
#     model="chatglm_turbo",  # 可选: chatglm_pro, chatglm_std
#     tissue_context="liver",
#     species="human"
# )

# 10D. 🤖 讯飞星火 (iFlytek Spark)
# mapper.setup_llm_expansion(
#     api_type="spark",
#     api_key="app_id:api_key:api_secret",  # 三个参数用冒号分隔
#     model="generalv3",  # 可选: general, generalv2
#     tissue_context="lung",
#     study_context="cancer"
# )

# 10E. 🤖 字节跳动豆包 (Doubao/火山引擎)
# mapper.setup_llm_expansion(
#     api_type="doubao", 
#     api_key="your_doubao_api_key",
#     model="doubao-pro-4k",  # 或其他模型名称
#     base_url="https://ark.cn-beijing.volces.com/api/v3",  # 可选，有默认值
#     tissue_context="kidney",
#     study_context="aging"
# )

# 10F. 🤖 自定义OpenAI兼容API（如vLLM部署的模型）
# mapper.setup_llm_expansion(
#     api_type="custom_openai",
#     api_key="your_custom_api_key",  # 可选，根据API要求
#     model="your_model_name",
#     base_url="http://your-server:8000/v1",  # 必需！自定义API地址
#     tissue_context="immune system",
#     extra_params={"stop": ["\n\n"]}  # 可选的额外参数
# )

# 🌐 =================== 自定义base_url增强功能 ===================

# 11A. 🌐 使用自定义base_url的OpenAI API (如Azure OpenAI)
# mapper.setup_llm_expansion(
#     api_type="openai",
#     api_key="your_azure_api_key",
#     model="gpt-35-turbo",  # Azure模型名称
#     base_url="https://your-resource.openai.azure.com/openai/deployments/your-deployment/",
#     extra_params={"api_version": "2023-05-15"}  # Azure特定参数
# )

# 11B. 🌐 私有部署的OpenAI兼容服务
# mapper.setup_llm_expansion(
#     api_type="custom_openai",
#     api_key="sk-xxx",  # 你的私有服务API密钥
#     model="llama2-7b-chat",  # 私有服务中的模型名
#     base_url="https://your-private-llm-api.com/v1",
#     tissue_context="brain",
#     extra_params={"temperature": 0.1, "max_tokens": 500}
# )

# 11C. 🌐 使用代理的国外模型服务
# mapper.setup_llm_expansion(
#     api_type="openai",
#     api_key="your_api_key",
#     model="gpt-4",
#     base_url="https://api.openai-proxy.com/v1",  # 代理服务地址
#     tissue_context="immune system"
# )

# Features:
# ✓ Standalone download function with multiple fallback sources
# ✓ Automatic network detection
# 🪞 HF-Mirror acceleration for Chinese users  
# 📁 Custom model save directory (no default cache)
# 🔄 Automatic fallback to official HuggingFace
# 🤖 LLM-powered abbreviation expansion
# 🧬 Context-aware cell type mapping
# 🆔 Full ontology ID support (including CL numbers)
# 📊 Comprehensive mapping results with top matches
# 💾 Enhanced CSV export with all ontology information
# 📥 ZIP file handling for compressed downloads
# 🇨🇳 Chinese domestic LLM support (通义千问, 文心一言, 智谱GLM, 讯飞星火, 豆包)
# 🌐 Enhanced custom base_url support for private deployments
# 📊 NEW: Cell Taxonomy support as additional ontology source
# 🧬 NEW: Unified NLP approach for both ontology and taxonomy
# 🎯 NEW: Gene marker-based cell type search
# 🔬 NEW: Rich gene metadata integration (ENTREZ, Ensembl, GO terms)
# 📈 NEW: Confidence boosting from multiple ontology sources

# 🔧 =================== Ontology ID 问题解决方案 ===================

# 如果遇到 ontology_id 全是 None 的问题，可以使用以下方法：

# 方法1: 检查本体数据状态
# mapper.check_ontology_status()

# 方法2: 从完整的本体文件创建资源
# success, cl_file = download_cl("new_ontology")
# if success:
#     mapper.create_ontology_resources(cl_file, save_embeddings=True)

# 方法3: 单独加载本体ID映射（如果有cl_popv.json文件）
# mapper.load_ontology_mappings("new_ontology/cl_popv.json")

# 方法4: 加载embeddings后再加载映射
# mapper.load_embeddings("ontology_embeddings.pkl")
# mapper.load_ontology_mappings("cl_popv.json")  # 补充加载ID映射

# 方法5: 重新保存embeddings以包含ID映射
# # 先加载完整数据
# mapper.create_ontology_resources("cl.json")
# # 重新保存embeddings（现在包含ID映射）
# mapper.save_embeddings()
"""
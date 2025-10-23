import re
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (Python GEO-to-SRA fetcher)"}

# ---------- 1) 抓取 GEO SOFT 文本 ----------
def fetch_geo_text(accession: str, timeout: int = 120) -> str:
    """
    读取 GEO Series/GSM 的 SOFT 文本视图（包含最完整的 Series_* 字段）
    e.g. https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE157103&form=text&view=full
    """
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}&form=text&view=full"
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text

# ---------- 2) 解析为结构化数据 ----------
PRJNA_RE = re.compile(r"\bPRJNA\d+\b", re.I)
SRP_RE   = re.compile(r"\bSRP\d+\b", re.I)
GSM_RE   = re.compile(r"\bGSM\d+\b", re.I)

def _append(d: Dict[str, Any], key: str, val: str):
    """将重复键（如 Series_sample_id）收集为 list。"""
    if key not in d:
        d[key] = val
    else:
        if not isinstance(d[key], list):
            d[key] = [d[key]]
        d[key].append(val)

def parse_geo_soft_to_struct(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    解析 SOFT 文本中所有以 '!Series_' 开头的键值，并抽取关键信息。
    返回 (series_all, extracted)
    - series_all: 所有 !Series_* 的完整键值（重复键变 list）
    - extracted: 你关心的字段，规范化后的汇总
    """
    series_all: Dict[str, Any] = {}

    # 逐行解析 Series_* 的键值（如：!Series_platform_id = GPLxxxx）
    for ln in text.splitlines():
        if ln.startswith("!Series_"):
            # 例：!Series_sample_id = GSM123456
            parts = ln.split("=", 1)
            if len(parts) == 2:
                key = parts[0].lstrip("!").strip()   # 'Series_sample_id'
                val = parts[1].strip()
                _append(series_all, key, val)

    # 从 Series_relation 与全文中提取 PRJNA / SRP
    relations = series_all.get("Series_relation", [])
    if isinstance(relations, str):
        relations = [relations]

    prjna_candidates: List[str] = []
    srp_candidates: List[str] = []

    # 1) 从 relation 行里抓
    for item in relations:
        prjna_candidates += PRJNA_RE.findall(item)
        srp_candidates   += SRP_RE.findall(item)

    # 2) 全文兜底再抓一遍
    if not prjna_candidates:
        prjna_candidates += PRJNA_RE.findall(text)
    if not srp_candidates:
        srp_candidates   += SRP_RE.findall(text)

    # 去重、规范大写
    def _uniq_upper(xs): 
        seen, out = set(), []
        for x in xs:
            x = x.upper()
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    prjna_list = _uniq_upper(prjna_candidates)
    srp_list   = _uniq_upper(srp_candidates)

    # Series 号
    geo_acc = None
    for ln in text.splitlines():
        if ln.startswith("!Series_geo_accession"):
            # !Series_geo_accession = GSE157103
            geo_acc = ln.split("=", 1)[1].strip()
            break

    # 所有样本 GSM（有些 Series 不在 !Series_sample_id 列出，也兜底从全文扫描）
    sample_ids: List[str] = []
    ssid = series_all.get("Series_sample_id", [])
    if isinstance(ssid, str):
        ssid = [ssid]
    sample_ids += ssid
    sample_ids += GSM_RE.findall(text)
    sample_ids = _uniq_upper(sample_ids)

    # 物种与 taxid（Series 级汇总）
    # 允许重复、可能多物种；规范为不重复的列表
    organism_vals = series_all.get("Series_sample_organism", [])
    if isinstance(organism_vals, str):
        organism_vals = [organism_vals]
    organism_vals = [v.strip() for v in organism_vals]

    taxid_vals = series_all.get("Series_sample_taxid", [])
    if isinstance(taxid_vals, str):
        taxid_vals = [taxid_vals]
    taxid_vals = [v.strip() for v in taxid_vals]

    # 组装 extracted
    extracted = {
        "geo_acc": geo_acc,
        "BioProject": prjna_list[0] if prjna_list else None,
        "BioProject_all": prjna_list or [],
        "SRAnum": srp_list[0] if srp_list else None,
        "SRAnum_all": srp_list or [],
        "sample_ids": sample_ids,
        "sample_organism": sorted(list({x for x in organism_vals if x})),
        "sample_taxid": sorted(list({x for x in taxid_vals if x})),
    }

    return series_all, extracted

# ---------- 3) 一键函数：从 GSE 拉取并写 JSON ----------
def geo_accession_to_meta_json(accession: str, out_dir: str | Path = ".") -> Path:
    """
    给 GSE/GSM accession，拉取 SOFT 文本，解析出：
      - geo_acc、BioProject(PRJNA)、SRAnum(SRP)
      - sample_ids（列表）
      - sample_organism（去重列表）
      - sample_taxid（去重列表）
    同时把所有 !Series_* 键值以字典形式完整保存。
    输出：{accession}_meta.json
    """
    out_path = Path(out_dir) / f"{accession}_meta.json"
    if out_path.exists():
        print("SKIPPED (already exists)")
        return out_path
    else:
        text = fetch_geo_text(accession)
        series_all, extracted = parse_geo_soft_to_struct(text)
    
        payload = {
            "accession": accession,
            "extracted": extracted,
            "series_all": series_all  # 保留“所有信息”的结构化版本（重复键已合并为 list）
        }
    
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

'''
How to use
# 1) 直接生成 meta JSON（包含完整 !Series_* 字段与提取的关键字段）
p = geo_accession_to_meta_json("GSE157103", out_dir="meta")
print("Saved:", p)

# 2) 读取并取出 PRJNA / SRP，用于后续 EDirect:
import json
d = json.loads(Path(p).read_text(encoding="utf-8"))
prjna = d["extracted"]["BioProject"] or (d["extracted"]["BioProject_all"][0] if d["extracted"]["BioProject_all"] else None)
srp   = d["extracted"]["SRA"] or (d["extracted"]["SRAnum_all"][0] if d["extracted"]["SRAnum_all"] else None)
print("PRJNA:", prjna, " SRP:", srp)
'''
"""LLM-assisted scoping: split objectives, expand queries, add constraints.

Uses an OpenAI-compatible chat-completions endpoint if available; otherwise
falls back to a simple heuristic splitter.
"""

from __future__ import annotations

from typing import Dict, List
import os


def suggest_brief(request: str) -> Dict[str, List[str]]:
    """Return {objectives: [...], constraints: [...]} derived from request.

    Constraints may include tokens like:
    - date:>=2022
    - domain:pubmed|ncbi|arxiv
    - modality:scRNA-seq|scATAC-seq
    """
    try:
        import requests, json

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.getenv("OPENAI_MODEL", "gpt-5")
        if not api_key:
            raise RuntimeError("no key")
        system = (
            "You split a research request into concrete objectives and constraints. "
            "Return STRICT JSON: {\"objectives\":[..], \"constraints\":[..]}. "
            "Constraints should include optional date filters (e.g., 'date:>=2022') and domain hints (e.g., 'domain:pubmed|arxiv')."
        )
        user = f"Request: {request}\nReturn JSON only."
        payload = {"model": model, "messages": [{"role":"system","content":system},{"role":"user","content":user}], "temperature": 0}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        resp = requests.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "{}")
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("bad format")
        return {"objectives": list(map(str, data.get("objectives", []))), "constraints": list(map(str, data.get("constraints", [])))}
    except Exception:
        # heuristic fallback
        parts = [p.strip() for p in request.replace(";", ",").split(",") if p.strip()]
        return {"objectives": parts[:3], "constraints": []}


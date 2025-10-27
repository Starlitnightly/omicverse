from __future__ import annotations

"""
Minimal GitHub repository content extractor using PyGithub if available.

Collects README (if present) and basic repo metadata into markdown files.
"""

from typing import List, Tuple


def extract(repo_slug: str) -> List[Tuple[str, str]]:
    """Extracts README and metadata for a GitHub repo, e.g., 'owner/name'.

    Honors anonymous access; will use GITHUB_TOKEN from environment if set.
    Returns a list of (filename, markdown_content).
    """
    try:
        from github import Github  # PyGithub
        import os
        import base64
    except Exception as exc:  # pragma: no cover - optional
        raise RuntimeError("GitHub extraction requires PyGithub") from exc

    token = os.environ.get("GITHUB_TOKEN")
    gh = Github(token) if token else Github()
    repo = gh.get_repo(repo_slug)

    results: List[Tuple[str, str]] = []

    # README
    try:
        readme = repo.get_readme()
        content = base64.b64decode(readme.content or b"").decode("utf-8", errors="replace")
        results.append(("github-readme.md", f"# README for {repo_slug}\n\n" + content))
    except Exception:
        pass

    # Basic metadata
    meta = {
        "full_name": repo.full_name,
        "description": repo.description or "",
        "stars": repo.stargazers_count,
        "forks": repo.forks_count,
        "open_issues": repo.open_issues_count,
        "default_branch": repo.default_branch,
        "topics": ", ".join(getattr(repo, "get_topics", lambda: [])() or []),
        "html_url": repo.html_url,
    }
    meta_md = ["# Repository Metadata", "", f"Repository: {meta['full_name']}"]
    for k, v in meta.items():
        meta_md.append(f"- {k}: {v}")
    results.append(("github-metadata.md", "\n".join(meta_md)))

    return results


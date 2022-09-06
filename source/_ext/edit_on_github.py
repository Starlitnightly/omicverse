"""Based on: gist.github.com/MantasVaitkunas/7c16de233812adcb7028"""

import os
import warnings

__licence__ = "BSD (3 clause)"

from typing import Any, Dict, Tuple, Optional

from sphinx.addnodes import document
from sphinx.application import Sphinx


def get_github_repo(app: Sphinx, path: str) -> Tuple[str, str]:
    if path.endswith(".ipynb"):
        return app.config.github_nb_repo, "/tutorials/"
    if path.endswith(".rst") and "auto_examples" in path and "index.rst" not in path:
        # TODO(michalk8): ugly trick to point .rst files in `auto_examples` to `.py`
        path = (
            path.replace("auto_examples", "examples").replace(".rst", ".py") + "#FIXME/"
        )
        if not path.startswith("/"):
            path = "/" + path
        return app.config.github_repo, path
    return app.config.github_repo, "/docs/source/"


def html_page_context(
    app: Sphinx,
    pagename: str,
    templatename: str,
    context: Dict[str, Any],
    doctree: Optional[document],
) -> Optional[str]:
    if doctree is None:
        return
    if templatename != "page.html":
        return
    if not app.config.github_repo:
        warnings.warn("`github_repo `not specified")
        return

    if not app.config.github_nb_repo:
        nb_repo = f"{app.config.github_repo}_notebooks"
        app.config.github_nb_repo = nb_repo

    path = os.path.relpath(doctree.get("source"), app.builder.srcdir)
    repo, conf_py_path = get_github_repo(app, path)

    # For sphinx_rtd_theme.
    context["display_github"] = True
    context["github_user"] = "theislab"
    # TODO(michalk): master/dev
    context["github_version"] = "master"
    context["github_repo"] = repo
    context["conf_py_path"] = conf_py_path


def setup(app: Sphinx) -> None:
    app.add_config_value("github_nb_repo", "", True)
    app.add_config_value("github_repo", "", True)
    app.connect("html-page-context", html_page_context)
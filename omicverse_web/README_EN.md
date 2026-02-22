<h1 align="center">OmicVerse Web</h1>

<div align="center">
  <strong>English</strong> | <a href="README_CN.md">中文</a>
</div>

## 1. Introduction

OmicVerse Web is the browser interface of OmicVerse, built with Flask backend and HTML/CSS/JavaScript frontend.

Current capabilities include:

- Landing page and documentation entry
- Single-cell analysis page (clustering, trajectory, DEG, annotation)
- File manager, notebook, terminal, kernel execution
- Standard plotting and GPU plotting endpoints
- Environment management (pip/conda) and agent endpoint

## 2. Directory Structure

```text
omicverse_web/
├── app.py                                # Main Flask app and core APIs
├── index.html                            # Landing page (route /)
├── single_cell_analysis_standalone.html  # Analysis page (route /analysis)
├── start_server.py                       # Launcher
├── routes/                               # Blueprint routes
│   ├── data.py
│   ├── files.py
│   ├── kernel.py
│   ├── notebooks.py
│   └── terminal.py
├── services/                             # Service layer (Agent, Kernel)
├── utils/                                # Utilities
├── static/                               # Static assets (css/js/font/picture)
├── server/                               # Shared backend modules
├── models/                               # Model assets
├── data/                                 # Data directory
├── temp/                                 # Temporary directory
├── fbs/                                  # FlatBuffers schema
├── dist/                                 # Build artifacts
├── pyproject.toml                        # Dependencies and packaging config
├── LICENSE
└── README.md
```

## 3. Quick Start

Requirements:

- Python >= 3.8

Install:

```bash
cd omicverse_web
pip install -e .
```

Start (recommended):

```bash
python3 start_server.py
```

or:

```bash
python3 app.py
```

Access:

- Landing page: `http://localhost:5050/`
- Analysis page: `http://localhost:5050/analysis`

## 4. Main Routes and APIs (selected)

Page routes:

- `GET /`
- `GET /analysis`
- `GET /legacy`

Core APIs (selected):

- `POST /api/execute_code`
- `POST /api/execute_code_stream`
- `GET /api/status`
- `POST /api/plot`
- `POST /api/plot_gpu`
- `POST /api/agent/run`

Registered Blueprint prefixes:

- `/api/kernel`
- `/api/files`
- `/api` (data)
- `/api/notebooks`
- `/api/terminal`

## 5. Development Notes

Frontend:

- Landing page: `index.html`
- Analysis page: `single_cell_analysis_standalone.html`
- Static assets: `static/css/`, `static/js/`

Backend:

- Main app: `app.py`
- Sub-routes: `routes/`
- Services: `services/`

Script entry (`pyproject.toml`):

- `omicverse-web = omicverse_web.start_server:main`

## 6. Citation

If you use OmicVerse in your research, please cite:

> OmicVerse: a framework for bridging and deepening insights across bulk and single-cell sequencing  
> Zeng, Z., Ma, Y., Hu, L. et al.  
> Nature Communications (2024), 15:5983.  
> DOI: https://doi.org/10.1038/s41467-024-50194-3

## 7. License

This directory is licensed under GNU General Public License v3.0 (GPL-3.0).

Please refer to `omicverse_web/LICENSE`.

<h1 align="center">OmicVerse Web</h1>

<div align="center">
  <a href="README_EN.md">English</a> | <strong>中文</strong>
</div>

## 1. 简介

OmicVerse Web 是 OmicVerse 的浏览器端界面，后端基于 Flask，前端为 HTML/CSS/JavaScript。

当前能力包括：

- 首页展示与文档入口
- 单细胞分析页面（聚类、轨迹、差异分析、注释等）
- 文件管理、Notebook、终端、内核执行
- 常规绘图与 GPU 绘图接口
- 环境管理（pip/conda）与 Agent 接口

## 2. 目录结构

```text
omicverse_web/
├── app.py                                # Flask 主应用与核心 API
├── index.html                            # 首页（路由 /）
├── single_cell_analysis_standalone.html  # 分析页面（路由 /analysis）
├── start_server.py                       # 启动脚本
├── routes/                               # Blueprint 路由
│   ├── data.py
│   ├── files.py
│   ├── kernel.py
│   ├── notebooks.py
│   └── terminal.py
├── services/                             # 服务层（Agent、Kernel）
├── utils/                                # 工具函数
├── static/                               # 静态资源（css/js/font/picture）
├── server/                               # 服务端通用模块
├── models/                               # 模型资源
├── data/                                 # 数据目录
├── temp/                                 # 临时目录
├── fbs/                                  # FlatBuffers schema
├── dist/                                 # 打包产物
├── pyproject.toml                        # 依赖与打包配置
├── LICENSE
└── README.md
```

## 3. 快速开始

环境要求：

- Python >= 3.8

安装：

```bash
cd omicverse_web
pip install -e .
```

启动（推荐）：

```bash
python3 start_server.py
```

或：

```bash
python3 app.py
```

访问地址：

- 首页: `http://localhost:5050/`
- 分析页: `http://localhost:5050/analysis`

## 4. 主要路由与 API（节选）

页面路由：

- `GET /`
- `GET /analysis`
- `GET /legacy`

核心 API（节选）：

- `POST /api/execute_code`
- `POST /api/execute_code_stream`
- `GET /api/status`
- `POST /api/plot`
- `POST /api/plot_gpu`
- `POST /api/agent/run`

注册的 Blueprint 前缀：

- `/api/kernel`
- `/api/files`
- `/api`（data）
- `/api/notebooks`
- `/api/terminal`

## 5. 开发说明

前端：

- 首页：`index.html`
- 分析页：`single_cell_analysis_standalone.html`
- 静态资源：`static/css/`、`static/js/`

后端：

- 主入口：`app.py`
- 子路由：`routes/`
- 服务层：`services/`

脚本入口（`pyproject.toml`）：

- `omicverse-web = omicverse_web.start_server:main`

## 6. 引用

如果你在研究中使用了 OmicVerse，请引用：

> OmicVerse: a framework for bridging and deepening insights across bulk and single-cell sequencing  
> Zeng, Z., Ma, Y., Hu, L. et al.  
> Nature Communications (2024), 15:5983.  
> DOI: https://doi.org/10.1038/s41467-024-50194-3

## 7. 许可证

本目录采用 GNU General Public License v3.0（GPL-3.0）。

请以 `omicverse_web/LICENSE` 为准。

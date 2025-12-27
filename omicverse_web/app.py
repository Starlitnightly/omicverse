from flask import Flask, request, jsonify, send_file, send_from_directory, make_response
from flask_cors import CORS
import scanpy as sc
import numpy as np
import pandas as pd
import os
import tempfile
from werkzeug.utils import secure_filename
import warnings
import json
import logging
from http import HTTPStatus
from concurrent.futures import ThreadPoolExecutor
from werkzeug.exceptions import RequestEntityTooLarge
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import threading
import base64
import io
import traceback
import asyncio

from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import sys
import mimetypes
import shutil

# Import our high-performance data adaptor
from server.data_adaptor.anndata_adaptor import HighPerformanceAnndataAdaptor



warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

# Configuration
# No upload size limit - explicitly set to None to satisfy Flask's accessor
app.config['MAX_CONTENT_LENGTH'] = None
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

# Global variables to store current data and adaptor
current_adaptor = None
current_adata = None
current_filename = None
thread_pool = ThreadPoolExecutor(max_workers=4)
kernel_lock = threading.Lock()


class InProcessKernelExecutor:
    """Lightweight in-process ipykernel executor for shared state."""
    def __init__(self):
        self.kernel_manager = None
        self.shell = None

    def _ensure_kernel(self):
        if self.kernel_manager is not None:
            return
        try:
            from ipykernel.inprocess import InProcessKernelManager
        except ImportError as exc:
            raise RuntimeError('ipykernel is required for code execution') from exc
        self.kernel_manager = InProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.shell = self.kernel_manager.kernel.shell
        plt.switch_backend('Agg')
        self.shell.user_ns.update({
            'sc': sc,
            'pd': pd,
            'np': np,
            'plt': plt,
        })

    def restart(self):
        with kernel_lock:
            if self.kernel_manager is not None:
                try:
                    self.kernel_manager.shutdown_kernel(now=True)
                except Exception:
                    pass
            self.kernel_manager = None
            self.shell = None
            self._ensure_kernel()

    def sync_adata(self, adata):
        self._ensure_kernel()
        self.shell.user_ns['adata'] = adata

    def execute(self, code, adata=None, user_ns=None):
        self._ensure_kernel()
        with kernel_lock:
            original_ns = self.shell.user_ns
            if user_ns is not None:
                self.shell.user_ns = user_ns
            if adata is not None:
                self.shell.user_ns['adata'] = adata
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            before_figs = set(plt.get_fignums())
            try:
                result = None
                error_msg = None
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    result = self.shell.run_cell(code, store_history=True)
                if result.error_before_exec or result.error_in_exec:
                    err = result.error_before_exec or result.error_in_exec
                    error_msg = ''.join(traceback.format_exception(err.__class__, err, err.__traceback__))
                output = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue()

                figures = []
                after_figs = set(plt.get_fignums())
                new_figs = [num for num in after_figs if num not in before_figs]
                for fig_num in new_figs:
                    fig = plt.figure(fig_num)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight')
                    figures.append(base64.b64encode(buf.getvalue()).decode('ascii'))
                    plt.close(fig)

                last_result = result.result if result else None
                adata_value = self.shell.user_ns.get('adata')
                return {
                    'output': output,
                    'stderr': stderr,
                    'error': error_msg,
                    'result': last_result,
                    'figures': figures,
                    'adata': adata_value
                }
            finally:
                if user_ns is not None:
                    self.shell.user_ns = original_ns


kernel_executor = InProcessKernelExecutor()
current_kernel_name = 'python3'
kernel_names = {}
kernel_sessions = {}
agent_instance = None
agent_config_signature = None
notebook_root = os.getcwd()
file_root = Path(notebook_root).resolve()


def normalize_kernel_id(kernel_id):
    if not kernel_id:
        return 'default.ipynb'
    return str(kernel_id)

def build_kernel_namespace(include_adata=False):
    namespace = {
        'sc': sc,
        'pd': pd,
        'np': np,
        'plt': plt,
    }
    if include_adata:
        namespace['adata'] = current_adata
    return namespace


def reset_kernel_namespace(kernel_id):
    kernel_id = normalize_kernel_id(kernel_id)
    if kernel_id == 'default.ipynb':
        kernel_executor.restart()
        if current_adata is not None:
            kernel_executor.sync_adata(current_adata)
        return
    kernel_sessions[kernel_id] = {
        'user_ns': build_kernel_namespace()
    }


def get_agent_instance(config):
    import omicverse as ov
    global agent_instance, agent_config_signature
    if config is None:
        config = {}
    signature_payload = {
        'model': config.get('model') or 'gpt-5',
        'api_key': config.get('apiKey') or '',
        'endpoint': config.get('apiBase') or None,
    }
    signature = json.dumps(signature_payload, sort_keys=True)
    if agent_instance is None or signature != agent_config_signature:
        agent_instance = ov.Agent(
            model=signature_payload['model'],
            api_key=signature_payload['api_key'] or None,
            endpoint=signature_payload['endpoint'] or None,
            use_notebook_execution=False
        )
        agent_config_signature = signature
    return agent_instance


def run_agent_stream(agent, prompt, adata):
    async def _runner():
        code = None
        result_adata = None
        result_shape = None
        llm_text = ''
        async for event in agent.stream_async(prompt, adata):
            if event.get('type') == 'llm_chunk':
                llm_text += event.get('content', '')
            elif event.get('type') == 'code':
                code = event.get('content')
            elif event.get('type') == 'result':
                result_adata = event.get('content')
                result_shape = event.get('shape')
            elif event.get('type') == 'error':
                raise RuntimeError(event.get('content', 'Agent error'))
        return {
            'code': code,
            'llm_text': llm_text,
            'result_adata': result_adata,
            'result_shape': result_shape
        }

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        result_container = {}
        error_container = {}

        def _run_in_thread():
            try:
                result_container['value'] = asyncio.run(_runner())
            except BaseException as exc:
                error_container['error'] = exc

        thread = threading.Thread(target=_run_in_thread, name='OmicVerseAgentRunner')
        thread.start()
        thread.join()
        if 'error' in error_container:
            raise error_container['error']
        return result_container.get('value')

    return asyncio.run(_runner())


def run_agent_chat(agent, prompt):
    async def _runner():
        if not agent._llm:
            raise RuntimeError("LLM backend is not initialized")
        chat_prompt = (
            "You are an OmicVerse assistant. Answer in natural language only, "
            "avoid code unless explicitly requested.\n\nUser: " + prompt
        )
        return await agent._llm.run(chat_prompt)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        result_container = {}
        error_container = {}

        def _run_in_thread():
            try:
                result_container['value'] = asyncio.run(_runner())
            except BaseException as exc:
                error_container['error'] = exc

        thread = threading.Thread(target=_run_in_thread, name='OmicVerseAgentChatRunner')
        thread.start()
        thread.join()
        if 'error' in error_container:
            raise error_container['error']
        return result_container.get('value')

    return asyncio.run(_runner())


def agent_requires_adata(prompt):
    if not prompt:
        return False
    lowered = prompt.lower()
    keywords = [
        'adata', 'qc', 'quality', 'cluster', 'clustering', 'umap', 'tsne', 'pca',
        'embedding', 'neighbors', 'leiden', 'louvain', 'marker', 'differential',
        'hvg', 'highly variable', 'preprocess', 'normalize', 'visualize', 'plot',
        '降维', '聚类', '可视化', '差异', '标记', '质控', '预处理'
    ]
    return any(keyword in lowered for keyword in keywords)


def get_kernel_context(kernel_id):
    kernel_id = normalize_kernel_id(kernel_id)
    if kernel_id == 'default.ipynb':
        kernel_executor._ensure_kernel()
        return kernel_executor, kernel_executor.shell.user_ns
    session = kernel_sessions.get(kernel_id)
    if session is None:
        session = {
            'user_ns': build_kernel_namespace()
        }
        kernel_sessions[kernel_id] = session
    return kernel_executor, session['user_ns']


def ensure_default_notebook():
    default_path = file_root / 'default.ipynb'
    if default_path.exists():
        return
    try:
        try:
            import nbformat
            nb = nbformat.v4.new_notebook()
            nb.cells = [nbformat.v4.new_code_cell(source='')]
            with open(default_path, 'w', encoding='utf-8') as handle:
                nbformat.write(nb, handle)
        except ImportError:
            minimal = {
                "cells": [{
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": []
                }],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5
            }
            with open(default_path, 'w', encoding='utf-8') as handle:
                handle.write(json.dumps(minimal))
    except Exception as e:
        logging.error(f"Default notebook creation failed: {e}")


ensure_default_notebook()


def resolve_browse_path(rel_path):
    rel_path = rel_path or ''
    target = (file_root / rel_path).resolve()
    if target != file_root and file_root not in target.parents:
        raise ValueError('Invalid path')
    return target


def is_allowed_text_file(path_obj):
    allowed = {
        '.txt', '.py', '.md', '.json', '.csv', '.tsv', '.yaml', '.yml', '.log',
        '.ini', '.toml', '.js', '.css', '.html'
    }
    return path_obj.suffix.lower() in allowed


def is_image_file(path_obj):
    return path_obj.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp'}


def estimate_var_size(obj):
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.nbytes
    except Exception:
        pass
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
            return int(obj.memory_usage(deep=True).sum())
    except Exception:
        pass
    try:
        if obj.__class__.__name__ == 'AnnData':
            size = 0
            try:
                size += obj.X.data.nbytes if hasattr(obj.X, 'data') else obj.X.nbytes
            except Exception:
                pass
            try:
                size += int(obj.obs.memory_usage(deep=True).sum())
            except Exception:
                pass
            try:
                size += int(obj.var.memory_usage(deep=True).sum())
            except Exception:
                pass
            return size
    except Exception:
        pass
    try:
        return sys.getsizeof(obj)
    except Exception:
        return 0


def get_process_memory_mb():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if rss > 10**7:
            return rss / (1024 * 1024)
        return rss / 1024
    except Exception:
        return None


def summarize_var(name, value):
    summary = {
        'name': name,
        'type': type(value).__name__,
        'preview': ''
    }
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            summary['preview'] = f'ndarray shape={value.shape} dtype={value.dtype}'
            return summary
    except Exception:
        pass
    try:
        import pandas as pd
        if isinstance(value, pd.DataFrame):
            summary['preview'] = f'DataFrame shape={value.shape}'
            return summary
        if isinstance(value, pd.Series):
            summary['preview'] = f'Series len={len(value)} dtype={value.dtype}'
            return summary
    except Exception:
        pass
    try:
        if value.__class__.__name__ == 'AnnData':
            shape = getattr(value, 'shape', None)
            summary['preview'] = f'AnnData shape={shape}'
            return summary
    except Exception:
        pass
    try:
        preview = repr(value)
        preview = preview.replace('\n', ' ')
        summary['preview'] = preview[:160]
        return summary
    except Exception:
        summary['preview'] = '<unavailable>'
        return summary


def resolve_var_path(name, ns):
    if not name or name.startswith('_') or '__' in name:
        raise KeyError('Invalid variable name')
    parts = name.split('.')
    if parts[0] not in ns:
        raise KeyError('Variable not found')
    obj = ns[parts[0]]
    for part in parts[1:]:
        if part in ('obs', 'var', 'uns', 'obsm', 'layers', 'X'):
            obj = getattr(obj, part)
        else:
            raise KeyError('Unsupported attribute')
    return obj


def sync_adaptor_with_adata():
    """Keep adaptor in sync after kernel or tool updates."""
    if current_adaptor is None or current_adata is None:
        return
    try:
        current_adaptor.adata = current_adata
        current_adaptor.n_obs = current_adata.n_obs
        current_adaptor.n_vars = current_adata.n_vars
        current_adaptor._build_indexes()
    except Exception:
        pass


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    max_bytes = app.config.get('MAX_CONTENT_LENGTH')
    max_mb = int(max_bytes / (1024 * 1024)) if isinstance(max_bytes, (int, float)) and max_bytes else None
    payload = {'error': 'File too large'}
    if max_mb:
        payload['max_size_mb'] = max_mb
    return jsonify(payload), 413

@app.route('/api/upload', methods=['POST'])
def upload_file():
    global current_adaptor, current_adata, current_filename

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.endswith('.h5ad'):
        return jsonify({'error': 'File must be .h5ad format'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Create high-performance data adaptor
        current_adaptor = HighPerformanceAnndataAdaptor(filepath)
        # Expose underlying AnnData for tool endpoints
        current_adata = current_adaptor.adata
        current_filename = filename
        try:
            kernel_executor.sync_adata(current_adata)
        except Exception:
            pass

        # Get schema and summary
        schema = current_adaptor.get_schema()
        summary = current_adaptor.get_data_summary()
        chunk_info = current_adaptor.get_chunk_info()

        # Build response compatible with legacy UI expectations
        response_data = {
            'filename': filename,
            # Legacy/UI fields used by single-cell.js
            'n_cells': summary.get('n_obs', current_adaptor.n_obs),
            'n_genes': summary.get('n_vars', current_adaptor.n_vars),
            'embeddings': summary.get('embeddings', []),
            'obs_columns': list(current_adaptor.adata.obs.columns),
            'var_columns': list(current_adaptor.adata.var.columns),
            # New high-performance metadata
            'schema': schema,
            'chunk_info': chunk_info,
            'summary': summary,
            'success': True
        }

        # If no embeddings available, provide a synthetic 'random' embedding to avoid empty canvas
        if not response_data['embeddings']:
            response_data['embeddings'] = ['random']

        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Upload failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/notebooks/upload', methods=['POST'])
def upload_notebook():
    file = request.files.get('file')
    if file is None or file.filename == '':
        return jsonify({'error': 'No file provided'}), 400
    if not file.filename.endswith('.ipynb'):
        return jsonify({'error': 'File must be .ipynb format'}), 400

    try:
        try:
            import nbformat
        except ImportError:
            return jsonify({'error': '需要安装 nbformat 才能导入 .ipynb 文件'}), 400

        raw = file.read().decode('utf-8', errors='ignore')
        nb = nbformat.reads(raw, as_version=4)
        cells = []
        for cell in nb.cells:
            if cell.cell_type not in ('code', 'markdown'):
                continue
            source = cell.source
            if isinstance(source, list):
                source = ''.join(source)
            outputs = []
            if cell.cell_type == 'code':
                for output in cell.get('outputs', []):
                    outputs.append(output)
            cells.append({
                'cell_type': cell.cell_type,
                'source': source,
                'outputs': outputs
            })

        return jsonify({
            'filename': secure_filename(file.filename),
            'cells': cells
        })
    except Exception as e:
        logging.error(f"Notebook import failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/notebooks/list', methods=['GET'])
def list_notebooks():
    try:
        files = []
        for name in os.listdir(notebook_root):
            if name.endswith('.ipynb') and os.path.isfile(os.path.join(notebook_root, name)):
                files.append(name)
        files.sort()
        return jsonify({'files': files, 'root': notebook_root})
    except Exception as e:
        logging.error(f"Notebook list failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/notebooks/open', methods=['POST'])
def open_notebook():
    data = request.json if request.json else {}
    filename = data.get('filename', '')
    if not filename or not filename.endswith('.ipynb'):
        return jsonify({'error': 'Invalid filename'}), 400
    safe_name = secure_filename(filename)
    notebook_path = os.path.abspath(os.path.join(notebook_root, safe_name))
    if not notebook_path.startswith(os.path.abspath(notebook_root) + os.sep):
        return jsonify({'error': 'Invalid path'}), 400
    if not os.path.exists(notebook_path):
        return jsonify({'error': 'Notebook not found'}), 404

    try:
        try:
            import nbformat
        except ImportError:
            return jsonify({'error': '需要安装 nbformat 才能导入 .ipynb 文件'}), 400

        with open(notebook_path, 'r', encoding='utf-8', errors='ignore') as handle:
            nb = nbformat.read(handle, as_version=4)
        cells = []
        for cell in nb.cells:
            if cell.cell_type not in ('code', 'markdown'):
                continue
            source = cell.source
            if isinstance(source, list):
                source = ''.join(source)
            outputs = []
            if cell.cell_type == 'code':
                for output in cell.get('outputs', []):
                    outputs.append(output)
            cells.append({
                'cell_type': cell.cell_type,
                'source': source,
                'outputs': outputs
            })
        return jsonify({
            'filename': os.path.basename(notebook_path),
            'cells': cells
        })
    except Exception as e:
        logging.error(f"Notebook open failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/files/list', methods=['GET'])
def list_files():
    rel_path = request.args.get('path', '')
    try:
        target = resolve_browse_path(rel_path)
    except ValueError:
        return jsonify({'error': 'Invalid path'}), 400

    if not target.exists() or not target.is_dir():
        return jsonify({'error': 'Directory not found'}), 404

    entries = []
    for entry in target.iterdir():
        try:
            item = {
                'name': entry.name,
                'type': 'dir' if entry.is_dir() else 'file',
                'size': entry.stat().st_size if entry.is_file() else None,
                'ext': entry.suffix.lower() if entry.is_file() else None
            }
            entries.append(item)
        except Exception:
            continue

    entries.sort(key=lambda x: (0 if x['type'] == 'dir' else 1, x['name'].lower()))
    rel = '' if target == file_root else str(target.relative_to(file_root))
    parent = '' if target == file_root else str(target.parent.relative_to(file_root))
    return jsonify({'path': rel, 'parent': parent, 'entries': entries})


@app.route('/api/files/open', methods=['POST'])
def open_file():
    data = request.json if request.json else {}
    rel_path = data.get('path', '')
    try:
        target = resolve_browse_path(rel_path)
    except ValueError:
        return jsonify({'error': 'Invalid path'}), 400

    if not target.exists() or not target.is_file():
        return jsonify({'error': 'File not found'}), 404

    try:
        if target.suffix.lower() == '.ipynb':
            try:
                import nbformat
            except ImportError:
                return jsonify({'error': '需要安装 nbformat 才能导入 .ipynb 文件'}), 400
            with open(target, 'r', encoding='utf-8', errors='ignore') as handle:
                raw = handle.read()
            if not raw.strip():
                nb = nbformat.v4.new_notebook()
                nb.cells = [nbformat.v4.new_code_cell(source='')]
                with open(target, 'w', encoding='utf-8') as handle:
                    nbformat.write(nb, handle)
            else:
                nb = nbformat.reads(raw, as_version=4)
            cells = []
            for cell in nb.cells:
                if cell.cell_type not in ('code', 'markdown'):
                    continue
                source = cell.source
                if isinstance(source, list):
                    source = ''.join(source)
                outputs = []
                if cell.cell_type == 'code':
                    for output in cell.get('outputs', []):
                        outputs.append(output)
                cells.append({
                    'cell_type': cell.cell_type,
                    'source': source,
                    'outputs': outputs
                })
            return jsonify({
                'type': 'notebook',
                'name': target.name,
                'path': str(target.relative_to(file_root)),
                'cells': cells
            })

        if not is_allowed_text_file(target):
            if is_image_file(target):
                mime_type = mimetypes.guess_type(target.name)[0] or 'image/png'
                with open(target, 'rb') as handle:
                    encoded = base64.b64encode(handle.read()).decode('ascii')
                return jsonify({
                    'type': 'image',
                    'name': target.name,
                    'path': str(target.relative_to(file_root)),
                    'mime': mime_type,
                    'content': encoded
                })
            return jsonify({'error': 'Unsupported file type'}), 400

        with open(target, 'r', encoding='utf-8', errors='ignore') as handle:
            content = handle.read()

        return jsonify({
            'type': 'text',
            'name': target.name,
            'path': str(target.relative_to(file_root)),
            'ext': target.suffix.lower(),
            'content': content
        })
    except Exception as e:
        logging.error(f"File open failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/files/save', methods=['POST'])
def save_file():
    data = request.json if request.json else {}
    rel_path = data.get('path', '')
    file_type = data.get('type', '')

    try:
        target = resolve_browse_path(rel_path)
    except ValueError:
        return jsonify({'error': 'Invalid path'}), 400

    try:
        if file_type == 'notebook':
            if target.suffix.lower() != '.ipynb':
                return jsonify({'error': 'Notebook must be .ipynb'}), 400
            try:
                import nbformat
            except ImportError:
                return jsonify({'error': '需要安装 nbformat 才能保存 .ipynb 文件'}), 400

            cells_payload = data.get('cells', [])
            nb = nbformat.v4.new_notebook()
            nb_cells = []
            for cell in cells_payload:
                cell_type = cell.get('cell_type', 'code')
                source = cell.get('source', '')
                outputs = cell.get('outputs', []) if isinstance(cell.get('outputs', []), list) else []
                nb_outputs = []
                for output in outputs:
                    try:
                        nb_outputs.append(nbformat.from_dict(output))
                    except Exception:
                        continue
                if cell_type == 'markdown':
                    nb_cells.append(nbformat.v4.new_markdown_cell(source=source))
                elif cell_type == 'raw':
                    nb_cells.append(nbformat.v4.new_raw_cell(source=source))
                else:
                    nb_cells.append(nbformat.v4.new_code_cell(source=source, outputs=nb_outputs, execution_count=None))
            nb.cells = nb_cells
            with open(target, 'w', encoding='utf-8') as handle:
                nbformat.write(nb, handle)
            return jsonify({'success': True})

        if file_type == 'text':
            if not is_allowed_text_file(target):
                return jsonify({'error': 'Unsupported file type'}), 400
            content = data.get('content', '')
            with open(target, 'w', encoding='utf-8') as handle:
                handle.write(content)
            return jsonify({'success': True})

        return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        logging.error(f"File save failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/kernel/stats', methods=['GET'])
def kernel_stats():
    try:
        kernel_id = request.args.get('kernel_id')
        executor, ns = get_kernel_context(kernel_id)
        executor._ensure_kernel()
        vars_info = []
        for name, value in ns.items():
            if name.startswith('_'):
                continue
            if callable(value):
                continue
            module_name = getattr(value, '__module__', '')
            if module_name.startswith('builtins'):
                continue
            size_bytes = estimate_var_size(value)
            vars_info.append({
                'name': name,
                'type': type(value).__name__,
                'size_mb': round(size_bytes / (1024 * 1024), 3)
            })
        vars_info.sort(key=lambda x: x['size_mb'], reverse=True)
        return jsonify({
            'memory_mb': get_process_memory_mb(),
            'vars': vars_info[:10],
            'kernel_id': normalize_kernel_id(kernel_id)
        })
    except Exception as e:
        logging.error(f"Kernel stats failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/files/create', methods=['POST'])
def create_file_or_folder():
    data = request.json if request.json else {}
    rel_path = data.get('path', '')
    item_type = data.get('type', 'file')

    try:
        target = resolve_browse_path(rel_path)
    except ValueError:
        return jsonify({'error': 'Invalid path'}), 400

    if target.exists():
        return jsonify({'error': 'Path already exists'}), 400

    try:
        if item_type == 'folder':
            target.mkdir(parents=True, exist_ok=False)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch(exist_ok=False)
        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"Create failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/files/delete', methods=['POST'])
def delete_file_or_folder():
    data = request.json if request.json else {}
    rel_path = data.get('path', '')

    try:
        target = resolve_browse_path(rel_path)
    except ValueError:
        return jsonify({'error': 'Invalid path'}), 400

    if not target.exists():
        return jsonify({'error': 'Path not found'}), 404

    if target == file_root or str(target).startswith(str(file_root / '.')):
        return jsonify({'error': 'Refusing to delete'}), 400

    try:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"Delete failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/files/rename', methods=['POST'])
def rename_file_or_folder():
    data = request.json if request.json else {}
    src_path = data.get('src', '')
    dst_path = data.get('dst', '')
    try:
        src = resolve_browse_path(src_path)
        dst = resolve_browse_path(dst_path)
    except ValueError:
        return jsonify({'error': 'Invalid path'}), 400
    if not src.exists():
        return jsonify({'error': 'Source not found'}), 404
    if dst.exists():
        return jsonify({'error': 'Target exists'}), 400
    try:
        src.rename(dst)
        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"Rename failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/files/copy', methods=['POST'])
def copy_file_or_folder():
    data = request.json if request.json else {}
    src_path = data.get('src', '')
    dst_path = data.get('dst', '')
    try:
        src = resolve_browse_path(src_path)
        dst = resolve_browse_path(dst_path)
    except ValueError:
        return jsonify({'error': 'Invalid path'}), 400
    if not src.exists():
        return jsonify({'error': 'Source not found'}), 404
    if dst.exists():
        return jsonify({'error': 'Target exists'}), 400
    try:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"Copy failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/files/move', methods=['POST'])
def move_file_or_folder():
    data = request.json if request.json else {}
    src_path = data.get('src', '')
    dst_path = data.get('dst', '')
    try:
        src = resolve_browse_path(src_path)
        dst = resolve_browse_path(dst_path)
    except ValueError:
        return jsonify({'error': 'Invalid path'}), 400
    if not src.exists():
        return jsonify({'error': 'Source not found'}), 404
    if dst.exists():
        return jsonify({'error': 'Target exists'}), 400
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"Move failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/kernel/vars', methods=['GET'])
def kernel_vars():
    try:
        kernel_id = request.args.get('kernel_id')
        executor, ns = get_kernel_context(kernel_id)
        executor._ensure_kernel()
        vars_info = []
        for name, value in ns.items():
            if name.startswith('_'):
                continue
            if callable(value):
                continue
            module_name = getattr(value, '__module__', '')
            if module_name.startswith('builtins'):
                continue
            vars_info.append(summarize_var(name, value))
            try:
                if value.__class__.__name__ == 'AnnData':
                    obs_summary = summarize_var(f'{name}.obs', value.obs)
                    obs_summary['is_child'] = True
                    var_summary = summarize_var(f'{name}.var', value.var)
                    var_summary['is_child'] = True
                    vars_info.append(obs_summary)
                    vars_info.append(var_summary)
            except Exception:
                pass
        vars_info.sort(key=lambda x: x['name'].lower())
        return jsonify({'vars': vars_info[:50], 'kernel_id': normalize_kernel_id(kernel_id)})
    except Exception as e:
        logging.error(f"Kernel vars failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/kernel/var_detail', methods=['GET'])
def kernel_var_detail():
    name = request.args.get('name', '')
    try:
        kernel_id = request.args.get('kernel_id')
        executor, ns = get_kernel_context(kernel_id)
        executor._ensure_kernel()
        value = resolve_var_path(name, ns)
        try:
            import pandas as pd
            if isinstance(value, pd.DataFrame):
                df = value.iloc[:50, :50]
                return jsonify({
                    'type': 'dataframe',
                    'name': name,
                    'shape': list(value.shape),
                    'table': df.to_dict(orient='split')
                })
            if isinstance(value, pd.Series):
                df = value.to_frame().iloc[:50, :1]
                return jsonify({
                    'type': 'dataframe',
                    'name': name,
                    'shape': [len(value), 1],
                    'table': df.to_dict(orient='split')
                })
        except Exception:
            pass
        try:
            import numpy as np
            if isinstance(value, np.ndarray):
                preview = f'ndarray shape={value.shape} dtype={value.dtype}'
                return jsonify({
                    'type': 'text',
                    'name': name,
                    'content': preview
                })
        except Exception:
            pass
        try:
            if value.__class__.__name__ == 'AnnData':
                obs_cols = []
                var_cols = []
                try:
                    obs_cols = list(value.obs.columns)
                except Exception:
                    pass
                try:
                    var_cols = list(value.var.columns)
                except Exception:
                    pass
                summary = {
                    'shape': list(getattr(value, 'shape', [])),
                    'obs_columns': obs_cols,
                    'var_columns': var_cols,
                    'obsm_keys': list(getattr(value, 'obsm', {}).keys()),
                    'layers': list(getattr(value, 'layers', {}).keys())
                }
                return jsonify({
                    'type': 'anndata',
                    'name': name,
                    'summary': summary
                })
        except Exception:
            pass
        content = repr(value)
        content = content[:4000]
        return jsonify({
            'type': 'text',
            'name': name,
            'content': content
        })
    except Exception as e:
        logging.error(f"Kernel var detail failed: {e}")
        return jsonify({'error': str(e)}), 500

# New high-performance data endpoints using FlatBuffers

@app.route('/api/schema', methods=['GET'])
def get_schema():
    """Get data schema"""
    global current_adaptor

    if current_adaptor is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        schema = current_adaptor.get_schema()
        return jsonify({'schema': schema})
    except Exception as e:
        logging.error(f"Schema retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/obs', methods=['GET'])
def get_obs_data():
    """Get observation annotations as FlatBuffers"""
    global current_adaptor

    if current_adaptor is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        columns = request.args.getlist('columns')
        chunk_index = request.args.get('chunk', 0, type=int)

        if chunk_index > 0:
            # Chunked data request
            fbs_data = current_adaptor.get_chunked_data('obs', chunk_index, columns=columns if columns else None)
        else:
            # Full data request
            fbs_data = current_adaptor.get_obs_fbs(columns if columns else None)

        response = make_response(fbs_data)
        response.headers['Content-Type'] = 'application/octet-stream'
        response.headers['Cache-Control'] = 'public, max-age=3600'
        return response

    except Exception as e:
        logging.error(f"Obs data retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/embedding/<embedding_name>', methods=['GET'])
def get_embedding_data(embedding_name):
    """Get embedding coordinates as FlatBuffers"""
    global current_adaptor

    if current_adaptor is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        chunk_index = request.args.get('chunk', 0, type=int)
        out_format = request.args.get('format', 'fbs')

        if chunk_index > 0:
            if out_format == 'json':
                # JSON slice response
                if embedding_name == 'random' or embedding_name == 'X_random':
                    coords = current_adaptor._get_random_embedding()
                else:
                    embedding_key = f'X_{embedding_name}' if not embedding_name.startswith('X_') else embedding_name
                    coords = current_adaptor.adata.obsm[embedding_key]
                start = chunk_index * current_adaptor.chunk_size
                end = min(start + current_adaptor.chunk_size, current_adaptor.n_obs)
                return jsonify({'x': coords[start:end, 0].tolist(), 'y': coords[start:end, 1].tolist()})
            else:
                # FBS slice
                fbs_data = current_adaptor.get_chunked_data('embedding', chunk_index, embedding_name=embedding_name)
                response = make_response(fbs_data)
                response.headers['Content-Type'] = 'application/octet-stream'
                response.headers['Cache-Control'] = 'public, max-age=3600'
                return response
        else:
            if out_format == 'json':
                if embedding_name == 'random' or embedding_name == 'X_random':
                    coords = current_adaptor._get_random_embedding()
                else:
                    embedding_key = f'X_{embedding_name}' if not embedding_name.startswith('X_') else embedding_name
                    coords = current_adaptor.adata.obsm[embedding_key]
                return jsonify({'x': coords[:, 0].tolist(), 'y': coords[:, 1].tolist()})
            else:
                fbs_data = current_adaptor.get_embedding_fbs(embedding_name)
                response = make_response(fbs_data)
                response.headers['Content-Type'] = 'application/octet-stream'
                response.headers['Cache-Control'] = 'public, max-age=3600'
                return response

    except KeyError as e:
        return jsonify({'error': f'Embedding not found: {embedding_name}'}), 404
    except Exception as e:
        logging.error(f"Embedding data retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/expression', methods=['POST'])
def get_expression_data():
    """Get gene expression data as FlatBuffers"""
    global current_adaptor

    if current_adaptor is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        data = request.json
        gene_names = data.get('genes', [])
        cell_indices = data.get('cell_indices', None)

        if not gene_names:
            return jsonify({'error': 'No genes specified'}), 400

        fbs_data = current_adaptor.get_expression_fbs(gene_names, cell_indices)

        response = make_response(fbs_data)
        response.headers['Content-Type'] = 'application/octet-stream'
        return response

    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logging.error(f"Expression data retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/qc_prefixes', methods=['GET'])
def get_qc_prefixes():
    """Detect common QC gene prefixes (e.g., mitochondrial) from var_names."""
    global current_adata
    if current_adata is None:
        return jsonify({'error': 'No data loaded'}), 400
    try:
        import re
        var_names = current_adata.var_names.astype(str)
        prefixes = set()
        # Detect typical mt prefixes
        for name in var_names[: min(10000, len(var_names))]:  # sample to be safe
            if len(name) < 3:
                continue
            if re.match(r'^(mt|MT|Mt|mT)+', name):
                prefixes.add(name[:3])  # e.g., 'MT-' or 'mt-'
        # If underscore variants like 'mt_' present
        for name in var_names[: min(60000, len(var_names))]:
            if name.lower().startswith('mt'):
                prefixes.add(name[:3])
        # Fallback to common defaults
        if not prefixes:
            prefixes.update(['MT-', 'mt-'])
        return jsonify({'mt_prefixes': sorted(prefixes)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/filter', methods=['POST'])
def filter_cells():
    """Filter cells based on criteria"""
    global current_adaptor

    if current_adaptor is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        filters = request.json
        indices = current_adaptor.filter_cells(filters)

        return jsonify({
            'filtered_indices': indices,
            'count': len(indices)
        })

    except Exception as e:
        logging.error(f"Cell filtering failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/differential_expression', methods=['POST'])
def compute_differential_expression():
    """Compute differential expression between groups"""
    global current_adaptor

    if current_adaptor is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        data = request.json
        group1_indices = data.get('group1_indices', [])
        group2_indices = data.get('group2_indices', [])
        method = data.get('method', 'wilcoxon')
        n_genes = data.get('n_genes', 100)

        if not group1_indices or not group2_indices:
            return jsonify({'error': 'Both groups must have cells'}), 400

        # Run in thread pool to avoid blocking
        future = thread_pool.submit(
            current_adaptor.get_differential_expression,
            group1_indices, group2_indices, method, n_genes
        )

        result = future.result(timeout=30)  # 30 second timeout

        return jsonify(result)

    except Exception as e:
        logging.error(f"Differential expression failed: {e}")
        return jsonify({'error': str(e)}), 500

# Legacy endpoint for backward compatibility
@app.route('/api/plot', methods=['POST'])
def plot_data_legacy():
    """Legacy plot endpoint - returns JSON data for Plotly"""
    global current_adaptor

    if current_adaptor is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        data = request.json
        embedding = data.get('embedding', '')
        color_by = data.get('color_by', '')
        palette = data.get('palette', None)  # Get continuous palette parameter
        category_palette = data.get('category_palette', None)  # Get category palette parameter
        vmin = data.get('vmin', None)  # Get vmin parameter
        vmax = data.get('vmax', None)  # Get vmax parameter

        if not embedding:
            return jsonify({'error': 'No embedding specified'}), 400

        # For legacy compatibility, convert FlatBuffers data to JSON
        from server.common.fbs.matrix import decode_matrix_fbs

        # Get embedding data
        fbs_data = current_adaptor.get_embedding_fbs(embedding)
        coords_df = decode_matrix_fbs(fbs_data)

        plot_data = {
            'x': coords_df['x'].tolist(),
            'y': coords_df['y'].tolist(),
            'hover_text': [f'Cell {i}' for i in range(len(coords_df))]
        }

        # Handle coloring if requested
        if color_by:
            if color_by.startswith('obs:'):
                col_name = color_by.replace('obs:', '')
                obs_fbs = current_adaptor.get_obs_fbs([col_name])
                obs_df = decode_matrix_fbs(obs_fbs)

                if col_name in obs_df.columns:
                    values = obs_df[col_name]
                    if pd.api.types.is_numeric_dtype(values):
                        # Apply vmin/vmax clipping if specified
                        colors_array = values.values
                        if vmin is not None or vmax is not None:
                            colors_array = np.clip(colors_array,
                                                   vmin if vmin is not None else colors_array.min(),
                                                   vmax if vmax is not None else colors_array.max())

                        plot_data['colors'] = colors_array.tolist()
                        plot_data['colorscale'] = palette if palette else 'Viridis'
                        plot_data['color_label'] = col_name

                        # Add vmin/vmax to plot data for Plotly
                        if vmin is not None:
                            plot_data['cmin'] = vmin
                        if vmax is not None:
                            plot_data['cmax'] = vmax
                    else:
                        # Categorical handling
                        if pd.api.types.is_categorical_dtype(values):
                            categories = values
                        else:
                            categories = values.astype('category')

                        plot_data['color_label'] = col_name
                        plot_data['category_labels'] = categories.cat.categories.tolist()
                        plot_data['category_codes'] = categories.cat.codes.tolist()

                        n_categories = len(categories.cat.categories)
                        # Use category palette if specified
                        discrete_colors = get_discrete_colors(n_categories, category_palette)
                        plot_data['colors'] = [discrete_colors[code] for code in categories.cat.codes]
                        plot_data['discrete_colors'] = discrete_colors

            elif color_by.startswith('gene:'):
                gene_name = color_by.replace('gene:', '')
                try:
                    expr_fbs = current_adaptor.get_expression_fbs([gene_name])
                    expr_df = decode_matrix_fbs(expr_fbs)

                    if gene_name in expr_df.columns:
                        expression = expr_df[gene_name].values

                        # Apply vmin/vmax clipping if specified
                        if vmin is not None or vmax is not None:
                            expression = np.clip(expression,
                                               vmin if vmin is not None else expression.min(),
                                               vmax if vmax is not None else expression.max())

                        plot_data['colors'] = expression.tolist()
                        plot_data['colorscale'] = palette if palette else 'Viridis'
                        plot_data['color_label'] = f'{gene_name} expression'

                        # Add vmin/vmax to plot data for Plotly
                        if vmin is not None:
                            plot_data['cmin'] = vmin
                        if vmax is not None:
                            plot_data['cmax'] = vmax
                except Exception as e:
                    logging.warning(f"Gene expression retrieval failed for {gene_name}: {e}")

        return jsonify(plot_data)

    except Exception as e:
        logging.error(f"Legacy plot failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tools/<tool>', methods=['POST'])
def run_tool(tool):
    global current_adata

    if current_adata is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        params = request.json if request.json else {}

        if tool == 'normalize':
            target_sum = params.get('target_sum', 1e4)
            sc.pp.normalize_total(current_adata, target_sum=target_sum)

        elif tool == 'log1p':
            sc.pp.log1p(current_adata)

        elif tool == 'scale':
            max_value = params.get('max_value', 10)
            sc.pp.scale(current_adata, max_value=max_value)

        elif tool == 'hvg':
            n_genes = params.get('n_genes', 2000)
            method = params.get('method', 'seurat')

            if method == 'seurat':
                sc.pp.highly_variable_genes(current_adata, flavor='seurat',
                                           n_top_genes=n_genes)
            elif method == 'cell_ranger':
                sc.pp.highly_variable_genes(current_adata, flavor='cell_ranger',
                                           n_top_genes=n_genes)
            else:
                sc.pp.highly_variable_genes(current_adata, flavor='seurat_v3',
                                           n_top_genes=n_genes)

            current_adata = current_adata[:, current_adata.var.highly_variable]

        elif tool == 'pca':
            n_comps = params.get('n_comps', 50)
            sc.pp.pca(current_adata, n_comps=min(n_comps, min(current_adata.shape) - 1))

        elif tool == 'neighbors':
            n_neighbors = params.get('n_neighbors', 15)
            sc.pp.neighbors(current_adata, n_neighbors=n_neighbors)

        elif tool == 'umap':
            n_neighbors = params.get('n_neighbors', 15)
            min_dist = params.get('min_dist', 0.5)

            # Compute neighbors if not already done
            if 'neighbors' not in current_adata.uns:
                sc.pp.neighbors(current_adata, n_neighbors=n_neighbors)

            sc.tl.umap(current_adata, min_dist=min_dist)

        elif tool == 'tsne':
            perplexity = params.get('perplexity', 30)

            # Run PCA first if not done
            if 'X_pca' not in current_adata.obsm:
                sc.pp.pca(current_adata)

            sc.tl.tsne(current_adata, perplexity=perplexity)

        elif tool == 'leiden':
            resolution = params.get('resolution', 1.0)

            # Compute neighbors if not already done
            if 'neighbors' not in current_adata.uns:
                sc.pp.neighbors(current_adata)

            # Use omicverse's leiden if available
            sc.tl.leiden(current_adata, resolution=resolution)

        elif tool == 'louvain':
            resolution = params.get('resolution', 1.0)

            # Compute neighbors if not already done
            if 'neighbors' not in current_adata.uns:
                sc.pp.neighbors(current_adata)

            sc.tl.louvain(current_adata, resolution=resolution)

        elif tool == 'filter_cells':
            # Apply thresholds sequentially; Scanpy only accepts one per call
            for key in ('min_counts', 'min_genes', 'max_counts', 'max_genes'):
                v = params.get(key, None)
                if v is None or v == "":
                    continue
                try:
                    v = int(v)
                except Exception:
                    continue
                if key == 'min_counts':
                    sc.pp.filter_cells(current_adata, min_counts=v)
                elif key == 'min_genes':
                    sc.pp.filter_cells(current_adata, min_genes=v)
                elif key == 'max_counts':
                    sc.pp.filter_cells(current_adata, max_counts=v)
                elif key == 'max_genes':
                    sc.pp.filter_cells(current_adata, max_genes=v)

        elif tool == 'filter_genes':
            # Sequentially apply lower thresholds
            min_cells = params.get('min_cells', None)
            g_min_counts = params.get('min_counts', params.get('g_min_counts', None))
            max_cells = params.get('max_cells', None)
            g_max_counts = params.get('max_counts', params.get('g_max_counts', None))
            if g_min_counts is not None and g_min_counts != "":
                try:
                    sc.pp.filter_genes(current_adata, min_counts=int(g_min_counts))
                except Exception:
                    pass
            if min_cells is not None and min_cells != "":
                try:
                    sc.pp.filter_genes(current_adata, min_cells=int(min_cells))
                except Exception:
                    pass
            import numpy as _np
            X = current_adata.X.toarray() if hasattr(current_adata.X, 'toarray') else current_adata.X
            if max_cells is not None:
                expr_cells = (X > 0).sum(axis=0)
                expr_cells = _np.asarray(expr_cells).A1 if hasattr(expr_cells, 'A1') else _np.asarray(expr_cells)
                keep = expr_cells <= int(max_cells)
                current_adata._inplace_subset_var(keep)
                X = current_adata.X.toarray() if hasattr(current_adata.X, 'toarray') else current_adata.X
            if g_max_counts is not None:
                sums = _np.asarray(X).sum(axis=0)
                sums = sums.A1 if hasattr(sums, 'A1') else sums
                keep = sums <= float(g_max_counts)
                current_adata._inplace_subset_var(keep)

        elif tool == 'filter_outliers':
            # Compute QC metrics if not present and filter by mitochondrial percentage
            import numpy as _np
            # Determine mt prefixes from params or detection
            req_prefixes = params.get('mt_prefixes')
            if req_prefixes:
                mt_prefixes = [p.strip() for p in str(req_prefixes).split(',') if p.strip()]
            else:
                # Detect typical prefixes from var_names
                import re
                mt_prefixes = []
                for name in current_adata.var_names.astype(str):
                    if re.match(r'^(mt|MT|Mt|mT)[-_].+', name):
                        mt_prefixes = list({name[:3] for name in current_adata.var_names.astype(str) if len(name) >= 3})
                        break
                if not mt_prefixes:
                    mt_prefixes = ['MT-', 'mt-']
            # Annotate mt mask
            lname = _np.array([str(x) for x in current_adata.var_names])
            mt_mask = _np.zeros(lname.shape[0], dtype=bool)
            for pref in mt_prefixes:
                try:
                    mt_mask |= _np.char.startswith(lname, pref)
                except Exception:
                    pass
            current_adata.var['mt'] = mt_mask
            # Annotate ribosomal genes (RPS/RPL)
            upper_names = _np.char.upper(lname)
            ribo_mask = _np.char.startswith(upper_names, 'RPS') | _np.char.startswith(upper_names, 'RPL')
            current_adata.var['ribo'] = ribo_mask
            # Annotate hemoglobin genes (HB but not HBP)
            import re as _re
            hb_mask = _np.array([bool(_re.search(r'^(HB(?!P))', nm, flags=_re.I)) for nm in lname])
            current_adata.var['hb'] = hb_mask
            # Calculate QC metrics
            sc.pp.calculate_qc_metrics(current_adata, qc_vars=['mt', 'ribo', 'hb'], inplace=True, log1p=True)
            # Build a combined keep mask
            keep = _np.ones(current_adata.n_obs, dtype=bool)
            # Apply mt threshold
            max_mt_percent = params.get('max_mt_percent', None)
            if max_mt_percent is not None:
                pct_col = None
                for c in current_adata.obs.columns:
                    lc = str(c).lower()
                    if 'pct' in lc and 'mt' in lc:
                        pct_col = c; break
                if pct_col is None and 'pct_counts_mt' in current_adata.obs.columns:
                    pct_col = 'pct_counts_mt'
                if pct_col is not None:
                    keep &= (current_adata.obs[pct_col].astype(float) <= float(max_mt_percent)).values
            # Apply ribo threshold
            max_ribo_percent = params.get('max_ribo_percent', None)
            if max_ribo_percent is not None:
                pct_col = None
                for c in current_adata.obs.columns:
                    lc = str(c).lower()
                    if 'pct' in lc and ('ribo' in lc or 'rps' in lc or 'rpl' in lc):
                        pct_col = c; break
                if pct_col is None and 'pct_counts_ribo' in current_adata.obs.columns:
                    pct_col = 'pct_counts_ribo'
                if pct_col is not None:
                    keep &= (current_adata.obs[pct_col].astype(float) <= float(max_ribo_percent)).values
            # Apply hemoglobin threshold
            max_hb_percent = params.get('max_hb_percent', None)
            if max_hb_percent is not None:
                pct_col = None
                for c in current_adata.obs.columns:
                    lc = str(c).lower()
                    if 'pct' in lc and 'hb' in lc:
                        pct_col = c; break
                if pct_col is None and 'pct_counts_hb' in current_adata.obs.columns:
                    pct_col = 'pct_counts_hb'
                if pct_col is not None:
                    keep &= (current_adata.obs[pct_col].astype(float) <= float(max_hb_percent)).values
            # Subset if any threshold applied
            if keep.sum() != current_adata.n_obs:
                current_adata._inplace_subset_obs(keep)

        elif tool == 'doublets':
            # Use Scanpy's integrated Scrublet wrapper
            batch_key = params.get('batch_key') or None
            scrublet_kwargs = dict(
                sim_doublet_ratio=float(params.get('sim_doublet_ratio', 2.0)),
                expected_doublet_rate=float(params.get('expected_doublet_rate', 0.05)),
                stdev_doublet_rate=float(params.get('stdev_doublet_rate', 0.02)),
                synthetic_doublet_umi_subsampling=float(params.get('synthetic_doublet_umi_subsampling', 1.0)),
                knn_dist_metric=params.get('knn_dist_metric', 'euclidean'),
                normalize_variance=bool(params.get('normalize_variance', True)),
                log_transform=bool(params.get('log_transform', False)),
                mean_center=bool(params.get('mean_center', True)),
                n_prin_comps=int(params.get('n_prin_comps', 30))
            )
            try:
                sc.pp.scrublet(current_adata, batch_key=batch_key, **scrublet_kwargs)
            except Exception as e:
                return jsonify({'error': f'Scrublet 执行失败: {e}'}), 400

            # Try common column names written by scanpy scrublet
            pred_col = None
            for c in ('predicted_doublet', 'predicted_doublets'):
                if c in current_adata.obs.columns:
                    pred_col = c; break
            score_col = None
            for c in ('doublet_score', 'doublet_scores'):
                if c in current_adata.obs.columns:
                    score_col = c; break
            # If predicted flags present, filter out doublets
            if pred_col is not None:
                keep = ~current_adata.obs[pred_col].astype(bool)
                current_adata._inplace_subset_obs(keep.values)
        
        else:
            return jsonify({'error': f'Unknown tool: {tool}'}), 400

        # Sync adaptor and kernel with modified AnnData
        sync_adaptor_with_adata()
        try:
            kernel_executor.sync_adata(current_adata)
        except Exception:
            pass

        # Return updated info
        info = {
            'filename': current_filename,
            'n_cells': current_adata.n_obs,
            'n_genes': current_adata.n_vars,
            'embeddings': [emb.replace('X_', '') for emb in current_adata.obsm.keys()],
            'obs_columns': list(current_adata.obs.columns),
            'var_columns': list(current_adata.var.columns)
        }

        return jsonify(info)

    except Exception as e:
        # Provide more helpful message if scrublet missing
        msg = str(e)
        if 'scrublet' in msg.lower():
            msg = '需要安装 scrublet 包才能运行双细胞检测（pip install scrublet）'
        return jsonify({'error': msg}), 500

def get_discrete_colors(n_categories, palette_name=None):
    """获取离散分类颜色"""
    # 预定义的颜色调色板
    color_palettes = {
        'set1': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999'],
        'set2': ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'],
        'set3': ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd'],
        'paired': ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'],
        'plotly3': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
        'pastel1': ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc', '#e5d8bd', '#fddaec', '#f2f2f2'],
        'pastel2': ['#b3e2cd', '#fdcdac', '#cbd5e8', '#f4cae4', '#e6f5c9', '#fff2ae', '#f1e2cc', '#cccccc'],
        'dark2': ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666'],
        'accent': ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f', '#bf5b17', '#666666']
    }

    # Use matplotlib colormaps if available
    matplotlib_palettes = {
        'tab10': 'tab10',
        'tab20': 'tab20',
        'tab20b': 'tab20b',
        'tab20c': 'tab20c'
    }

    # If a specific palette is requested
    if palette_name and palette_name in matplotlib_palettes:
        cmap = plt.get_cmap(matplotlib_palettes[palette_name])
        # Get colors from the colormap
        n_cmap = cmap.N if hasattr(cmap, 'N') else 256
        base_colors = []
        for i in range(min(n_categories, n_cmap)):
            rgba = cmap(i / n_cmap if n_cmap > 20 else i)
            base_colors.append(mcolors.rgb2hex(rgba))
        # Extend if needed by cycling
        colors = base_colors.copy()
        while len(colors) < n_categories:
            idx = len(colors) % len(base_colors)
            colors.append(base_colors[idx])
        return colors[:n_categories]
    elif palette_name and palette_name in color_palettes:
        palette = color_palettes[palette_name]
    else:
        # Default palette selection based on number of categories
        if n_categories <= 5:
            palette = color_palettes['set1']
        elif n_categories <= 8:
            palette = color_palettes['set2']
        elif n_categories <= 10:
            palette = color_palettes['set3']
        elif n_categories <= 12:
            palette = color_palettes['paired']
        elif n_categories <= 20:
            palette = color_palettes['set3']
        else:
            palette = color_palettes['plotly3']

    # 如果需要的颜色数量超过调色板长度，循环使用
    if n_categories > len(palette):
        colors = []
        for i in range(n_categories):
            colors.append(palette[i % len(palette)])
        return colors
    else:
        return palette[:n_categories]

@app.route('/api/save', methods=['POST'])
def save_data():
    global current_adata, current_filename

    if current_adata is None:
        return jsonify({'error': 'No data to save'}), 400

    try:
        # Save to temporary file
        output_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                  f'processed_{current_filename}')
        current_adata.write_h5ad(output_path)

        return send_file(output_path, as_attachment=True,
                        download_name=f'processed_{current_filename}')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'loaded': current_adata is not None,
        'filename': current_filename,
        'cells': current_adata.n_obs if current_adata else 0,
        'genes': current_adata.n_vars if current_adata else 0,
    })

@app.route('/api/genes', methods=['GET'])
def get_genes():
    """Get all gene names"""
    global current_adata

    if current_adata is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        genes = list(current_adata.var_names)
        return jsonify({'genes': genes})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/gene_search', methods=['POST'])
def search_genes():
    """Search for genes by name pattern"""
    global current_adata

    if current_adata is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        data = request.json
        pattern = data.get('pattern', '').upper()

        if not pattern:
            return jsonify({'genes': []})

        # Search genes that match the pattern
        matching_genes = [gene for gene in current_adata.var_names if pattern in gene.upper()]

        # Limit results to avoid too many matches
        matching_genes = matching_genes[:20]

        return jsonify({'genes': matching_genes})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data_info', methods=['GET'])
def get_data_info():
    """Get detailed information about the loaded data"""
    global current_adata
    
    if current_adata is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    try:
        info = {
            'shape': list(current_adata.shape),
            'obs_keys': list(current_adata.obs.keys()),
            'var_keys': list(current_adata.var.keys()),
            'obsm_keys': list(current_adata.obsm.keys()) if current_adata.obsm else [],
            'uns_keys': list(current_adata.uns.keys()) if current_adata.uns else [],
            'layers': list(current_adata.layers.keys()) if current_adata.layers else []
        }
        
        # Add some statistics
        if hasattr(current_adata.X, 'toarray'):
            X = current_adata.X.toarray()
        else:
            X = current_adata.X
            
        info['statistics'] = {
            'mean_genes_per_cell': float(np.mean(np.sum(X > 0, axis=1))),
            'mean_counts_per_cell': float(np.mean(np.sum(X, axis=1))),
            'total_counts': float(np.sum(X))
        }
        
        return jsonify(info)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/execute_code', methods=['POST'])
def execute_code():
    """Execute Python code with access to current_adata"""
    global current_adata

    try:
        payload = request.json if request.json else {}
        code = payload.get('code', '')
        if not code:
            return jsonify({'error': '没有提供代码'}), 400
        kernel_id = normalize_kernel_id(payload.get('kernel_id'))
        executor, ns = get_kernel_context(kernel_id)
        if kernel_id == 'default.ipynb' and current_adata is None:
            return jsonify({'error': '没有加载数据。请先上传H5AD文件。'}), 400

        try:
            shared_adata = current_adata if kernel_id == 'default.ipynb' else None
            execution = executor.execute(code, shared_adata, user_ns=ns)
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

        output = execution.get('output') or ''
        stderr = execution.get('stderr') or ''
        if stderr:
            output = output + '\n' + stderr if output else stderr

        if execution.get('error'):
            return jsonify({
                'error': execution['error'],
                'output': output,
                'figures': execution.get('figures', [])
            }), 200

        result = execution.get('result')
        if result is not None:
            result = str(result)

        new_adata = execution.get('adata')
        data_updated = False
        if kernel_id == 'default.ipynb' and new_adata is not None:
            current_adata = new_adata
            data_updated = True
            sync_adaptor_with_adata()
            try:
                kernel_executor.sync_adata(current_adata)
            except Exception:
                pass

        data_info = None
        if data_updated:
            data_info = {
                'filename': current_filename,
                'n_cells': current_adata.n_obs,
                'n_genes': current_adata.n_vars,
                'embeddings': [emb.replace('X_', '') for emb in current_adata.obsm.keys()],
                'obs_columns': list(current_adata.obs.columns),
                'var_columns': list(current_adata.var.columns)
            }

        return jsonify({
            'output': output,
            'result': result,
            'figures': execution.get('figures', []),
            'data_updated': data_updated,
            'data_info': data_info,
            'kernel_id': kernel_id,
            'success': True
        })

    except Exception as e:
        import traceback
        return jsonify({'error': traceback.format_exc()}), 500

@app.route('/api/kernel/list', methods=['GET'])
def kernel_list():
    kernel_id = normalize_kernel_id(request.args.get('kernel_id'))
    current_name = current_kernel_name if kernel_id == 'default.ipynb' else kernel_names.get(kernel_id, 'python3')
    return jsonify({
        'kernels': [
            {
                'name': 'python3',
                'display_name': 'Python 3 (ipykernel)'
            }
        ],
        'default': 'python3',
        'current': current_name,
        'kernel_id': kernel_id
    })

@app.route('/api/kernel/select', methods=['POST'])
def kernel_select():
    global current_kernel_name, current_adata
    payload = request.get_json(silent=True) or {}
    name = payload.get('name')
    kernel_id = normalize_kernel_id(payload.get('kernel_id'))
    if not name:
        return jsonify({'error': '缺少内核名称'}), 400
    if name != 'python3':
        return jsonify({'error': '当前仅支持 Python 3 (ipykernel)'}), 400
    try:
        reset_kernel_namespace(kernel_id)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    if kernel_id == 'default.ipynb':
        current_kernel_name = name
    else:
        kernel_names[kernel_id] = name
    return jsonify({'current': name, 'kernel_id': kernel_id})

@app.route('/api/export_plot_data', methods=['POST'])
def export_plot_data():
    """Export current plot data as CSV"""
    global current_adata
    
    if current_adata is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    try:
        data = request.json
        embedding = data.get('embedding', '')
        color_by = data.get('color_by', '')
        
        if not embedding:
            return jsonify({'error': 'No embedding specified'}), 400
        
        # Get embedding data
        embedding_key = f'X_{embedding}' if not embedding.startswith('X_') else embedding
        
        if embedding_key not in current_adata.obsm:
            return jsonify({'error': f'Embedding {embedding} not found'}), 404
        
        coords = current_adata.obsm[embedding_key]
        
        # Create DataFrame
        df_data = {
            'cell_id': current_adata.obs_names,
            f'{embedding}_1': coords[:, 0],
            f'{embedding}_2': coords[:, 1]
        }
        
        # Add color information if specified
        if color_by:
            if color_by.startswith('obs:'):
                col_name = color_by.replace('obs:', '')
                if col_name in current_adata.obs.columns:
                    df_data[col_name] = current_adata.obs[col_name]
            elif color_by.startswith('gene:'):
                gene_name = color_by.replace('gene:', '')
                if gene_name in current_adata.var_names:
                    gene_idx = current_adata.var_names.get_loc(gene_name)
                    if hasattr(current_adata.X, 'toarray'):
                        expression = current_adata.X[:, gene_idx].toarray().flatten()
                    else:
                        expression = current_adata.X[:, gene_idx].flatten()
                    df_data[f'{gene_name}_expression'] = expression
        
        df = pd.DataFrame(df_data)
        
        # Save to temporary file
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'plot_data_{embedding}.csv')
        df.to_csv(output_path, index=False)
        
        return send_file(output_path, as_attachment=True, 
                        download_name=f'plot_data_{embedding}.csv')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/agent/run', methods=['POST'])
def agent_run():
    """Run OmicVerse Agent on current adata."""
    global current_adata
    payload = request.json if request.json else {}
    prompt = (payload.get('message') or '').strip()
    if not prompt:
        return jsonify({'error': '没有提供问题'}), 400
    config = payload.get('config') or {}
    system_prompt = (config.get('systemPrompt') or '').strip()
    if system_prompt:
        prompt = f"{system_prompt}\n\n{prompt}"
    try:
        agent = get_agent_instance(config)
        if current_adata is None:
            reply = run_agent_chat(agent, prompt)
            return jsonify({
                'reply': reply,
                'code': None,
                'data_updated': False,
                'data_info': None
            })
        result = run_agent_stream(agent, prompt, current_adata)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

    new_adata = result.get('result_adata')
    data_updated = False
    if new_adata is not None:
        current_adata = new_adata
        data_updated = True
        sync_adaptor_with_adata()
        try:
            kernel_executor.sync_adata(current_adata)
        except Exception:
            pass

    data_info = None
    if data_updated:
        data_info = {
            'filename': current_filename,
            'n_cells': current_adata.n_obs,
            'n_genes': current_adata.n_vars,
            'embeddings': [emb.replace('X_', '') for emb in current_adata.obsm.keys()],
            'obs_columns': list(current_adata.obs.columns),
            'var_columns': list(current_adata.var.columns)
        }

    reply = '已完成分析。'
    if result.get('result_shape'):
        shape = result.get('result_shape')
        reply = f'已完成分析，结果数据维度为 {shape[0]} × {shape[1]}。'

    return jsonify({
        'reply': reply,
        'code': result.get('code'),
        'data_updated': data_updated,
        'data_info': data_info
    })

@app.route('/')
def index():
    # Redirect to single cell analysis interface
    return send_from_directory(app.root_path, 'single_cell_analysis_standalone.html')

@app.route('/legacy')
def legacy_index():
    # Prefer built Horizon UI (CRA) if available
    cra_build = os.path.join(app.root_path, 'design_ref', 'horizon-ui-chakra', 'build')
    cra_index = os.path.join(cra_build, 'index.html')
    if os.path.exists(cra_index):
        return send_from_directory(cra_build, 'index.html')

    # Else prefer built Vite UI if available
    ui_dist = os.path.join(app.root_path, 'ui', 'dist')
    ui_index = os.path.join(ui_dist, 'index.html')
    if os.path.exists(ui_index):
        return send_from_directory(ui_dist, 'index.html')
    # Fallback to legacy static index
    return send_from_directory(app.root_path, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    # Serve Horizon UI (CRA) build files if present
    cra_build = os.path.join(app.root_path, 'design_ref', 'horizon-ui-chakra', 'build')
    cra_file = os.path.join(cra_build, path)
    if os.path.exists(cra_file):
        return send_from_directory(cra_build, path)

    # Serve Vite UI build files if present
    ui_dist = os.path.join(app.root_path, 'ui', 'dist')
    ui_file = os.path.join(ui_dist, path)
    if os.path.exists(ui_file):
        return send_from_directory(ui_dist, path)

    # Serve legacy static assets
    if path.startswith('static/'):
        rel = path[len('static/'):]
        return send_from_directory(app.static_folder, rel)

    full_path = os.path.join(app.root_path, path)
    if os.path.isfile(full_path):
        return send_from_directory(app.root_path, path)

    # SPA fallback
    if os.path.exists(os.path.join(cra_build, 'index.html')):
        return send_from_directory(cra_build, 'index.html')
    if os.path.exists(os.path.join(ui_dist, 'index.html')):
        return send_from_directory(ui_dist, 'index.html')
    return send_from_directory(app.root_path, 'index.html')

if __name__ == '__main__':
    # Use a safer default port to avoid conflicts (e.g., AirPlay on macOS)
    port = int(os.environ.get('PORT', 5050))
    app.run(debug=True, host='0.0.0.0', port=port)

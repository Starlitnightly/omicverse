"""
OmicVerse Web Application - Refactored
=======================================
Main Flask application with modular blueprint architecture.
"""

from flask import Flask, request, jsonify, send_file, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import scanpy as sc
import numpy as np
import pandas as pd
import os
import tempfile
import warnings
import json
import logging
import threading
import time
import queue
import io

# Import services
from services.kernel_service import InProcessKernelExecutor, normalize_kernel_id, get_kernel_context
from services.agent_service import (
    get_agent_instance, run_agent_stream, run_agent_chat, make_turn_id,
    stream_agent_events, get_turn_buffer, clear_turn_buffer,
    cancel_active_turn, get_active_turn_for_session,
    build_harness_initialize_payload, load_trace,
    resolve_pending_approval, resolve_pending_question,
)
from services.agent_session_service import session_manager
from utils.notebook_helpers import ensure_default_notebook

# Import blueprints
from routes import kernel, files, data, notebooks
from routes.terminal import terminal_bp

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

# Disable werkzeug request logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = None
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

# Remote mode detection (Phase 4)
OV_WEB_REMOTE_MODE = os.environ.get('OV_WEB_REMOTE_MODE', '0') == '1'

# Global state container (for easier blueprint access)
class AppState:
    """Container for global application state."""
    def __init__(self):
        self.current_adaptor = None
        self.current_adata = None
        self.deg_results   = None     # DataFrame from last DEG analysis
        self.deg_method    = None     # method used for last DEG analysis
        self.deg_condition = None     # condition column used for last DEG analysis
        # DCT analysis state
        self.dct_results        = None   # DataFrame from last DCT analysis
        self.dct_method         = None   # 'sccoda' or 'milopy'
        self.dct_model          = None   # model object (Sccoda or Milo)
        self.dct_data           = None   # sccoda_data (MuData) or mdata (MuData)
        self.dct_adata          = None   # filtered adata used for plotting
        self.dct_condition      = None   # condition column
        self.dct_ctrl_group     = None
        self.dct_test_group     = None
        self.dct_cell_type_key  = None
        self.dct_sample_key     = None
        self.current_filename = None
        self.is_preview_mode = False   # True when opened with backed='r'
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.kernel_lock = threading.Lock()
        self.kernel_executor = None
        self.current_kernel_name = 'python3'
        self.kernel_names = {}
        self.kernel_sessions = {}
        self.notebook_root = os.getcwd()
        self.file_root = Path(self.notebook_root).resolve()

# Create global state instance
state = AppState()
state.kernel_executor = InProcessKernelExecutor(state.kernel_lock)

# Utility function
def sync_adaptor_with_adata():
    """Keep adaptor in sync after kernel or tool updates."""
    if state.current_adaptor is None or state.current_adata is None:
        return
    try:
        state.current_adaptor.adata = state.current_adata
        state.current_adaptor.n_obs = state.current_adata.n_obs
        state.current_adaptor.n_vars = state.current_adata.n_vars
        state.current_adaptor._build_indexes()
        # Clear embedding cache so adaptor re-reads from adata.obsm
        # (important after cell-count changes where X_random is auto-subset)
        if hasattr(state.current_adaptor, '_embedding_cache'):
            state.current_adaptor._embedding_cache.clear()
    except Exception:
        pass


# Helper function for discrete colors
# ── OmicVerse built-in discrete palettes ─────────────────────────────────────
_OV_SC_COLOR = [
    '#1F577B', '#A56BA7', '#E0A7C8', '#E069A6', '#941456',
    '#FCBC10', '#EF7B77', '#279AD7', '#F0EEF0', '#EAEFC5',
    '#7CBB5F', '#368650', '#A499CC', '#5E4D9A', '#78C2ED',
    '#866017', '#9F987F', '#E0DFED', '#01A0A7', '#75C8CC',
    '#F0D7BC', '#D5B26C', '#D5DA48', '#B6B812', '#9DC3C3',
    '#A89C92', '#FEE00C', '#FEF2A1',
]

_OV_CET = [
    '#d60000','#8c3bff','#018700','#00acc6','#97ff00','#ff7ed1','#6b004f','#ffa52f',
    '#00009c','#857067','#004942','#4f2a00','#00fdcf','#bcb6ff','#95b379','#bf03b8',
    '#2466a1','#280041','#dbb3af','#fdf490','#4f445b','#a37c00','#ff7066','#3f806e',
    '#82000c','#a37bb3','#344d00','#9ae4ff','#eb0077','#2d000a','#5d90ff','#00c61f',
    '#5701aa','#001d00','#9a4600','#959ea5','#9a425b','#001f31','#c8c300','#ffcfff',
    '#00bd9a','#3615ff','#2d2424','#df57ff','#bde6bf','#7e4497','#524f3b','#d86600',
    '#647438','#c17287','#6e7489','#809c03','#bd8a64','#623338','#cacdda','#6beb82',
    '#213f69','#a17eff','#fd03ca','#75bcfd','#d8c382','#cda3cd','#6d4f00','#006974',
    '#469e5d','#93c6bf','#f9ff00','#bf5444','#00643b','#5b4fa8','#521f64','#4f5eff',
    '#7e8e77','#b808f9','#8a91c3','#b30034','#87607e','#9e0075','#ffddc3','#500800',
    '#1a0800','#4b89b5','#00dfdf','#c8fff9','#2f3415','#ff2646','#ff97aa','#03001a',
    '#c860b1','#c3a136','#7c4f3a','#f99e77','#566464','#d193ff','#2d1f69','#411a34',
    '#af9397','#629e99','#bcdd7b','#ff5d93','#0f2823','#b8bdac','#743b64','#0f000c',
    '#7e6ebc','#9e6b3b','#ff4600','#7e0087','#ffcd3d','#2f3b42','#fda5ff','#89013d',
]

_OV_VIBRANT = [
    '#FF0000','#00CC00','#0000FF','#FFAA00','#FF00FF','#00CCCC',
    '#FF6600','#CC0066','#66CC00','#0066CC','#6600CC','#00CC66',
    '#FF3333','#33AA33','#3333FF','#FFCC33','#FF33CC','#33CCCC',
    '#FF8833','#CC3388','#88CC33','#3388CC','#8833CC','#33CC88',
]

_CUSTOM_PALETTES = {
    'omicverse':    _OV_SC_COLOR,
    'omicverse_56': _OV_CET[:56],
    'omicverse_112': _OV_CET[:112],
    'vibrant':      _OV_VIBRANT,
}


def _color_to_hex(color):
    """Convert any matplotlib-compatible color string to hex."""
    try:
        import matplotlib.colors as mcolors
        return mcolors.to_hex(color)
    except Exception:
        return str(color)


def get_uns_colors(adata, col_name, n_categories):
    """
    Return colors from adata.uns['{col_name}_colors'] if available, else None.
    AnnData stores per-category colors in this key aligned with category order.
    """
    key = f'{col_name}_colors'
    if adata is None or key not in adata.uns:
        return None
    uns = list(adata.uns[key])
    # Cycle if uns has fewer colors than categories
    return [_color_to_hex(uns[i % len(uns)]) for i in range(n_categories)]


def get_uns_colors_for_labels(adata, col_name, labels):
    """
    Return hex colors from adata.uns['{col_name}_colors'] aligned to *labels*.

    This is safer than get_uns_colors because it maps colors by category NAME
    rather than by index, preventing colour mismatches when the category order
    produced by ``vals.astype('category')`` differs from the original order
    stored in ``adata.obs[col_name].cat.categories``.
    """
    key = f'{col_name}_colors'
    if adata is None or key not in adata.uns:
        return None
    uns_colors = list(adata.uns[key])
    # Build a name→hex mapping from the original adata category order
    try:
        orig_col = adata.obs[col_name]
        if hasattr(orig_col, 'cat'):
            orig_cats = list(orig_col.cat.categories)
            color_map = {
                cat: _color_to_hex(uns_colors[i % len(uns_colors)])
                for i, cat in enumerate(orig_cats)
            }
            result = [color_map.get(lbl) for lbl in labels]
            if all(c is not None for c in result):
                return result
    except Exception:
        pass
    # Fallback: assume same order (old behaviour)
    return [_color_to_hex(uns_colors[i % len(uns_colors)]) for i in range(len(labels))]


_PALETTE_NAME_MAP = {
    # Correct case-insensitive names for matplotlib qualitative palettes
    'set1': 'Set1', 'set2': 'Set2', 'set3': 'Set3',
    'paired': 'Paired', 'accent': 'Accent',
    'dark2': 'Dark2', 'pastel1': 'Pastel1', 'pastel2': 'Pastel2',
}


def get_discrete_colors(n_categories, palette_name=None):
    """Get discrete color palette for categorical data."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colors import ListedColormap

    # Normalize palette name (frontend may send lowercase)
    if palette_name:
        palette_name = _PALETTE_NAME_MAP.get(palette_name.lower(), palette_name)

    # OmicVerse custom palettes
    if palette_name in _CUSTOM_PALETTES:
        base = _CUSTOM_PALETTES[palette_name]
        return [base[i % len(base)] for i in range(n_categories)]

    # Default → OmicVerse sc_color
    if not palette_name or palette_name == 'default':
        return [_OV_SC_COLOR[i % len(_OV_SC_COLOR)] for i in range(n_categories)]

    # Matplotlib named palette
    try:
        cmap = plt.get_cmap(palette_name)
        if isinstance(cmap, ListedColormap):
            # Qualitative: pick discrete slots (cycling if needed)
            return [mcolors.to_hex(cmap.colors[i % len(cmap.colors)]) for i in range(n_categories)]
        else:
            # Continuous: spread evenly across [0, 1]
            return [mcolors.to_hex(cmap(i / max(1, n_categories - 1))) for i in range(n_categories)]
    except Exception:
        # Fallback to default
        return [_OV_SC_COLOR[i % len(_OV_SC_COLOR)] for i in range(n_categories)]


# ============================================================================
# Register Blueprints
# ============================================================================

# Kernel blueprint
app.register_blueprint(kernel.bp, url_prefix='/api/kernel')
kernel.bp.state = state

# Files blueprint
app.register_blueprint(files.bp, url_prefix='/api/files')
files.bp.file_root = state.file_root

# Data blueprint
app.register_blueprint(data.bp, url_prefix='/api')
data.bp.state = state
data.bp.upload_folder = app.config['UPLOAD_FOLDER']

# Notebooks blueprint
app.register_blueprint(notebooks.bp, url_prefix='/api/notebooks')
notebooks.bp.notebook_root = state.notebook_root

# Terminal blueprint (PTY-based interactive shell)
app.register_blueprint(terminal_bp)


# ============================================================================
# Code Execution Routes (not in blueprints due to complexity)
# ============================================================================

def _serialize_execution_result(raw_result, max_rows=50, max_cols=20):
    """Serialize execution result with structured table payload when possible."""
    if raw_result is None:
        return {'kind': None, 'text': None}

    try:
        if isinstance(raw_result, pd.DataFrame):
            df = raw_result.iloc[:max_rows, :max_cols].copy()
            df = df.astype(object).where(pd.notna(df), None)
            dtypes = {str(col): str(dtype) for col, dtype in raw_result.dtypes.items()}
            return {
                'kind': 'dataframe',
                'text': None,
                'shape': [int(raw_result.shape[0]), int(raw_result.shape[1])],
                'dtypes': dtypes,
                'table': df.to_dict(orient='split')
            }

        if isinstance(raw_result, pd.Series):
            col_name = str(raw_result.name) if raw_result.name is not None else 'value'
            df = raw_result.to_frame(name=col_name).iloc[:max_rows, :1].copy()
            df = df.astype(object).where(pd.notna(df), None)
            return {
                'kind': 'dataframe',
                'text': None,
                'shape': [int(raw_result.shape[0]), 1],
                'dtypes': {col_name: str(raw_result.dtype)},
                'table': df.to_dict(orient='split')
            }
    except Exception:
        pass

    try:
        return {'kind': 'text', 'text': str(raw_result)}
    except Exception:
        return {'kind': 'text', 'text': '<unserializable result>'}


def _parse_df_limit(value, default_value, minimum, maximum):
    """Parse and clamp DataFrame preview limit from request payload."""
    try:
        parsed = int(value)
    except Exception:
        return default_value
    return max(minimum, min(maximum, parsed))


@app.route('/api/execute_code', methods=['POST'])
def execute_code():
    """Execute Python code with access to current_adata."""
    try:
        payload = request.json if request.json else {}
        code = payload.get('code', '')
        if not code:
            return jsonify({'error': '没有提供代码'}), 400

        kernel_id = normalize_kernel_id(payload.get('kernel_id'))
        timeout = payload.get('timeout', 300)
        df_max_rows = _parse_df_limit(payload.get('df_max_rows'), 50, 1, 500)
        df_max_cols = _parse_df_limit(payload.get('df_max_cols'), 20, 1, 200)
        executor, ns = get_kernel_context(kernel_id, state.kernel_executor, state.kernel_sessions)

        try:
            shared_adata = state.current_adata if kernel_id == 'default.ipynb' else None
            execution = executor.execute(code, shared_adata, user_ns=ns, timeout=timeout)
        except KeyboardInterrupt:
            return jsonify({
                'interrupted': True,
                'output': 'Execution interrupted by user',
                'error': None,
                'figures': [],
                'success': False
            }), 200
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

        result_info = _serialize_execution_result(
            execution.get('result'),
            max_rows=df_max_rows,
            max_cols=df_max_cols
        )

        new_adata = execution.get('adata')
        data_updated = False
        if kernel_id == 'default.ipynb' and new_adata is not None:
            state.current_adata = new_adata
            data_updated = True
            sync_adaptor_with_adata()
            try:
                state.kernel_executor.sync_adata(state.current_adata)
            except Exception:
                pass

        data_info = None
        if data_updated:
            data_info = {
                'filename': state.current_filename,
                'n_cells': state.current_adata.n_obs,
                'n_genes': state.current_adata.n_vars,
                'embeddings': _canonical_embedding_keys(state.current_adata),
                'obs_columns': list(state.current_adata.obs.columns),
                'var_columns': list(state.current_adata.var.columns)
            }

        return jsonify({
            'output': output,
            'result': result_info.get('text'),
            'result_kind': result_info.get('kind'),
            'result_shape': result_info.get('shape'),
            'result_dtypes': result_info.get('dtypes'),
            'result_table': result_info.get('table'),
            'figures': execution.get('figures', []),
            'data_updated': data_updated,
            'data_info': data_info,
            'kernel_id': kernel_id,
            'interrupted': False,
            'success': True
        })

    except Exception as e:
        import traceback
        return jsonify({'error': traceback.format_exc()}), 500


@app.route('/api/execute_code_stream', methods=['POST'])
def execute_code_stream():
    """Execute Python code with streaming output using Server-Sent Events."""
    try:
        payload = request.json if request.json else {}
        code = payload.get('code', '')
        if not code:
            return jsonify({'error': '没有提供代码'}), 400
        kernel_id = normalize_kernel_id(payload.get('kernel_id'))
        timeout = payload.get('timeout', 300)
        df_max_rows = _parse_df_limit(payload.get('df_max_rows'), 50, 1, 500)
        df_max_cols = _parse_df_limit(payload.get('df_max_cols'), 20, 1, 200)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Create a queue for streaming output
    output_queue = queue.Queue()

    # Custom output stream
    class StreamOutput:
        def __init__(self, stream_type='output'):
            self.stream_type = stream_type
            self.buffer = io.StringIO()

        def write(self, text):
            if text:
                output_queue.put((self.stream_type, text))
                self.buffer.write(text)
            return len(text) if text else 0

        def flush(self):
            pass

        def getvalue(self):
            return self.buffer.getvalue()

    def generate():
        """Generator function for SSE."""
        try:
            yield f"data: {json.dumps({'type': 'start'})}\n\n"

            executor, ns = get_kernel_context(kernel_id, state.kernel_executor, state.kernel_sessions)
            execution_result = {'done': False, 'error': None, 'result': None, 'figures': [], 'data_info': None}

            def execute_in_thread():
                try:
                    shared_adata = state.current_adata if kernel_id == 'default.ipynb' else None
                    stdout_stream = StreamOutput('output')
                    stderr_stream = StreamOutput('stderr')

                    result = executor.execute(code, shared_adata, user_ns=ns, timeout=timeout,
                                            stdout=stdout_stream, stderr=stderr_stream)

                    execution_result['result'] = result
                    execution_result['figures'] = result.get('figures', [])

                    new_adata = result.get('adata')
                    if kernel_id == 'default.ipynb' and new_adata is not None:
                        state.current_adata = new_adata
                        sync_adaptor_with_adata()
                        try:
                            state.kernel_executor.sync_adata(state.current_adata)
                        except Exception:
                            pass

                        execution_result['data_info'] = {
                            'filename': state.current_filename,
                            'n_cells': state.current_adata.n_obs,
                            'n_genes': state.current_adata.n_vars,
                            'embeddings': _canonical_embedding_keys(state.current_adata),
                            'obs_columns': list(state.current_adata.obs.columns),
                            'var_columns': list(state.current_adata.var.columns)
                        }

                except Exception as e:
                    execution_result['error'] = str(e)
                finally:
                    execution_result['done'] = True
                    output_queue.put(('done', None))

            # Start execution thread
            exec_thread = threading.Thread(target=execute_in_thread)
            exec_thread.start()

            # Stream output
            while True:
                try:
                    item = output_queue.get(timeout=0.1)
                    if item[0] == 'done':
                        break
                    stream_type, text = item
                    yield f"data: {json.dumps({'type': stream_type, 'text': text})}\n\n"
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                    if execution_result['done']:
                        break

            exec_thread.join(timeout=1)

            # Send completion
            if execution_result['error']:
                yield f"data: {json.dumps({'type': 'error', 'text': execution_result['error']})}\n\n"
            else:
                result_info = {'kind': None, 'text': None}
                if execution_result['result']:
                    result_info = _serialize_execution_result(
                        execution_result['result'].get('result'),
                        max_rows=df_max_rows,
                        max_cols=df_max_cols
                    )

                result_data = {
                    'type': 'complete',
                    'output': execution_result['result'].get('output', '') if execution_result['result'] else '',
                    'error': execution_result['result'].get('error') if execution_result['result'] else None,
                    'result': result_info.get('text'),
                    'result_kind': result_info.get('kind'),
                    'result_shape': result_info.get('shape'),
                    'result_dtypes': result_info.get('dtypes'),
                    'result_table': result_info.get('table'),
                    'figures': execution_result['figures'],
                    'data_updated': execution_result['data_info'] is not None,
                    'data_info': execution_result['data_info']
                }
                yield f"data: {json.dumps(result_data)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


# ============================================================================
# Tools Routes
# ============================================================================

from utils.adata_helpers import (
    analyze_data_state as _analyze_data_state,
    canonical_embedding_keys as _canonical_embedding_keys,
    resolve_embedding_key as _resolve_embedding_key,
)


def _snapshot_adata(adata):
    """Capture the structural state of an AnnData for before/after comparison."""
    return {
        'shape':  (adata.n_obs, adata.n_vars),
        'obs':    set(adata.obs.columns),
        'var':    set(adata.var.columns),
        'uns':    set(adata.uns.keys()),
        'obsm':   set(adata.obsm.keys()),
        'obsp':   set(adata.obsp.keys()),
        'layers': set(adata.layers.keys()),
    }


def _diff_adata(before, after, duration=0.0):
    """Return a structured diff between two snapshots."""
    diff = {
        'shape_before': list(before['shape']),
        'shape_after':  list(after['shape']),
        'duration':     round(duration, 3),
        'changes':      {},
    }
    for slot in ('obs', 'var', 'uns', 'obsm', 'obsp', 'layers'):
        added   = sorted(after[slot] - before[slot])
        removed = sorted(before[slot] - after[slot])
        if added or removed:
            diff['changes'][slot] = {}
            if added:   diff['changes'][slot]['added']   = added
            if removed: diff['changes'][slot]['removed'] = removed
    return diff


@app.route('/api/tools/<tool>', methods=['POST'])
def run_tool(tool):
    """Run a single-cell analysis tool and return a structural diff of the AnnData."""
    if state.current_adata is None:
        return jsonify({'error': 'No data loaded'}), 400

    import time as _time
    import numpy as _np, re as _re

    try:
        params = request.json if request.json else {}
        snap_before = _snapshot_adata(state.current_adata)
        t0 = _time.time()
        predicted_col  = None  # set by annotation tools
        pseudotime_col = None  # set by trajectory tools
        tool_figures   = []    # figures produced by tools (e.g., PAGA)

        # ── Preprocessing ────────────────────────────────────────────────────
        if tool == 'normalize':
            target_sum = params.get('target_sum', 1e4)
            sc.pp.normalize_total(state.current_adata, target_sum=target_sum)

        elif tool == 'log1p':
            sc.pp.log1p(state.current_adata)

        elif tool == 'scale':
            max_value = params.get('max_value', 10)
            sc.pp.scale(state.current_adata, max_value=max_value)

        elif tool == 'hvg':
            n_genes = params.get('n_genes', 2000)
            method  = params.get('method', 'seurat')
            sc.pp.highly_variable_genes(state.current_adata, flavor=method, n_top_genes=n_genes)
            state.current_adata = state.current_adata[:, state.current_adata.var.highly_variable].copy()

        # ── QC ───────────────────────────────────────────────────────────────
        elif tool == 'filter_cells':
            for key in ('min_counts', 'min_genes', 'max_counts', 'max_genes'):
                v = params.get(key)
                if v is None or v == '':
                    continue
                try:
                    v = int(v)
                except Exception:
                    continue
                sc.pp.filter_cells(state.current_adata, **{key: v})

        elif tool == 'filter_genes':
            min_cells    = params.get('min_cells')
            g_min_counts = params.get('min_counts', params.get('g_min_counts'))
            max_cells    = params.get('max_cells')
            g_max_counts = params.get('max_counts', params.get('g_max_counts'))
            if g_min_counts not in (None, ''):
                sc.pp.filter_genes(state.current_adata, min_counts=int(g_min_counts))
            if min_cells not in (None, ''):
                sc.pp.filter_genes(state.current_adata, min_cells=int(min_cells))
            X = (state.current_adata.X.toarray()
                 if hasattr(state.current_adata.X, 'toarray')
                 else _np.asarray(state.current_adata.X))
            if max_cells not in (None, ''):
                expr_cells = (X > 0).sum(axis=0)
                expr_cells = _np.asarray(expr_cells).ravel()
                state.current_adata._inplace_subset_var(expr_cells <= int(max_cells))
                X = (state.current_adata.X.toarray()
                     if hasattr(state.current_adata.X, 'toarray')
                     else _np.asarray(state.current_adata.X))
            if g_max_counts not in (None, ''):
                sums = _np.asarray(X).sum(axis=0).ravel()
                state.current_adata._inplace_subset_var(sums <= float(g_max_counts))

        elif tool == 'filter_outliers':
            req_prefixes = params.get('mt_prefixes')
            if req_prefixes:
                mt_prefixes = [p.strip() for p in str(req_prefixes).split(',') if p.strip()]
            else:
                mt_prefixes = []
                for name in state.current_adata.var_names.astype(str):
                    if _re.match(r'^(mt|MT|Mt|mT)[-_].+', name):
                        mt_prefixes = list({n[:3] for n in state.current_adata.var_names.astype(str) if len(n) >= 3})
                        break
                if not mt_prefixes:
                    mt_prefixes = ['MT-', 'mt-']
            lname = _np.array([str(x) for x in state.current_adata.var_names])
            mt_mask = _np.zeros(len(lname), dtype=bool)
            for pref in mt_prefixes:
                mt_mask |= _np.char.startswith(lname, pref)
            state.current_adata.var['mt'] = mt_mask
            upper = _np.char.upper(lname)
            state.current_adata.var['ribo'] = (
                _np.char.startswith(upper, 'RPS') | _np.char.startswith(upper, 'RPL'))
            state.current_adata.var['hb'] = _np.array(
                [bool(_re.search(r'^(HB(?!P))', n, _re.I)) for n in lname])
            sc.pp.calculate_qc_metrics(
                state.current_adata, qc_vars=['mt', 'ribo', 'hb'], inplace=True, log1p=True)
            keep = _np.ones(state.current_adata.n_obs, dtype=bool)
            for pct_param, col_hint in [('max_mt_percent', 'mt'),
                                         ('max_ribo_percent', 'ribo'),
                                         ('max_hb_percent', 'hb')]:
                threshold = params.get(pct_param)
                if threshold is None:
                    continue
                pct_col = next((c for c in state.current_adata.obs.columns
                                if 'pct' in c.lower() and col_hint in c.lower()), None)
                if pct_col:
                    keep &= (state.current_adata.obs[pct_col].astype(float)
                             <= float(threshold)).values
            if keep.sum() != state.current_adata.n_obs:
                state.current_adata._inplace_subset_obs(keep)

        elif tool == 'doublets':
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
                sc.pp.scrublet(state.current_adata, batch_key=batch_key, **scrublet_kwargs)
            except Exception as e:
                return jsonify({'error': f'Scrublet 执行失败: {e}'}), 400
            pred_col = next((c for c in ('predicted_doublet', 'predicted_doublets')
                             if c in state.current_adata.obs.columns), None)
            if pred_col:
                keep = ~state.current_adata.obs[pred_col].astype(bool)
                state.current_adata._inplace_subset_obs(keep.values)

        # ── Dimensionality reduction ──────────────────────────────────────────
        elif tool == 'pca':
            n_comps = params.get('n_comps', 50)
            sc.pp.pca(state.current_adata,
                      n_comps=min(n_comps, min(state.current_adata.shape) - 1))

        elif tool == 'neighbors':
            sc.pp.neighbors(state.current_adata,
                            n_neighbors=params.get('n_neighbors', 15),
                            n_pcs=params.get('n_pcs', 50))

        elif tool == 'umap':
            if 'neighbors' not in state.current_adata.uns:
                sc.pp.neighbors(state.current_adata, n_neighbors=15, n_pcs=30)
            sc.tl.umap(state.current_adata, min_dist=params.get('min_dist', 0.5))

        elif tool == 'tsne':
            sc.tl.tsne(state.current_adata, perplexity=params.get('perplexity', 30))

        # ── Clustering ───────────────────────────────────────────────────────
        elif tool == 'leiden':
            sc.tl.leiden(state.current_adata, resolution=params.get('resolution', 1.0))

        elif tool == 'louvain':
            sc.tl.louvain(state.current_adata, resolution=params.get('resolution', 1.0))

        # ── Cell Annotation ──────────────────────────────────────────────────
        elif tool == 'celltypist':
            import omicverse as ov
            pkl_path = params.get('pkl_path', '').strip()
            if not pkl_path:
                return jsonify({'error': 'CellTypist 模型路径不能为空，请先下载模型'}), 400
            if not os.path.exists(pkl_path):
                return jsonify({'error': f'模型文件不存在: {pkl_path}'}), 400
            obj = ov.single.Annotation(state.current_adata)
            obj.add_reference_pkl(pkl_path)
            obj.annotate(method='celltypist')
            predicted_col = 'celltypist_prediction'

        elif tool == 'gpt4celltype':
            import omicverse as ov
            cluster_key = params.get('cluster_key', 'leiden')
            if cluster_key not in state.current_adata.obs.columns:
                return jsonify({'error': f'聚类键 {cluster_key} 不存在，请先运行聚类'}), 400
            api_key = params.get('api_key', '').strip()
            if api_key:
                os.environ['AGI_API_KEY'] = api_key
            kw = dict(
                tissuename=params.get('tissuename', ''),
                speciename=params.get('speciename', 'human'),
                model=params.get('model', 'qwen-plus'),
                provider=params.get('provider', 'qwen'),
                topgenenumber=int(params.get('topgenenumber', 10)),
            )
            base_url = params.get('base_url', '').strip()
            if base_url:
                kw['base_url'] = base_url
            obj = ov.single.Annotation(state.current_adata)
            obj.annotate(method='gpt4celltype', cluster_key=cluster_key, **kw)
            predicted_col = 'gpt4celltype_prediction'

        elif tool == 'scsa':
            import omicverse as ov
            cluster_key = params.get('cluster_key', 'leiden')
            if cluster_key not in state.current_adata.obs.columns:
                return jsonify({'error': f'聚类键 {cluster_key} 不存在，请先运行聚类'}), 400
            db_path = params.get('db_path', '').strip()
            obj = ov.single.Annotation(state.current_adata)
            if db_path and os.path.exists(db_path):
                obj.add_reference_scsa_db(db_path)
            obj.annotate(
                method='scsa',
                cluster_key=cluster_key,
                foldchange=float(params.get('foldchange', 1.5)),
                pvalue=float(params.get('pvalue', 0.05)),
                celltype=params.get('celltype', 'normal'),
                target=params.get('target', 'cellmarker'),
                tissue=params.get('tissue', 'All'),
            )
            predicted_col = 'scsa_prediction'

        # ── Trajectory Analysis ──────────────────────────────────────────────
        elif tool == 'diffusion_map':
            import omicverse as ov
            groupby = params.get('groupby', 'leiden')
            use_rep = params.get('use_rep', 'X_pca')
            n_comps = int(params.get('n_comps', 50))
            origin  = params.get('origin_cells', '').strip()
            basis   = params.get('basis', 'X_umap')
            if use_rep not in state.current_adata.obsm:
                use_rep = next((k for k in state.current_adata.obsm if 'pca' in k.lower()),
                               list(state.current_adata.obsm.keys())[0])
            if basis not in state.current_adata.obsm:
                basis = list(state.current_adata.obsm.keys())[0]
            Traj = ov.single.TrajInfer(state.current_adata, basis=basis,
                                       groupby=groupby, use_rep=use_rep, n_comps=n_comps)
            if origin:
                Traj.set_origin_cells(origin)
            Traj.inference(method='diffusion_map')
            pseudotime_col = 'dpt_pseudotime'

        elif tool == 'slingshot':
            import omicverse as ov
            groupby    = params.get('groupby', 'leiden')
            use_rep    = params.get('use_rep', 'X_pca')
            n_comps    = int(params.get('n_comps', 50))
            origin     = params.get('origin_cells', '').strip()
            terminal   = params.get('terminal_cells', '').strip()
            num_epochs = int(params.get('num_epochs', 1))
            basis      = params.get('basis', 'X_umap')
            if use_rep not in state.current_adata.obsm:
                use_rep = next((k for k in state.current_adata.obsm if 'pca' in k.lower()),
                               list(state.current_adata.obsm.keys())[0])
            if basis not in state.current_adata.obsm:
                basis = list(state.current_adata.obsm.keys())[0]
            Traj = ov.single.TrajInfer(state.current_adata, basis=basis,
                                       groupby=groupby, use_rep=use_rep, n_comps=n_comps)
            if origin:
                Traj.set_origin_cells(origin)
            if terminal:
                Traj.set_terminal_cells([t.strip() for t in terminal.split(',') if t.strip()])
            Traj.inference(method='slingshot', num_epochs=num_epochs)
            pseudotime_col = 'slingshot_pseudotime'

        elif tool == 'palantir':
            import omicverse as ov
            groupby       = params.get('groupby', 'leiden')
            use_rep       = params.get('use_rep', 'X_pca')
            n_comps       = int(params.get('n_comps', 50))
            origin        = params.get('origin_cells', '').strip()
            terminal      = params.get('terminal_cells', '').strip()
            num_waypoints = int(params.get('num_waypoints', 500))
            basis         = params.get('basis', 'X_umap')
            if use_rep not in state.current_adata.obsm:
                use_rep = next((k for k in state.current_adata.obsm if 'pca' in k.lower()),
                               list(state.current_adata.obsm.keys())[0])
            if basis not in state.current_adata.obsm:
                basis = list(state.current_adata.obsm.keys())[0]
            Traj = ov.single.TrajInfer(state.current_adata, basis=basis,
                                       groupby=groupby, use_rep=use_rep, n_comps=n_comps)
            if origin:
                Traj.set_origin_cells(origin)
            if terminal:
                Traj.set_terminal_cells([t.strip() for t in terminal.split(',') if t.strip()])
            Traj.inference(method='palantir', num_waypoints=num_waypoints)
            pseudotime_col = 'palantir_pseudotime'

        elif tool == 'paga':
            import omicverse as ov
            import base64
            import matplotlib.pyplot as plt
            use_time_prior = params.get('use_time_prior', '').strip()
            groups         = params.get('groups', 'leiden')
            use_rep        = params.get('use_rep', 'X_pca')
            basis          = params.get('basis', 'umap')   # plot_paga uses no X_ prefix
            if use_rep not in state.current_adata.obsm:
                use_rep = next((k for k in state.current_adata.obsm if 'pca' in k.lower()), None)
            if use_rep:
                sc.pp.neighbors(state.current_adata, use_rep=use_rep)
            elif 'neighbors' not in state.current_adata.uns:
                sc.pp.neighbors(state.current_adata)
            paga_kw = dict(vkey='paga', groups=groups)
            if use_time_prior and use_time_prior in state.current_adata.obs.columns:
                paga_kw['use_time_prior'] = use_time_prior
            ov.utils.cal_paga(state.current_adata, **paga_kw)
            before_figs = set(plt.get_fignums())
            ov.utils.plot_paga(state.current_adata, basis=basis, size=50, alpha=0.1,
                               title=f'PAGA ({use_time_prior or groups})',
                               min_edge_width=2, node_size_scale=1.5,
                               show=False, legend_loc=False)
            after_figs = set(plt.get_fignums())
            for fig_num in [n for n in after_figs if n not in before_figs]:
                fig = plt.figure(fig_num)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                tool_figures.append(base64.b64encode(buf.getvalue()).decode('ascii'))
                plt.close(fig)

        elif tool == 'sctour':
            import omicverse as ov
            groupby          = params.get('groupby', 'leiden')
            use_rep          = params.get('use_rep', 'X_pca')
            n_comps          = int(params.get('n_comps', 50))
            alpha_recon_lec  = float(params.get('alpha_recon_lec', 0.5))
            alpha_recon_lode = float(params.get('alpha_recon_lode', 0.5))
            basis            = params.get('basis', 'X_umap')
            if use_rep not in state.current_adata.obsm:
                use_rep = next((k for k in state.current_adata.obsm if 'pca' in k.lower()),
                               list(state.current_adata.obsm.keys())[0])
            if basis not in state.current_adata.obsm:
                basis = list(state.current_adata.obsm.keys())[0]
            Traj = ov.single.TrajInfer(state.current_adata, basis=basis,
                                       groupby=groupby, use_rep=use_rep, n_comps=n_comps)
            Traj.inference(method='sctour',
                           alpha_recon_lec=alpha_recon_lec, alpha_recon_lode=alpha_recon_lode)
            pseudotime_col = 'sctour_pseudotime'

        else:
            return jsonify({'error': f'Unknown tool: {tool}'}), 400

        duration = _time.time() - t0
        snap_after = _snapshot_adata(state.current_adata)
        diff = _diff_adata(snap_before, snap_after, duration)

        sync_adaptor_with_adata()

        resp = {
            'success':       True,
            'n_cells':       state.current_adata.n_obs,
            'n_genes':       state.current_adata.n_vars,
            'embeddings':    [k.replace('X_', '') for k in state.current_adata.obsm.keys()],
            'obs_columns':   list(state.current_adata.obs.columns),
            'var_columns':   list(state.current_adata.var.columns),
            'uns_keys':      list(state.current_adata.uns.keys()),
            'layers':        list(state.current_adata.layers.keys()),
            'diff':          diff,
        }
        resp['data_state'] = _analyze_data_state(state.current_adata)
        if predicted_col:
            resp['predicted_col'] = predicted_col
        if pseudotime_col and pseudotime_col in state.current_adata.obs.columns:
            resp['pseudotime_col'] = pseudotime_col
        if tool_figures:
            resp['figures'] = tool_figures
        return jsonify(resp)

    except Exception as e:
        logging.error(f"Tool {tool} failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """Return current adata state so the frontend can restore UI after a page refresh."""
    if state.current_adata is None:
        return jsonify({'loaded': False})
    import numpy as _np_stat
    adata_stat = state.current_adata
    # Build obsm_ndims: {key: ndim} for the custom axis dim picker
    obsm_ndims = {}
    for k, v in adata_stat.obsm.items():
        try:
            arr = _np_stat.asarray(v)
            obsm_ndims[k] = int(arr.shape[1]) if arr.ndim >= 2 else 1
        except Exception:
            obsm_ndims[k] = 2
    return jsonify({
        'loaded':        True,
        'filename':      state.current_filename or 'data.h5ad',
        'n_cells':       adata_stat.n_obs,
        'n_genes':       adata_stat.n_vars,
        'embeddings':    _canonical_embedding_keys(adata_stat),
        'obs_columns':   list(adata_stat.obs.columns),
        'var_columns':   list(adata_stat.var.columns),
        'uns_keys':      list(adata_stat.uns.keys()),
        'layers':        list(adata_stat.layers.keys()),
        'data_state':    _analyze_data_state(adata_stat),
        'preview_mode':  getattr(state, 'is_preview_mode', False),
        'obsm_ndims':    obsm_ndims,
    })


# ============================================================================
# Trajectory Visualization Endpoints
# ============================================================================

@app.route('/api/trajectory/plot_embedding', methods=['POST'])
def trajectory_plot_embedding():
    """Render a pseudotime-colored embedding with optional PAGA overlay."""
    if state.current_adata is None:
        return jsonify({'error': 'No data loaded'}), 400
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import base64
        import omicverse as ov
        import numpy as _np

        params = request.json or {}
        pseudotime_col      = params.get('pseudotime_col', '').strip()
        basis               = params.get('basis', 'X_umap')
        cmap                = params.get('cmap', 'Reds')
        point_size          = float(params.get('point_size', 3))
        paga_overlay        = bool(params.get('paga_overlay', False))
        paga_groups         = params.get('paga_groups', '').strip()
        paga_min_edge_width = float(params.get('paga_min_edge_width', 2))
        paga_node_scale     = float(params.get('paga_node_size_scale', 1.5))

        adata = state.current_adata

        # Auto-detect pseudotime column
        if not pseudotime_col:
            for col in ('dpt_pseudotime', 'palantir_pseudotime',
                        'slingshot_pseudotime', 'sctour_pseudotime'):
                if col in adata.obs.columns:
                    pseudotime_col = col
                    break
        if not pseudotime_col or pseudotime_col not in adata.obs.columns:
            return jsonify({'error': '未找到拟时序列，请先运行轨迹推断工具'}), 400

        # Validate / fall-back basis
        if basis not in adata.obsm:
            basis = next((k for k in adata.obsm if 'umap' in k.lower()),
                         list(adata.obsm.keys())[0])

        coords    = adata.obsm[basis]
        pseudotime = _np.array(adata.obs[pseudotime_col].values, dtype=float)

        fig, ax = plt.subplots(figsize=(6, 5))
        sc_plot = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=pseudotime, cmap=cmap, s=point_size,
            alpha=0.8, linewidths=0, rasterized=True
        )
        plt.colorbar(sc_plot, ax=ax, label=pseudotime_col,
                     fraction=0.046, pad=0.04)

        # PAGA overlay
        if paga_overlay:
            basis_name = basis.replace('X_', '')
            # Ensure PAGA is computed with correct groups
            try:
                if paga_groups and paga_groups in adata.obs.columns:
                    if 'neighbors' not in adata.uns:
                        sc.pp.neighbors(adata)
                    paga_kw = dict(vkey='paga', groups=paga_groups)
                    if pseudotime_col in adata.obs.columns:
                        paga_kw['use_time_prior'] = pseudotime_col
                    ov.utils.cal_paga(adata, **paga_kw)
                ov.utils.plot_paga(
                    adata, basis=basis_name, size=0, alpha=0.0,
                    title='', min_edge_width=paga_min_edge_width,
                    node_size_scale=paga_node_scale,
                    show=False, legend_loc=False, ax=ax
                )
            except Exception as paga_err:
                logging.warning(f"PAGA overlay failed: {paga_err}")

        basis_label = basis.replace('X_', '').upper()
        ax.set_xlabel(f'{basis_label} 1', fontsize=9)
        ax.set_ylabel(f'{basis_label} 2', fontsize=9)
        ax.set_title(pseudotime_col, fontsize=10)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        plt.close(fig)
        return jsonify({'figure': base64.b64encode(buf.getvalue()).decode('ascii')})

    except Exception as e:
        logging.error(f"trajectory_plot_embedding failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/trajectory/plot_heatmap', methods=['POST'])
def trajectory_plot_heatmap():
    """Render a gene-expression × pseudotime heatmap."""
    if state.current_adata is None:
        return jsonify({'error': 'No data loaded'}), 400
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import base64
        import numpy as _np

        params = request.json or {}
        genes_raw      = params.get('genes', '')
        pseudotime_col = params.get('pseudotime_col', '').strip()
        layer          = params.get('layer', '').strip()
        n_bins         = max(5, int(params.get('n_bins', 50)))
        cmap           = params.get('cmap', 'RdBu_r')

        adata = state.current_adata

        # Parse gene list (comma / newline separated)
        genes = [g.strip() for g in genes_raw.replace('\n', ',').split(',')
                 if g.strip()]
        if not genes:
            return jsonify({'error': '请输入至少一个基因名称'}), 400
        genes = [g for g in genes if g in adata.var_names]
        if not genes:
            return jsonify({'error': '所有输入基因均不在数据集中，请检查基因名称'}), 400

        # Auto-detect pseudotime
        if not pseudotime_col:
            for col in ('dpt_pseudotime', 'palantir_pseudotime',
                        'slingshot_pseudotime', 'sctour_pseudotime'):
                if col in adata.obs.columns:
                    pseudotime_col = col
                    break
        if not pseudotime_col or pseudotime_col not in adata.obs.columns:
            return jsonify({'error': '未找到拟时序列，请先运行轨迹推断工具'}), 400

        # Build expression matrix (slice genes first to avoid densifying all features)
        if layer and layer in adata.layers:
            X = adata.layers[layer]
        else:
            X = adata.X

        # Gene indices
        var_names = list(adata.var_names)
        gene_idx  = [var_names.index(g) for g in genes]
        expr      = X[:, gene_idx]           # (n_cells, n_genes)
        if hasattr(expr, 'toarray'):
            expr = expr.toarray()
        else:
            expr = _np.asarray(expr)

        # Sort by pseudotime, drop NaN cells
        pt = _np.array(adata.obs[pseudotime_col].values, dtype=float)
        valid = ~_np.isnan(pt)
        pt, expr = pt[valid], expr[valid]
        order     = _np.argsort(pt)
        expr      = expr[order]              # (n_valid_cells, n_genes)

        # Bin into n_bins equal-size groups
        n_cells  = len(order)
        bin_size = max(1, n_cells // n_bins)
        bins     = []
        for i in range(n_bins):
            s, e = i * bin_size, min((i + 1) * bin_size, n_cells)
            if s >= n_cells:
                break
            bins.append(expr[s:e].mean(axis=0))
        heatmap = _np.array(bins).T           # (n_genes, n_actual_bins)

        # Row-wise 0-1 normalisation per gene
        rmin = heatmap.min(axis=1, keepdims=True)
        rmax = heatmap.max(axis=1, keepdims=True)
        denom = _np.where(rmax - rmin > 1e-10, rmax - rmin, 1.0)
        heatmap_norm = (heatmap - rmin) / denom

        # Draw
        n_actual = heatmap_norm.shape[1]
        fig_h = max(3, len(genes) * 0.55 + 1.5)
        fig, ax = plt.subplots(figsize=(8, fig_h))
        im = ax.imshow(heatmap_norm, aspect='auto', cmap=cmap,
                       interpolation='bilinear', vmin=0, vmax=1)
        ax.set_yticks(range(len(genes)))
        ax.set_yticklabels(genes, fontsize=9)
        tick_pos = [0, n_actual // 4, n_actual // 2, 3 * n_actual // 4, n_actual - 1]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=8)
        ax.set_xlabel('Pseudotime →', fontsize=9)
        ax.set_title(f'Gene trends  ({pseudotime_col})', fontsize=10)
        plt.colorbar(im, ax=ax, label='Norm. expression', fraction=0.025, pad=0.04)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
        plt.close(fig)
        return jsonify({'figure': base64.b64encode(buf.getvalue()).decode('ascii')})

    except Exception as e:
        logging.error(f"trajectory_plot_heatmap failed: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Differential Expression (DEG) Endpoints
# ============================================================================

@app.route('/api/deg/get_groups', methods=['GET'])
def deg_get_groups():
    """Return unique values for a given obs column (for group selectors)."""
    if state.current_adata is None:
        return jsonify({'groups': []})
    col = request.args.get('col', '').strip()
    if not col or col not in state.current_adata.obs.columns:
        return jsonify({'groups': []})
    try:
        groups = state.current_adata.obs[col].dropna().astype(str).unique().tolist()
        return jsonify({'groups': sorted(groups)})
    except Exception as e:
        return jsonify({'groups': [], 'error': str(e)})


@app.route('/api/deg/analyze', methods=['POST'])
def deg_analyze():
    """Run DEG analysis and store results; return summary statistics."""
    if state.current_adata is None:
        return jsonify({'error': 'No data loaded'}), 400
    try:
        import omicverse as ov
        import numpy as _np

        params = request.json or {}
        condition     = params.get('condition', '').strip()
        ctrl_group    = params.get('ctrl_group', '').strip()
        test_group    = params.get('test_group', '').strip()
        celltype_key  = params.get('celltype_key', '').strip()
        celltype_group = params.get('celltype_group', [])   # list of strings
        method        = params.get('method', 'wilcoxon').strip()
        max_cells     = int(params.get('max_cells', 100000))

        # Validate required fields
        if not condition or condition not in state.current_adata.obs.columns:
            return jsonify({'error': f'条件列 "{condition}" 不存在，请检查 obs 列名'}), 400
        if not ctrl_group:
            return jsonify({'error': '请选择对照组（ctrl_group）'}), 400
        if not test_group:
            return jsonify({'error': '请选择实验组（test_group）'}), 400
        if ctrl_group == test_group:
            return jsonify({'error': '对照组与实验组不能相同'}), 400
        if not celltype_key or celltype_key not in state.current_adata.obs.columns:
            return jsonify({'error': f'细胞类型列 "{celltype_key}" 不存在，请检查 obs 列名'}), 400

        # Build DEG object
        deg_obj = ov.single.DEG(
            state.current_adata,
            condition=condition,
            ctrl_group=ctrl_group,
            test_group=test_group,
            method=method,
        )

        # Determine cell type groups
        if not celltype_group:
            ct_groups = None  # use all cell types
        else:
            ct_groups = celltype_group

        deg_obj.run(
            celltype_key=celltype_key,
            celltype_group=ct_groups,
            max_cells=max_cells,
        )

        results = deg_obj.get_results()

        # Store results for subsequent volcano/violin requests
        state.deg_results  = results
        state.deg_method   = method
        state.deg_condition = condition  # keep for violin groupby default

        # Build summary stats
        if method in ('wilcoxon', 't-test'):
            if 'log2FC' in results.columns and 'sig' in results.columns:
                n_sig_up   = int(((results['sig'] == 'sig') & (results['log2FC'] > 0)).sum())
                n_sig_down = int(((results['sig'] == 'sig') & (results['log2FC'] < 0)).sum())
            else:
                n_sig_up, n_sig_down = 0, 0
        else:
            n_sig_up, n_sig_down = 0, 0
        n_total = len(results)

        # Serialize ALL results for client-side filtering
        keep_cols = ['log2FC', 'pvalue', 'padj', 'sig']
        for col in ('pct_ctrl', 'pct_test'):
            if col in results.columns:
                keep_cols.append(col)
        res_js = results[keep_cols].copy()
        if 'pct_ctrl' not in res_js.columns:
            res_js['pct_ctrl'] = 0.0
        if 'pct_test' not in res_js.columns:
            res_js['pct_test'] = 0.0
        # Round floats and sanitize NaN/Inf for JSON
        res_js['log2FC']   = res_js['log2FC'].round(4)
        res_js['pct_ctrl'] = res_js['pct_ctrl'].round(2)
        res_js['pct_test'] = res_js['pct_test'].round(2)
        res_js = res_js.replace([float('inf'), float('-inf')], [999.0, -999.0])
        res_js = res_js.fillna(0.0)
        res_js = res_js.reset_index().rename(columns={'index': 'gene'})
        all_results = res_js.to_dict(orient='records')

        return jsonify({
            'n_total':     n_total,
            'n_sig_up':    n_sig_up,
            'n_sig_down':  n_sig_down,
            'method':      method,
            'condition':   condition,
            'all_results': all_results,
        })

    except Exception as e:
        logging.error(f"deg_analyze failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/deg/plot_volcano', methods=['POST'])
def deg_plot_volcano():
    """Generate a volcano plot from the stored DEG results using ov.pl.volcano."""
    if state.deg_results is None:
        return jsonify({'error': '请先运行 DEG 分析'}), 400
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as _np
        import base64
        import omicverse as ov

        params = request.json or {}
        fc_thresh   = float(params.get('fc_thresh', 1.0))
        padj_thresh = float(params.get('padj_thresh', 0.05))
        label_top   = int(params.get('label_top', 10))

        res = state.deg_results
        method = state.deg_method or 'wilcoxon'
        condition_label = getattr(state, 'deg_condition', '') or 'Volcano Plot'

        if method in ('wilcoxon', 't-test'):
            if 'log2FC' not in res.columns or 'padj' not in res.columns:
                return jsonify({'error': '结果缺少 log2FC 或 padj 列'}), 400

            # ov.pl.volcano expects sig column with 'up'/'down'/'normal'
            res_v = res.copy()
            res_v['sig'] = 'normal'
            res_v.loc[(res_v['log2FC'] >  fc_thresh) & (res_v['padj'] < padj_thresh), 'sig'] = 'up'
            res_v.loc[(res_v['log2FC'] < -fc_thresh) & (res_v['padj'] < padj_thresh), 'sig'] = 'down'

            ax = ov.pl.volcano(res_v, pval_name='padj', fc_name='log2FC',
                               pval_threshold=padj_thresh,
                               fc_max=fc_thresh, fc_min=-fc_thresh,
                               plot_genes_num=label_top,
                               figsize=(6, 5), title=condition_label)
            fig = ax.get_figure()

        else:
            # memento-de: use de_coef vs -log10(de_pval) — plain scatter fallback
            if 'de_coef' not in res.columns or 'de_pval' not in res.columns:
                return jsonify({'error': '结果缺少 de_coef 或 de_pval 列'}), 400
            coef = _np.array(res['de_coef'].values, dtype=float)
            neg_logp = -_np.log10(_np.where(res['de_pval'] > 0, res['de_pval'], 1e-300))

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(coef, neg_logp, s=15, c='#7aa2f7', alpha=0.7, linewidths=0, rasterized=True)
            ax.set_xlabel('DE coefficient', fontsize=10)
            ax.set_ylabel('-log₁₀(p-value)', fontsize=10)
            ax.set_title('Volcano Plot (memento-de)', fontsize=11)
            for spine in ('top', 'right'):
                ax.spines[spine].set_visible(False)
            plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
        plt.close(fig)
        return jsonify({'figure': base64.b64encode(buf.getvalue()).decode('ascii')})

    except Exception as e:
        logging.error(f"deg_plot_volcano failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/deg/plot_violin', methods=['POST'])
def deg_plot_violin():
    """Generate violin plots for specified genes using ov.pl.violin."""
    if state.current_adata is None:
        return jsonify({'error': 'No data loaded'}), 400
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import omicverse as ov
        import base64

        params    = request.json or {}
        genes_raw = params.get('genes', '')
        groupby   = params.get('groupby', '').strip()
        layer     = params.get('layer', '').strip() or None

        genes = [g.strip() for g in genes_raw.replace('\n', ',').split(',') if g.strip()]
        if not genes:
            return jsonify({'error': '请输入至少一个基因名称'}), 400

        adata = state.current_adata
        genes = [g for g in genes if g in adata.var_names]
        if not genes:
            return jsonify({'error': '所有输入基因均不在数据集中，请检查基因名称'}), 400

        if not groupby or groupby not in adata.obs.columns:
            groupby = getattr(state, 'deg_condition', None) or ''
        if not groupby or groupby not in adata.obs.columns:
            return jsonify({'error': '分组列不存在，请选择有效的 obs 列'}), 400

        n = len(genes)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig_w = max(5, ncols * 3.5)
        fig_h = max(4, nrows * 3.2)

        # Use a single gene string when n==1 (ov.pl.violin prefers that)
        keys = genes[0] if n == 1 else genes

        ov.pl.violin(adata, keys=keys, groupby=groupby,
                     layer=layer, stripplot=True, jitter=True,
                     show=False, show_boxplot=False, show_means=True,
                     rotation=30, figsize=(fig_w, fig_h))
        fig = plt.gcf()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
        plt.close(fig)
        return jsonify({'figure': base64.b64encode(buf.getvalue()).decode('ascii')})

    except Exception as e:
        logging.error(f"deg_plot_violin failed: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# DCT (Differential Cell Type) Analysis Endpoints
# ============================================================================

@app.route('/api/dct/analyze', methods=['POST'])
def dct_analyze():
    """Run DCT analysis (scCODA or milopy) and store results."""
    if state.current_adata is None:
        return jsonify({'error': 'No data loaded'}), 400
    try:
        import scanpy as sc
        import numpy as _np
        import pandas as pd

        params         = request.json or {}
        condition      = params.get('condition', '').strip()
        ctrl_group     = params.get('ctrl_group', '').strip()
        test_group     = params.get('test_group', '').strip()
        cell_type_key  = params.get('cell_type_key', '').strip()
        sample_key     = params.get('sample_key', '').strip() or None
        method         = params.get('method', 'sccoda').strip()
        use_rep        = params.get('use_rep', 'X_pca').strip()
        est_fdr        = float(params.get('est_fdr', 0.2))
        n_neighbors    = int(params.get('n_neighbors', 150))
        prop           = float(params.get('prop', 0.1))

        adata = state.current_adata

        # Validate common parameters
        for col, label in [(condition, '条件列'), (cell_type_key, '细胞类型列')]:
            if not col or col not in adata.obs.columns:
                return jsonify({'error': f'{label} "{col}" 不存在，请检查 obs 列名'}), 400
        if not ctrl_group or not test_group:
            return jsonify({'error': '请选择对照组和实验组'}), 400
        if ctrl_group == test_group:
            return jsonify({'error': '对照组与实验组不能相同'}), 400

        # Filter to relevant conditions
        adata_sub = adata[adata.obs[condition].astype(str).isin([ctrl_group, test_group])].copy()
        if len(adata_sub) == 0:
            return jsonify({'error': f'在条件列 "{condition}" 中未找到指定分组的细胞'}), 400

        # Store context for plotting
        state.dct_condition     = condition
        state.dct_ctrl_group    = ctrl_group
        state.dct_test_group    = test_group
        state.dct_cell_type_key = cell_type_key
        state.dct_sample_key    = sample_key
        state.dct_method        = method
        state.dct_adata         = adata_sub

        if method == 'sccoda':
            import pertpy as pt
            if sample_key and sample_key not in adata_sub.obs.columns:
                return jsonify({'error': f'样本列 "{sample_key}" 不存在于 obs，请检查列名是否正确'}), 400

            model = pt.tl.Sccoda()
            sccoda_data = model.load(
                adata_sub,
                type='cell_level',
                cell_type_identifier=cell_type_key,
                sample_identifier=sample_key,
                covariate_obs=[condition],
            )
            sccoda_data = model.prepare(
                sccoda_data,
                modality_key='coda',
                formula=condition,          # use actual column name
            )
            model.run_nuts(sccoda_data, modality_key='coda',
                           num_samples=5000, num_warmup=500)
            model.credible_effects(sccoda_data, modality_key='coda')
            model.set_fdr(sccoda_data, modality_key='coda', est_fdr=est_fdr)

            results = model.get_effect_df(sccoda_data, modality_key='coda')

            state.dct_model   = model
            state.dct_data    = sccoda_data
            state.dct_results = results

            # Build results for client
            results_reset = results.reset_index()
            # Rename columns for display
            col_map = {
                'Cell Type':        'cell_type',
                'Effect':           'effect',
                'Final Parameter':  'final_parameter',
                'Inclusion probability': 'inclusion_prob',
                'Is credible':      'is_credible',
                'Expected sample':  'expected_sample',
                'log2-fold change': 'log2fc',
            }
            results_reset = results_reset.rename(columns={k: v for k, v in col_map.items() if k in results_reset.columns})
            # Sanitize for JSON
            results_reset = results_reset.replace([float('inf'), float('-inf')], [999.0, -999.0])
            results_reset = results_reset.fillna(0)
            # Round numerics
            for c in results_reset.select_dtypes(include=_np.number).columns:
                results_reset[c] = results_reset[c].round(4)

            n_credible = int(results.get('Is credible', pd.Series(dtype=bool)).astype(bool).sum()) if 'Is credible' in results.columns else 0

            return jsonify({
                'method':     method,
                'n_total':    len(results),
                'n_credible': n_credible,
                'all_results': results_reset.to_dict(orient='records'),
            })

        elif method == 'milopy':
            # Validate milopy-specific params
            if not sample_key:
                return jsonify({'error': 'milopy 方法需要提供样本列 (sample_key)，请在参数面板中选择一个 obs 列'}), 400
            if sample_key not in adata_sub.obs.columns:
                return jsonify({'error': f'样本列 "{sample_key}" 不存在于 obs，请检查列名是否正确'}), 400
            if not use_rep or use_rep not in adata_sub.obsm:
                # fallback
                if 'X_pca' in adata_sub.obsm:
                    use_rep = 'X_pca'
                else:
                    return jsonify({'error': f'嵌入空间 "{use_rep}" 不存在于 obsm，请先运行 PCA'}), 400

            from omicverse.single._milo_dev import Milo
            milo = Milo()
            mdata = milo.load(adata_sub)
            sc.pp.neighbors(mdata['rna'], use_rep=use_rep, n_neighbors=n_neighbors)
            milo.make_nhoods(mdata['rna'], prop=prop)
            mdata = milo.count_nhoods(mdata, sample_col=sample_key)
            milo.da_nhoods(
                mdata,
                design=f'~{condition}',
                model_contrasts=f'{condition}[{test_group}]-{condition}[{ctrl_group}]',
                solver='edger',
            )
            # Build nhood graph (for visualization) - use use_rep as basis
            milo.build_nhood_graph(mdata, basis=use_rep)
            milo.annotate_nhoods(mdata, anno_col=cell_type_key)

            state.dct_model   = milo
            state.dct_data    = mdata
            state.dct_results = mdata['milo'].var.copy()

            results = mdata['milo'].var.copy()

            # Annotate with mix_threshold
            results['nhood_annotation'] = results['nhood_annotation'].astype(str)
            results.loc[results['nhood_annotation_frac'] < 0.6, 'nhood_annotation'] = 'Mixed'

            # Build results for client
            keep = ['nhood_annotation', 'nhood_annotation_frac', 'logFC', 'PValue', 'SpatialFDR', 'Nhood_size']
            keep = [c for c in keep if c in results.columns]
            res_sub = results[keep].copy()
            res_sub = res_sub.replace([float('inf'), float('-inf')], [999.0, -999.0]).fillna(0)
            for c in res_sub.select_dtypes(include=_np.number).columns:
                res_sub[c] = res_sub[c].round(4)
            res_sub = res_sub.reset_index().rename(columns={'index': 'nhood_id'})

            # Aggregate by cell type
            sig_mask = results['SpatialFDR'] < 0.1 if 'SpatialFDR' in results.columns else pd.Series(False, index=results.index)
            n_sig = int(sig_mask.sum())
            n_up  = int((sig_mask & (results.get('logFC', pd.Series(0, index=results.index)) > 0)).sum())
            n_down= int((sig_mask & (results.get('logFC', pd.Series(0, index=results.index)) < 0)).sum())

            return jsonify({
                'method':      method,
                'n_total':     len(results),
                'n_sig':       n_sig,
                'n_sig_up':    n_up,
                'n_sig_down':  n_down,
                'all_results': res_sub.to_dict(orient='records'),
            })
        else:
            return jsonify({'error': f'不支持的方法: {method}'}), 400

    except Exception as e:
        logging.error(f"dct_analyze failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/dct/plot_composition', methods=['POST'])
def dct_plot_composition():
    """Generate a stacked bar chart of cell type proportions per sample/condition."""
    if state.dct_adata is None:
        return jsonify({'error': '请先运行 DCT 分析'}), 400
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as _np
        import base64

        adata      = state.dct_adata
        condition  = state.dct_condition
        ct_key     = state.dct_cell_type_key
        sample_key = state.dct_sample_key
        method     = state.dct_method

        import pandas as pd

        # Compute proportions per sample (or per condition if no sample_key)
        group_by = sample_key if sample_key and sample_key in adata.obs.columns else condition

        ct_counts = pd.crosstab(adata.obs[group_by], adata.obs[ct_key])
        ct_prop   = ct_counts.div(ct_counts.sum(axis=1), axis=0)

        # Order by condition label
        if sample_key and sample_key in adata.obs.columns and condition in adata.obs.columns:
            sample_cond = adata.obs[[sample_key, condition]].drop_duplicates().set_index(sample_key)
            ct_prop['__cond'] = [sample_cond.loc[s, condition] if s in sample_cond.index else '' for s in ct_prop.index]
            ct_prop = ct_prop.sort_values('__cond').drop(columns='__cond')

        n_ct = ct_prop.shape[1]
        colors = plt.cm.tab20(_np.linspace(0, 1, n_ct)) if n_ct <= 20 else plt.cm.hsv(_np.linspace(0, 1, n_ct))

        fig, ax = plt.subplots(figsize=(max(6, len(ct_prop) * 0.7), 4.5))
        bottom = _np.zeros(len(ct_prop))
        for i, ct in enumerate(ct_prop.columns):
            vals = ct_prop[ct].values
            ax.bar(range(len(ct_prop)), vals, bottom=bottom,
                   label=ct, color=colors[i], width=0.8, linewidth=0)
            bottom += vals

        ax.set_xticks(range(len(ct_prop)))
        ax.set_xticklabels(ct_prop.index, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Proportion', fontsize=10)
        ax.set_title(f'Cell Type Composition by {group_by}', fontsize=11)
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7,
                  framealpha=0.5, ncol=max(1, n_ct // 20))
        ax.set_ylim(0, 1)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
        plt.close(fig)
        return jsonify({'figure': base64.b64encode(buf.getvalue()).decode('ascii')})

    except Exception as e:
        logging.error(f"dct_plot_composition failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/dct/plot_effects', methods=['POST'])
def dct_plot_effects():
    """Generate effects / beeswarm plot for DCT results."""
    if state.dct_results is None:
        return jsonify({'error': '请先运行 DCT 分析'}), 400
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as _np
        import base64

        method = state.dct_method

        if method == 'sccoda':
            results = state.dct_results
            if results is None or len(results) == 0:
                return jsonify({'error': 'scCODA 分析结果为空'}), 400

            # Bar plot of effect sizes (log2-fold change or Final Parameter)
            fc_col = 'log2-fold change' if 'log2-fold change' in results.columns else \
                     'Final Parameter' if 'Final Parameter' in results.columns else None
            if fc_col is None:
                return jsonify({'error': '结果中无法找到效应量列'}), 400

            credible_col = 'Is credible' if 'Is credible' in results.columns else None
            results = results.copy().dropna(subset=[fc_col])
            results = results.sort_values(fc_col)

            is_credible = results[credible_col].astype(bool) if credible_col else \
                          _np.zeros(len(results), dtype=bool)

            colors = ['#e06c75' if c and v > 0 else '#5ba4cf' if c and v < 0 else '#aaaaaa'
                      for c, v in zip(is_credible, results[fc_col])]

            fig, ax = plt.subplots(figsize=(6, max(3, len(results) * 0.35 + 1)))
            y_pos = range(len(results))
            # Get cell type names
            ct_names = results.index.tolist() if results.index.name == 'Cell Type' else \
                       results['Cell Type'].tolist() if 'Cell Type' in results.columns else \
                       results.index.tolist()

            ax.barh(y_pos, results[fc_col].values, color=colors, height=0.6, linewidth=0)
            ax.axvline(0, color='#555', lw=1, ls='--')
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(ct_names, fontsize=9)
            ax.set_xlabel('log₂-fold change' if fc_col == 'log2-fold change' else 'Effect', fontsize=10)
            ax.set_title('scCODA Effects (credible cell types highlighted)', fontsize=10)
            for spine in ('top', 'right'):
                ax.spines[spine].set_visible(False)
            plt.tight_layout()

        elif method == 'milopy':
            mdata  = state.dct_data
            milo   = state.dct_model
            ct_key = state.dct_cell_type_key

            # Use built-in beeswarm plot
            try:
                fig = milo.plot_da_beeswarm(mdata, return_fig=True)
                if fig is None:
                    raise RuntimeError('plot_da_beeswarm returned None')
            except Exception:
                # Fallback: manual beeswarm from stored results
                results = state.dct_results
                if 'nhood_annotation' not in results.columns or 'logFC' not in results.columns:
                    return jsonify({'error': '结果缺少必要的列，请检查分析是否成功'}), 400

                import seaborn as sns
                fig, ax = plt.subplots(figsize=(6, max(3.5, len(results['nhood_annotation'].unique()) * 0.4 + 1)))
                results_plot = results.copy()
                results_plot['nhood_annotation'] = results_plot['nhood_annotation'].astype(str)
                alpha_col = 'SpatialFDR' if 'SpatialFDR' in results_plot.columns else None
                results_plot['_sig'] = (results_plot[alpha_col] < 0.1) if alpha_col else False

                sorted_annos = (results_plot[['nhood_annotation', 'logFC']]
                                .groupby('nhood_annotation').median()
                                .sort_values('logFC', ascending=True).index.tolist())

                sns.violinplot(data=results_plot, y='nhood_annotation', x='logFC',
                               order=sorted_annos, inner=None, orient='h',
                               linewidth=0, scale='width', ax=ax)
                sns.stripplot(data=results_plot, y='nhood_annotation', x='logFC',
                              order=sorted_annos, size=2,
                              hue='_sig', palette=['grey', 'black'],
                              orient='h', alpha=0.5, ax=ax)
                ax.axvline(0, color='black', ls='--', lw=1)
                ax.set_xlabel('log Fold Change', fontsize=10)
                ax.set_title('Milopy DA Beeswarm', fontsize=11)
                for spine in ('top', 'right'):
                    ax.spines[spine].set_visible(False)
                plt.tight_layout()
        else:
            return jsonify({'error': f'不支持的方法: {method}'}), 400

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
        plt.close(fig)
        return jsonify({'figure': base64.b64encode(buf.getvalue()).decode('ascii')})

    except Exception as e:
        logging.error(f"dct_plot_effects failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Annotation Helper Endpoints
# ============================================================================

@app.route('/api/annotation/celltypist_models', methods=['GET'])
def get_celltypist_models():
    """Fetch the CellTypist model registry and return as JSON list."""
    try:
        from omicverse.single._annotation import _celltypist_models_description
        df = _celltypist_models_description()
        keep = ['model', 'description', 'No_celltypes', 'source', 'date', 'default']
        cols = [c for c in keep if c in df.columns]
        records = df[cols].where(df[cols].notna(), None).to_dict(orient='records')
        return jsonify({'models': records})
    except Exception as e:
        logging.error(f"CellTypist model list failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/annotation/download_celltypist_model', methods=['POST'])
def download_celltypist_model():
    """Download a CellTypist model pkl file and return its absolute path."""
    try:
        payload = request.json or {}
        model_name = payload.get('model_name', '').strip()
        if not model_name:
            return jsonify({'error': 'model_name 不能为空'}), 400

        from omicverse.single._annotation import _celltypist_models_description
        from omicverse.datasets import download_data

        df = _celltypist_models_description()
        row = df[df['model'] == model_name]
        if row.empty:
            return jsonify({'error': f'模型 {model_name} 未在 CellTypist 注册表中找到'}), 404

        url = row.iloc[0].get('url')
        if not url:
            return jsonify({'error': f'模型 {model_name} 缺少下载 URL'}), 500

        save_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(save_dir, exist_ok=True)
        downloaded = download_data(url, file_path=model_name, dir=save_dir)
        abs_path = os.path.abspath(downloaded)
        return jsonify({'success': True, 'path': abs_path})
    except Exception as e:
        logging.error(f"CellTypist model download failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/annotation/download_scsa_db', methods=['POST'])
def download_scsa_db():
    """Download the SCSA marker database and return its absolute path."""
    try:
        from omicverse.datasets import download_data

        save_dir = os.path.join(os.getcwd(), 'temp')
        os.makedirs(save_dir, exist_ok=True)
        db_name = 'pySCSA_2024_v1_plus.db'
        mirrors = [
            'https://stacks.stanford.edu/file/cv694yk7414/pySCSA_2023_v2_plus.db',
            'https://figshare.com/ndownloader/files/41369037',
        ]
        abs_path = None
        for url in mirrors:
            try:
                downloaded = download_data(url, file_path=db_name, dir=save_dir)
                abs_path = os.path.abspath(downloaded)
                break
            except Exception as dl_err:
                logging.warning(f"SCSA DB download from {url} failed: {dl_err}")

        if abs_path is None:
            return jsonify({'error': 'SCSA 数据库下载失败，请检查网络连接'}), 500
        return jsonify({'success': True, 'path': abs_path})
    except Exception as e:
        logging.error(f"SCSA DB download failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/annotation/celltypist_model_path', methods=['GET'])
def get_celltypist_model_path():
    """Return the local path for a CellTypist model if it has already been downloaded."""
    model_name = request.args.get('model_name', '').strip()
    if not model_name:
        return jsonify({'error': 'model_name is required'}), 400
    save_dir = os.path.join(os.getcwd(), 'models')
    # Try common extensions that download_data might produce
    for ext in ('', '.pkl', '.gz'):
        candidate = os.path.join(save_dir, model_name + ext)
        if os.path.exists(candidate):
            return jsonify({'exists': True, 'path': os.path.abspath(candidate)})
    return jsonify({'exists': False})


# ============================================================================
# Python Environment Management Endpoints
# ============================================================================

_PYPI_MIRRORS = [
    {'name': 'PyPI (official)',   'url': 'https://pypi.org/simple'},
    {'name': '清华 Tsinghua',     'url': 'https://pypi.tuna.tsinghua.edu.cn/simple'},
    {'name': '阿里 Aliyun',      'url': 'https://mirrors.aliyun.com/pypi/simple'},
    {'name': '中科大 USTC',      'url': 'https://pypi.mirrors.ustc.edu.cn/simple'},
]


@app.route('/api/env/info', methods=['GET'])
def env_info():
    """Return current Python environment information."""
    import sys, platform, importlib.metadata as _meta, shutil, subprocess

    # ── System ───────────────────────────────────────────────────────────────
    system = {
        'os':       platform.system(),
        'os_ver':   platform.version(),
        'machine':  platform.machine(),
        'hostname': platform.node(),
    }
    try:
        import psutil
        mem = psutil.virtual_memory()
        system['ram_total_gb'] = round(mem.total / 1024**3, 1)
        system['ram_used_gb']  = round(mem.used  / 1024**3, 1)
        system['cpu_count']    = psutil.cpu_count(logical=True)
    except ImportError:
        pass

    # ── Python ───────────────────────────────────────────────────────────────
    python_info = {
        'version':    sys.version,
        'executable': sys.executable,
        'prefix':     sys.prefix,
    }
    # pip version from metadata — zero subprocess overhead
    try:
        python_info['pip'] = _meta.version('pip')
    except _meta.PackageNotFoundError:
        pass

    # uv version via subprocess (uv starts in <100ms, no heavy runtime)
    uv_path = shutil.which('uv')
    if uv_path:
        try:
            python_info['uv'] = subprocess.check_output(
                [uv_path, '--version'], stderr=subprocess.STDOUT, timeout=2, text=True
            ).strip().split('\n')[0]
        except Exception:
            python_info['uv'] = uv_path

    # ── GPU ──────────────────────────────────────────────────────────────────
    # Get torch version from metadata (no import needed — fast)
    gpu_info = {'cuda_available': False, 'mps_available': False, 'devices': []}
    try:
        gpu_info['torch_version'] = _meta.version('torch')
    except _meta.PackageNotFoundError:
        pass
    # Only query CUDA/MPS if torch is already loaded in this process (no cold import)
    import sys as _sys
    if 'torch' in _sys.modules:
        torch = _sys.modules['torch']
        try:
            if torch.cuda.is_available():
                gpu_info['cuda_available'] = True
                gpu_info['cuda_version'] = torch.version.cuda
                gpu_info['devices'] = [
                    {'index': i, 'name': torch.cuda.get_device_name(i),
                     'mem_gb': round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 1)}
                    for i in range(torch.cuda.device_count())
                ]
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpu_info['mps_available'] = True
                gpu_info['devices'] = [{'index': 0, 'name': 'Apple MPS', 'mem_gb': None}]
        except Exception:
            pass

    # ── Key packages ─────────────────────────────────────────────────────────
    _KEY_PKGS = [
        'omicverse', 'scanpy', 'anndata', 'numpy', 'pandas', 'scipy',
        'matplotlib', 'torch', 'torchvision', 'scvi-tools', 'harmonypy',
        'leidenalg', 'umap-learn', 'pynndescent', 'sklearn', 'scikit-learn',
        'cellrank', 'scvelo', 'squidpy', 'pertpy',
    ]
    key_pkgs = []
    for name in _KEY_PKGS:
        try:
            ver = _meta.version(name)
            key_pkgs.append({'name': name, 'version': ver, 'installed': True})
        except _meta.PackageNotFoundError:
            key_pkgs.append({'name': name, 'version': None, 'installed': False})

    # ── All installed packages ────────────────────────────────────────────────
    all_pkgs = sorted(
        [{'name': d.metadata['Name'], 'version': d.version}
         for d in _meta.distributions()
         if d.metadata.get('Name')],
        key=lambda x: x['name'].lower()
    )

    return jsonify({
        'system':   system,
        'python':   python_info,
        'gpu':      gpu_info,
        'key_pkgs': key_pkgs,
        'all_pkgs': all_pkgs,
    })


@app.route('/api/env/search_pypi', methods=['GET'])
def env_search_pypi():
    """Look up a package on PyPI and return its metadata."""
    import urllib.request, urllib.error
    name = request.args.get('package', '').strip()
    if not name:
        return jsonify({'error': 'package name required'}), 400
    try:
        url = f'https://pypi.org/pypi/{name}/json'
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read())
        info = data.get('info', {})
        return jsonify({
            'found': True,
            'name': info.get('name', name),
            'version': info.get('version', ''),
            'summary': info.get('summary', ''),
            'home_page': info.get('home_page') or info.get('project_url', ''),
            'license': info.get('license', ''),
            'requires_python': info.get('requires_python', ''),
        })
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return jsonify({'found': False, 'name': name})
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 502


@app.route('/api/env/test_mirrors', methods=['GET'])
def env_test_mirrors():
    """Measure latency to each PyPI mirror and return sorted results."""
    import urllib.request, urllib.error, time as _time
    results = []
    for m in _PYPI_MIRRORS:
        t0 = _time.time()
        try:
            req = urllib.request.Request(m['url'], method='HEAD')
            with urllib.request.urlopen(req, timeout=5):
                pass
            latency = round((_time.time() - t0) * 1000)
            results.append({'name': m['name'], 'url': m['url'],
                            'latency_ms': latency, 'ok': True})
        except Exception:
            results.append({'name': m['name'], 'url': m['url'],
                            'latency_ms': 9999, 'ok': False})
    results.sort(key=lambda x: x['latency_ms'])
    return jsonify({'mirrors': results})


@app.route('/api/env/install_pip', methods=['POST'])
def env_install_pip():
    """Stream uv pip install output via SSE."""
    import subprocess, shutil
    payload = request.json or {}
    package  = payload.get('package', '').strip()
    mirror   = payload.get('mirror', '').strip()
    extra    = payload.get('extra_args', '').strip()
    if not package:
        return jsonify({'error': 'package name required'}), 400

    uv_path = shutil.which('uv') or 'uv'
    cmd = [uv_path, 'pip', 'install', package]
    if mirror:
        cmd += ['--index-url', mirror]
    if extra:
        cmd += extra.split()

    def generate():
        yield f"data: {json.dumps({'type': 'cmd', 'text': ' '.join(cmd)})}\n\n"
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
            for line in iter(proc.stdout.readline, ''):
                yield f"data: {json.dumps({'type': 'output', 'text': line})}\n\n"
            proc.wait()
            ok = proc.returncode == 0
            yield f"data: {json.dumps({'type': 'complete', 'success': ok, 'returncode': proc.returncode})}\n\n"
        except FileNotFoundError:
            yield f"data: {json.dumps({'type': 'error', 'text': 'uv not found. Run: pip install uv'})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/api/env/install_conda', methods=['POST'])
def env_install_conda():
    """Stream mamba install output via SSE."""
    import subprocess, shutil
    payload  = request.json or {}
    package  = payload.get('package', '').strip()
    channels = payload.get('channels', ['conda-forge', 'bioconda'])
    extra    = payload.get('extra_args', '').strip()
    if not package:
        return jsonify({'error': 'package name required'}), 400

    mamba_path = shutil.which('mamba') or shutil.which('conda') or 'mamba'
    cmd = [mamba_path, 'install', '-y']
    for ch in channels:
        cmd += ['-c', ch]
    cmd.append(package)
    if extra:
        cmd += extra.split()

    def generate():
        yield f"data: {json.dumps({'type': 'cmd', 'text': ' '.join(cmd)})}\n\n"
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
            for line in iter(proc.stdout.readline, ''):
                yield f"data: {json.dumps({'type': 'output', 'text': line})}\n\n"
            proc.wait()
            ok = proc.returncode == 0
            yield f"data: {json.dumps({'type': 'complete', 'success': ok, 'returncode': proc.returncode})}\n\n"
        except FileNotFoundError:
            yield f"data: {json.dumps({'type': 'error', 'text': 'mamba/conda not found. Install mamba: conda install -c conda-forge mamba'})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


def _smooth_values_by_density(x_vals, y_vals, values, adjust, grid_size=220):
    """Fast KDE-like smoothing on a 2D grid for embedding coloring."""
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        return np.asarray(values, dtype=np.float64), False

    try:
        adj = float(adjust)
    except Exception:
        return np.asarray(values, dtype=np.float64), False

    if not np.isfinite(adj):
        return np.asarray(values, dtype=np.float64), False

    adj = float(np.clip(adj, 0.2, 4.0))
    x = np.asarray(x_vals, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if x.size < 8 or y.size != x.size or v.size != x.size:
        return v, False

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    if finite.sum() < 8:
        return v, False

    x = x[finite]
    y = y[finite]
    v = v[finite]

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)

    n = x.size
    g = int(np.clip(np.sqrt(n) * 1.4, 80, grid_size))
    sigma = 0.35 + 2.4 * adj

    xi = np.clip(((x - x_min) / x_span * (g - 1)).astype(np.int32), 0, g - 1)
    yi = np.clip(((y - y_min) / y_span * (g - 1)).astype(np.int32), 0, g - 1)

    val_grid = np.zeros((g, g), dtype=np.float64)
    cnt_grid = np.zeros((g, g), dtype=np.float64)
    np.add.at(val_grid, (yi, xi), v)
    np.add.at(cnt_grid, (yi, xi), 1.0)

    val_s = gaussian_filter(val_grid, sigma=sigma, mode='nearest')
    cnt_s = gaussian_filter(cnt_grid, sigma=sigma, mode='nearest')
    with np.errstate(divide='ignore', invalid='ignore'):
        sm_grid = np.where(cnt_s > 1e-10, val_s / cnt_s, np.nan)
    smoothed = sm_grid[yi, xi]
    smoothed = np.where(np.isfinite(smoothed), smoothed, v)

    out = np.asarray(values, dtype=np.float64).copy()
    out_idx = np.where(finite)[0]
    out[out_idx] = smoothed
    return out, True


def _parse_density_adjust(raw, default=1.0):
    try:
        val = float(raw)
    except Exception:
        val = default
    if not np.isfinite(val):
        val = default
    return float(np.clip(val, 0.2, 4.0))


@app.route('/api/plot', methods=['POST'])
def plot_data_legacy():
    """Legacy plot endpoint - returns JSON data for Plotly."""
    if state.current_adaptor is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        from server.common.fbs.matrix import decode_matrix_fbs

        data_req = request.json
        embedding = data_req.get('embedding', '')
        x_axis_str = data_req.get('x_axis', '')
        y_axis_str = data_req.get('y_axis', '')
        color_by = data_req.get('color_by', '')
        palette = data_req.get('palette', None)
        category_palette = data_req.get('category_palette', None)
        vmin = data_req.get('vmin', None)
        vmax = data_req.get('vmax', None)
        density_adjust = _parse_density_adjust(data_req.get('density_adjust', 1.0))
        density_active = bool(data_req.get('density_active', False))

        if x_axis_str and y_axis_str:
            # ── Custom axes path ─────────────────────────────────────────────
            adata = state.current_adata
            x_raw_np, x_label = _resolve_axis(adata, x_axis_str)
            y_raw_np, y_label = _resolve_axis(adata, y_axis_str)
            import numpy as _np2
            x_raw = _np2.asarray(x_raw_np, dtype=float).tolist()
            y_raw = _np2.asarray(y_raw_np, dtype=float).tolist()
            axis_labels = {'x': x_label, 'y': y_label}
        elif embedding == 'random':
            # ── Random layout ────────────────────────────────────────────────
            import numpy as _np2
            n_cells_r = state.current_adata.n_obs
            _rng = _np2.random.default_rng(42)
            x_raw = _rng.standard_normal(n_cells_r).tolist()
            y_raw = _rng.standard_normal(n_cells_r).tolist()
            axis_labels = {'x': 'Random 1', 'y': 'Random 2'}
        elif embedding:
            # ── Default embedding path ───────────────────────────────────────
            fbs_data = state.current_adaptor.get_embedding_fbs(embedding)
            coords_df = decode_matrix_fbs(fbs_data)
            x_raw = coords_df['x'].tolist()
            y_raw = coords_df['y'].tolist()
            axis_labels = None
        else:
            return jsonify({'error': 'No embedding or axis specified'}), 400

        n_cells = len(x_raw)

        # ── For large datasets apply spatial decimation ────────────────────────
        # Keep at most 150 K representative points so Plotly stays responsive.
        # The kept_indices list is used below to sub-select color/hover arrays.
        kept_indices = None
        if n_cells > _LARGE_DATASET_THRESHOLD:
            x_dec, y_dec, _, kept_indices = _spatial_decimate(
                x_raw, y_raw, None, target_n=150_000)
            x_raw, y_raw = x_dec, y_dec
            decimated = True
        else:
            decimated = False

        plot_data = {
            'x': x_raw,
            'y': y_raw,
            'hover_text': [f'Cell {i}' for i in (kept_indices if kept_indices else range(n_cells))],
            'decimated': decimated,
            'n_total': n_cells,
            'n_shown': len(x_raw),
            'density_enabled': False,
            'density_applied': False,
            'density_adjust': density_adjust,
            'density_message': 'Density adjust is off',
        }
        if axis_labels:
            plot_data['axis_labels'] = axis_labels

        # Helper: sub-select a full-length array by kept_indices (if decimated)
        def _subset(arr):
            if kept_indices is None:
                return arr
            return [arr[i] for i in kept_indices]

        def _subset_np(arr):
            if kept_indices is None:
                return arr
            import numpy as _np2
            return _np2.asarray(arr)[kept_indices]

        # Handle coloring if requested
        if color_by:
            if color_by.startswith('obs:'):
                col_name = color_by.replace('obs:', '')
                obs_fbs = state.current_adaptor.get_obs_fbs([col_name])
                obs_df = decode_matrix_fbs(obs_fbs)

                if col_name in obs_df.columns:
                    values = obs_df[col_name]
                    if pd.api.types.is_numeric_dtype(values):
                        colors_array = _subset_np(values.values)
                        density_applied = False
                        if density_active:
                            colors_array, density_applied = _smooth_values_by_density(
                                np.asarray(x_raw, dtype=np.float64),
                                np.asarray(y_raw, dtype=np.float64),
                                colors_array,
                                density_adjust
                            )
                        plot_data['density_enabled'] = True
                        plot_data['density_applied'] = bool(density_applied)
                        plot_data['density_message'] = '' if density_active else 'Density adjust is off'
                        if vmin is not None or vmax is not None:
                            colors_array = np.clip(colors_array,
                                                   vmin if vmin is not None else colors_array.min(),
                                                   vmax if vmax is not None else colors_array.max())
                        plot_data['colors'] = colors_array.tolist()
                        plot_data['colorscale'] = palette if palette else 'Viridis'
                        plot_data['color_label'] = col_name
                        if vmin is not None:
                            plot_data['cmin'] = vmin
                        if vmax is not None:
                            plot_data['cmax'] = vmax
                    else:
                        plot_data['density_enabled'] = False
                        plot_data['density_applied'] = False
                        plot_data['density_message'] = 'Density adjust is disabled for non-numeric features'
                        # Compute categories on full column, sub-select codes only
                        categories = values.astype('category') if not pd.api.types.is_categorical_dtype(values) else values
                        plot_data['color_label'] = col_name
                        cat_label_list = categories.cat.categories.tolist()
                        plot_data['category_labels'] = cat_label_list
                        codes_sub = _subset_np(categories.cat.codes.values)
                        plot_data['category_codes'] = codes_sub.tolist()
                        n_categories = len(cat_label_list)
                        if category_palette:
                            # User explicitly chose a palette — respect it
                            discrete_colors = get_discrete_colors(n_categories, category_palette)
                        else:
                            # Default: prefer adata.uns colors aligned to current label order
                            discrete_colors = (
                                get_uns_colors_for_labels(state.current_adata, col_name, cat_label_list)
                                or get_discrete_colors(n_categories, None)
                            )
                        plot_data['colors'] = [discrete_colors[code] for code in codes_sub]
                        plot_data['discrete_colors'] = discrete_colors

            elif color_by.startswith('gene:'):
                gene_name = color_by.replace('gene:', '')
                try:
                    expr_fbs = state.current_adaptor.get_expression_fbs([gene_name])
                    expr_df = decode_matrix_fbs(expr_fbs)
                    if gene_name in expr_df.columns:
                        expression = _subset_np(expr_df[gene_name].values)
                        density_applied = False
                        if density_active:
                            expression, density_applied = _smooth_values_by_density(
                                np.asarray(x_raw, dtype=np.float64),
                                np.asarray(y_raw, dtype=np.float64),
                                expression,
                                density_adjust
                            )
                        plot_data['density_enabled'] = True
                        plot_data['density_applied'] = bool(density_applied)
                        plot_data['density_message'] = '' if density_active else 'Density adjust is off'
                        if vmin is not None or vmax is not None:
                            expression = np.clip(expression,
                                               vmin if vmin is not None else expression.min(),
                                               vmax if vmax is not None else expression.max())
                        plot_data['colors'] = expression.tolist()
                        plot_data['colorscale'] = palette if palette else 'Viridis'
                        plot_data['color_label'] = f'{gene_name} expression'
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


# ============================================================================
# Large-scale Raster Plot (Datashader / fallback decimation)
# ============================================================================

# Number of cells above which we switch to raster / decimation mode.
_LARGE_DATASET_THRESHOLD = 200_000


def _resolve_axis(adata, axis_str: str):
    """Resolve an axis descriptor string to a 1-D numpy float array.

    Supported formats
    -----------------
    ``obsm:<key>:<dim>``   – slice of adata.obsm[key][:, dim]
    ``obs:<col>``          – adata.obs[col].values (numeric only)
    ``gene:<name>``        – expression vector for a single gene

    Returns (values_1d_np, label_str).
    Raises ValueError with a user-friendly message on failure.
    """
    import numpy as _np
    import scipy.sparse as _sp

    parts = axis_str.split(':')
    src = parts[0]

    if src == 'obsm':
        if len(parts) < 3:
            raise ValueError(f'Invalid obsm axis spec: "{axis_str}". Expected obsm:<key>:<dim>')
        key = ':'.join(parts[1:-1])
        dim = int(parts[-1])
        key = _resolve_embedding_key(adata, key) or key
        if key not in adata.obsm:
            raise ValueError(f'obsm key "{key}" not found in adata.obsm')
        arr = adata.obsm[key]
        if hasattr(arr, 'toarray'):
            arr = arr.toarray()
        arr = _np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            return arr, f'{key}'
        if dim >= arr.shape[1]:
            raise ValueError(f'Dimension {dim} out of range for obsm "{key}" (shape {arr.shape})')
        label = (key[2:] if key.startswith('X_') else key).upper() + f'_{dim + 1}'
        return arr[:, dim], label

    elif src == 'obs':
        if len(parts) < 2:
            raise ValueError(f'Invalid obs axis spec: "{axis_str}". Expected obs:<col>')
        col = ':'.join(parts[1:])
        if col not in adata.obs.columns:
            raise ValueError(f'obs column "{col}" not found')
        vals = adata.obs[col]
        try:
            return _np.asarray(vals, dtype=float), col
        except (ValueError, TypeError):
            raise ValueError(f'obs column "{col}" is not numeric and cannot be used as an axis')

    elif src == 'gene':
        if len(parts) < 2:
            raise ValueError(f'Invalid gene axis spec: "{axis_str}". Expected gene:<name>')
        gene = ':'.join(parts[1:])
        if gene not in adata.var_names:
            raise ValueError(f'Gene "{gene}" not found in adata.var_names')
        gi = list(adata.var_names).index(gene)
        X = adata.X
        if _sp.issparse(X):
            vals = _np.asarray(X[:, gi].todense()).flatten()
        else:
            vals = _np.asarray(X[:, gi], dtype=float).flatten()
        return vals, gene

    else:
        raise ValueError(f'Unknown axis source "{src}". Use obsm, obs, or gene.')


def _spatial_decimate(x, y, colors, target_n=150_000):
    """Grid-based spatial decimation that preserves cluster structure.

    Divides the 2-D embedding into a grid and keeps at most one point per
    cell so that visual density is represented uniformly. Unlike random
    subsampling this retains rare cell types that occupy sparse regions.
    """
    n = len(x)
    if n <= target_n:
        return x, y, colors, list(range(n))

    import numpy as _np
    x = _np.asarray(x, dtype=_np.float32)
    y = _np.asarray(y, dtype=_np.float32)

    # Desired grid resolution: sqrt of target_n gives a square grid.
    grid_side = int(_np.sqrt(target_n))
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    eps = 1e-6
    xi = ((x - x_min) / (x_max - x_min + eps) * (grid_side - 1)).astype(_np.int32)
    yi = ((y - y_min) / (y_max - y_min + eps) * (grid_side - 1)).astype(_np.int32)
    cell_key = xi * grid_side + yi

    # Shuffle first so grid occupation is random-fair, then keep first hit per cell.
    rng = _np.random.default_rng(42)
    order = rng.permutation(n)
    seen = {}
    kept = []
    for idx in order:
        k = int(cell_key[idx])
        if k not in seen:
            seen[k] = True
            kept.append(int(idx))

    kept = sorted(kept)
    colors_out = [colors[i] for i in kept] if colors is not None else None
    return x[kept].tolist(), y[kept].tolist(), colors_out, kept


@app.route('/api/plot_raster', methods=['POST'])
def plot_raster():
    """Server-side rasterized scatter using Datashader (falls back to PNG via matplotlib).

    Request body (JSON):
        embedding   – obsm key  (e.g. 'X_umap')
        color_by    – 'obs:<col>' | 'gene:<name>' | ''
        width       – canvas pixel width  (default 800)
        height      – canvas pixel height (default 600)
        x_range     – [xmin, xmax]  (optional, for zoom)
        y_range     – [ymin, ymax]  (optional, for zoom)
        palette     – colormap name (default 'Viridis')
        category_palette – categorical colormap name
        spread      – datashader pixel spread radius (default 1)

    Response JSON:
        image       – base64-encoded PNG
        x_range     – [xmin, xmax] of rendered region
        y_range     – [ymin, ymax] of rendered region
        n_total     – total cell count
        n_rendered  – cells in the current viewport (if x/y_range given)
        legend      – [{label, color}, …]  (categorical only)
        engine      – 'datashader' | 'matplotlib'
    """
    if state.current_adaptor is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        from server.common.fbs.matrix import decode_matrix_fbs
        from utils.adata_helpers import resolve_embedding_key as _resolve_key
        import base64, io as _io
        import numpy as _np

        req        = request.json or {}
        embedding  = req.get('embedding', '')
        color_by   = req.get('color_by', '')
        width      = int(req.get('width',  800))
        height     = int(req.get('height', 600))
        x_range    = req.get('x_range')   # [xmin, xmax] or None
        y_range    = req.get('y_range')   # [ymin, ymax] or None
        palette    = req.get('palette', 'Viridis')
        cat_pal    = req.get('category_palette', None)
        spread     = int(req.get('spread', 1))

        if not embedding:
            return jsonify({'error': 'embedding required'}), 400

        # ── coordinates ───────────────────────────────────────────────────────
        fbs_data  = state.current_adaptor.get_embedding_fbs(embedding)
        coords_df = decode_matrix_fbs(fbs_data)
        x_all = _np.asarray(coords_df['x'], dtype=_np.float64)
        y_all = _np.asarray(coords_df['y'], dtype=_np.float64)
        n_total = len(x_all)

        # Determine viewport bounds
        xmin = float(x_range[0]) if x_range else float(x_all.min())
        xmax = float(x_range[1]) if x_range else float(x_all.max())
        ymin = float(y_range[0]) if y_range else float(y_all.min())
        ymax = float(y_range[1]) if y_range else float(y_all.max())

        # ── colour values ─────────────────────────────────────────────────────
        color_values = None     # numpy array or None
        color_labels = None     # list[str] for categories
        discrete_colors = None  # list[str] hex colours  (len == n_categories)
        is_categorical = False
        color_label = ''

        if color_by.startswith('obs:'):
            col_name = color_by[4:]
            obs_fbs  = state.current_adaptor.get_obs_fbs([col_name])
            obs_df   = decode_matrix_fbs(obs_fbs)
            if col_name in obs_df.columns:
                vals = obs_df[col_name]
                color_label = col_name
                if pd.api.types.is_numeric_dtype(vals):
                    color_values = vals.to_numpy(dtype=_np.float64)
                else:
                    cats = vals.astype('category')
                    color_labels  = cats.cat.categories.tolist()
                    color_values  = cats.cat.codes.to_numpy(dtype=_np.int32)
                    is_categorical = True
                    n_cats = len(color_labels)
                    if cat_pal:
                        # User explicitly chose a palette — respect it
                        discrete_colors = get_discrete_colors(n_cats, cat_pal)
                    else:
                        discrete_colors = (
                            get_uns_colors_for_labels(state.current_adata, col_name, color_labels)
                            or get_discrete_colors(n_cats, None)
                        )

        elif color_by.startswith('gene:'):
            gene_name = color_by[5:]
            color_label = f'{gene_name} expression'
            try:
                expr_fbs = state.current_adaptor.get_expression_fbs([gene_name])
                expr_df  = decode_matrix_fbs(expr_fbs)
                if gene_name in expr_df.columns:
                    color_values = expr_df[gene_name].to_numpy(dtype=_np.float64)
            except Exception:
                pass

        # ── try Datashader first ───────────────────────────────────────────────
        engine = 'datashader'
        try:
            import datashader as ds
            import datashader.transfer_functions as tf
            import pandas as _pd_ds

            df_ds = _pd_ds.DataFrame({'x': x_all, 'y': y_all})
            canvas = ds.Canvas(plot_width=width, plot_height=height,
                               x_range=(xmin, xmax), y_range=(ymin, ymax))

            if is_categorical and color_values is not None:
                df_ds['cat'] = color_values.astype(str)
                agg = canvas.points(df_ds, 'x', 'y', ds.count_cat('cat'))
                # Build color_key from our discrete palette
                color_key = {str(i): c for i, c in enumerate(discrete_colors or [])}
                img = tf.shade(agg, color_key=color_key or None, how='eq_hist')
            elif color_values is not None and not is_categorical:
                df_ds['val'] = color_values
                agg = canvas.points(df_ds, 'x', 'y', ds.mean('val'))
                # Map palette name to colorcet / matplotlib
                cmap_name = palette.lower()
                img = tf.shade(agg, cmap=cmap_name, how='eq_hist')
            else:
                agg = canvas.points(df_ds, 'x', 'y', ds.count())
                img = tf.shade(agg, cmap=['#c6dbef', '#08306b'], how='eq_hist')

            if spread > 1:
                img = tf.spread(img, px=spread - 1)

            buf = _io.BytesIO()
            img.to_pil().save(buf, format='PNG')
            img_b64 = base64.b64encode(buf.getvalue()).decode()

        except ImportError:
            # ── Datashader not available — fall back to matplotlib rasterize ──
            engine = 'matplotlib'
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as _plt
            import matplotlib.colors as _mcolors

            dpi = 96
            fig_w = width  / dpi
            fig_h = height / dpi
            fig, ax = _plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')

            # Filter to viewport
            mask = ((x_all >= xmin) & (x_all <= xmax) &
                    (y_all >= ymin) & (y_all <= ymax))
            xv = x_all[mask]
            yv = y_all[mask]
            cv = color_values[mask] if color_values is not None else None

            n_view = int(mask.sum())
            pt = max(0.5, min(3.0, 50_000 / max(n_view, 1)))

            if is_categorical and cv is not None:
                for code, label in enumerate(color_labels or []):
                    cidx = cv == code
                    hex_c = discrete_colors[code] if discrete_colors else None
                    ax.scatter(xv[cidx], yv[cidx], s=pt, c=hex_c,
                               linewidths=0, rasterized=True, label=label)
            elif cv is not None:
                sc = ax.scatter(xv, yv, c=cv, s=pt, cmap=palette,
                                linewidths=0, rasterized=True)
                _plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.scatter(xv, yv, s=pt, c='#3182bd',
                           linewidths=0, rasterized=True)

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            _plt.tight_layout(pad=0.1)

            buf = _io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
            _plt.close(fig)
            img_b64 = base64.b64encode(buf.getvalue()).decode()

        # ── build legend for categories ────────────────────────────────────────
        legend = []
        if is_categorical and color_labels and discrete_colors:
            legend = [{'label': lbl, 'color': discrete_colors[i % len(discrete_colors)]}
                      for i, lbl in enumerate(color_labels)]

        return jsonify({
            'image':      img_b64,
            'x_range':    [xmin, xmax],
            'y_range':    [ymin, ymax],
            'n_total':    n_total,
            'legend':     legend,
            'color_label': color_label,
            'engine':     engine,
        })

    except Exception as e:
        logging.error(f"plot_raster failed: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# GPU (deck.gl) scatter – binary endpoint
# ============================================================================

@app.route('/api/plot_gpu', methods=['POST'])
def plot_gpu():
    """Return scatter data as a compact binary buffer for deck.gl rendering.

    Binary layout (all little-endian):
        [0-3]               : n_cells   (uint32)
        [4-7]               : json_len  (uint32)
        [8 .. 8+json_len)   : JSON metadata (UTF-8)
        [padded to 8 bytes] : float32 positions  x0,y0,x1,y1,… (n*8 bytes)
        [above + n*8]       : uint8  colors  r,g,b,a per cell   (n*4 bytes)
        [above + n*4]       : float32 hover_values per cell      (n*4 bytes)
                              (category code as float, raw value for continuous, NaN if none)

    JSON metadata fields:
        n_total, color_label, is_categorical,
        category_labels (list|null), colorscale (str|null)
    """
    if state.current_adaptor is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        import struct as _struct, json as _json
        from server.common.fbs.matrix import decode_matrix_fbs
        import numpy as _np

        req            = request.json or {}
        embedding      = req.get('embedding', '')
        x_axis_str     = req.get('x_axis', '')
        y_axis_str     = req.get('y_axis', '')
        color_by       = req.get('color_by', '')
        palette        = req.get('palette', 'viridis')
        cat_pal        = req.get('category_palette', None)
        vmin_req       = req.get('vmin', None)
        vmax_req       = req.get('vmax', None)
        density_adjust = _parse_density_adjust(req.get('density_adjust', 1.0))
        density_active = bool(req.get('density_active', False))

        # ── positions ─────────────────────────────────────────────────────────
        if x_axis_str and y_axis_str:
            adata = state.current_adata
            x_arr, x_label = _resolve_axis(adata, x_axis_str)
            y_arr, y_label = _resolve_axis(adata, y_axis_str)
            x_all = _np.asarray(x_arr, dtype=_np.float32)
            y_all = _np.asarray(y_arr, dtype=_np.float32)
            axis_labels = {'x': x_label, 'y': y_label}
        elif embedding == 'random':
            _rng2 = _np.random.default_rng(42)
            n_r   = state.current_adata.n_obs
            x_all = _rng2.standard_normal(n_r).astype(_np.float32)
            y_all = _rng2.standard_normal(n_r).astype(_np.float32)
            axis_labels = {'x': 'Random 1', 'y': 'Random 2'}
        elif embedding:
            fbs_data  = state.current_adaptor.get_embedding_fbs(embedding)
            coords_df = decode_matrix_fbs(fbs_data)
            x_all = _np.asarray(coords_df['x'], dtype=_np.float32)
            y_all = _np.asarray(coords_df['y'], dtype=_np.float32)
            axis_labels = None
        else:
            return jsonify({'error': 'embedding or x_axis/y_axis required'}), 400

        n = len(x_all)

        # interleaved x,y float32
        positions = _np.empty(n * 2, dtype=_np.float32)
        positions[0::2] = x_all
        positions[1::2] = y_all

        # ── colours ───────────────────────────────────────────────────────────
        colors_rgba   = _np.full((n, 4), [100, 149, 237, 200], dtype=_np.uint8)
        hover_values  = _np.full(n, _np.nan, dtype=_np.float32)

        meta = {
            'n_total':         n,
            'color_label':     '',
            'is_categorical':  False,
            'category_labels': None,
            'colorscale':      None,
            'density_enabled': False,
            'density_applied': False,
            'density_adjust':  density_adjust,
            'density_message': 'Density adjust is off',
        }
        if axis_labels:
            meta['axis_labels'] = axis_labels

        if color_by.startswith('obs:'):
            col_name = color_by[4:]
            obs_fbs  = state.current_adaptor.get_obs_fbs([col_name])
            obs_df   = decode_matrix_fbs(obs_fbs)
            if col_name in obs_df.columns:
                vals = obs_df[col_name]
                meta['color_label'] = col_name

                if pd.api.types.is_numeric_dtype(vals):
                    v    = vals.to_numpy(dtype=_np.float64)
                    density_applied = False
                    if density_active:
                        v, density_applied = _smooth_values_by_density(x_all, y_all, v, density_adjust)
                    vlo  = float(vmin_req) if vmin_req is not None else float(v.min())
                    vhi  = float(vmax_req) if vmax_req is not None else float(v.max())
                    span = max(vhi - vlo, 1e-10)
                    t    = _np.clip((v - vlo) / span, 0.0, 1.0)
                    from matplotlib import colormaps as _cmaps
                    cmap = _cmaps.get_cmap(palette.lower() if palette else 'viridis')
                    rgba_f      = cmap(t)                           # (n,4) float [0,1]
                    colors_rgba = (_np.clip(rgba_f, 0, 1) * 255).astype(_np.uint8)
                    hover_values = v.astype(_np.float32)
                    meta['colorscale'] = palette or 'viridis'
                    meta['vmin'] = round(vlo, 6)
                    meta['vmax'] = round(vhi, 6)
                    meta['density_enabled'] = True
                    meta['density_applied'] = bool(density_applied)
                    meta['density_message'] = '' if density_active else 'Density adjust is off'

                else:
                    cats   = vals.astype('category')
                    labels = cats.cat.categories.tolist()
                    codes  = cats.cat.codes.to_numpy(dtype=_np.int32)
                    n_cats = len(labels)
                    if cat_pal:
                        disc = get_discrete_colors(n_cats, cat_pal)
                    else:
                        disc = (
                            get_uns_colors_for_labels(state.current_adata, col_name, labels)
                            or get_discrete_colors(n_cats, None)
                        )
                    cat_hex_list = []
                    for code, hex_c in enumerate(disc):
                        mask = codes == code
                        try:
                            r = int(hex_c[1:3], 16)
                            g = int(hex_c[3:5], 16)
                            b = int(hex_c[5:7], 16)
                        except Exception:
                            r, g, b = 100, 149, 237
                            hex_c = '#6495ed'
                        if mask.any():
                            colors_rgba[mask] = [r, g, b, 200]
                        cat_hex_list.append(hex_c)
                    hover_values = codes.astype(_np.float32)
                    meta['is_categorical']    = True
                    meta['category_labels']   = labels
                    meta['category_colors']   = cat_hex_list   # exact hex per category
                    meta['density_enabled'] = False
                    meta['density_applied'] = False
                    meta['density_message'] = 'Density adjust is disabled for non-numeric features'

        elif color_by.startswith('gene:'):
            gene_name = color_by[5:]
            meta['color_label'] = f'{gene_name} expression'
            try:
                expr_fbs = state.current_adaptor.get_expression_fbs([gene_name])
                expr_df  = decode_matrix_fbs(expr_fbs)
                if gene_name in expr_df.columns:
                    v    = expr_df[gene_name].to_numpy(dtype=_np.float64)
                    density_applied = False
                    if density_active:
                        v, density_applied = _smooth_values_by_density(x_all, y_all, v, density_adjust)
                    vlo  = float(vmin_req) if vmin_req is not None else float(v.min())
                    vhi  = float(vmax_req) if vmax_req is not None else float(v.max())
                    span = max(vhi - vlo, 1e-10)
                    t    = _np.clip((v - vlo) / span, 0.0, 1.0)
                    from matplotlib import colormaps as _cmaps
                    cmap = _cmaps.get_cmap(palette.lower() if palette else 'viridis')
                    rgba_f      = cmap(t)
                    colors_rgba = (_np.clip(rgba_f, 0, 1) * 255).astype(_np.uint8)
                    hover_values = v.astype(_np.float32)
                    meta['colorscale'] = palette
                    meta['density_enabled'] = True
                    meta['density_applied'] = bool(density_applied)
                    meta['density_message'] = '' if density_active else 'Density adjust is off'
            except Exception as _ge:
                logging.warning(f'plot_gpu gene expr {gene_name}: {_ge}')

        # ── pack binary ───────────────────────────────────────────────────────
        meta_json   = _json.dumps(meta).encode('utf-8')
        json_len    = len(meta_json)
        total_hdr   = 8 + json_len
        pad         = (8 - (total_hdr % 8)) % 8   # align next section to 8 bytes

        buf = (
            _struct.pack('<II', n, json_len)        # 8 bytes header
            + meta_json                              # json_len bytes
            + b'\x00' * pad                         # alignment padding
            + positions.tobytes()                    # n * 8 bytes  (float32 x,y)
            + colors_rgba.tobytes()                  # n * 4 bytes  (uint8  r,g,b,a)
            + hover_values.tobytes()                 # n * 4 bytes  (float32 values)
        )

        from flask import Response as _Resp
        return _Resp(buf, mimetype='application/octet-stream',
                     headers={'Cache-Control': 'no-cache'})

    except Exception as e:
        logging.error(f'plot_gpu failed: {e}')
        return jsonify({'error': str(e)}), 500


# ============================================================================
# App Config / Remote Mode (Phase 4)
# ============================================================================

@app.route('/api/config', methods=['GET'])
def app_config():
    """Return client-visible application configuration.

    The frontend uses this to adapt behavior (e.g. key storage policy)
    based on whether the server is running in remote mode.
    """
    return jsonify({
        'remote_mode': OV_WEB_REMOTE_MODE,
    })


if OV_WEB_REMOTE_MODE:
    @app.after_request
    def _add_security_headers(response):
        """Add security headers in remote mode to harden the deployment."""
        response.headers.setdefault('X-Content-Type-Options', 'nosniff')
        response.headers.setdefault('X-Frame-Options', 'SAMEORIGIN')
        response.headers.setdefault(
            'Referrer-Policy', 'strict-origin-when-cross-origin'
        )
        return response


# ============================================================================
# Agent Routes
# ============================================================================

@app.route('/api/agent/run', methods=['POST'])
def agent_run():
    """Run OmicVerse Agent on current adata."""
    payload = request.json if request.json else {}
    prompt = (payload.get('message') or '').strip()
    if not prompt:
        return jsonify({'error': '没有提供问题'}), 400

    config = payload.get('config') or {}
    system_prompt = (config.get('systemPrompt') or '').strip()
    if system_prompt:
        prompt = f"{system_prompt}\n\n{prompt}"

    session_id = request.headers.get('X-Agent-Session-Id', make_turn_id())

    try:
        agent = get_agent_instance(config)
        if state.current_adata is None:
            reply = run_agent_chat(agent, prompt, session_id=session_id)
            return jsonify({
                'reply': reply,
                'code': None,
                'data_updated': False,
                'data_info': None
            })

        result = run_agent_stream(agent, prompt, state.current_adata, session_id=session_id)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

    new_adata = result.get('result_adata')
    data_updated = False
    if new_adata is not None:
        state.current_adata = new_adata
        data_updated = True
        sync_adaptor_with_adata()
        try:
            state.kernel_executor.sync_adata(state.current_adata)
        except Exception:
            pass

    data_info = None
    if data_updated:
        data_info = {
            'filename': state.current_filename,
            'n_cells': state.current_adata.n_obs,
            'n_genes': state.current_adata.n_vars,
            'embeddings': [emb.replace('X_', '') for emb in state.current_adata.obsm.keys()],
            'obs_columns': list(state.current_adata.obs.columns),
            'var_columns': list(state.current_adata.var.columns)
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


# ---------------------------------------------------------------------------
# Streaming chatbot endpoint (Phase 1)
# ---------------------------------------------------------------------------

@app.route('/api/agent/chat/stream', methods=['POST'])
def agent_chat_stream():
    """Stream agent events as SSE for a single chatbot turn.

    Accepts the same payload as ``/api/agent/run``::

        { "message": "...", "config": { "model": "...", ... } }

    Returns ``text/event-stream`` with one JSON object per ``data:`` line.
    Event types: status, llm_chunk, tool_call, code, result, error, done,
    usage, heartbeat, stream_end.
    """
    payload = request.json if request.json else {}
    prompt = (payload.get('message') or '').strip()
    if not prompt:
        return jsonify({'error': '没有提供问题'}), 400

    config = payload.get('config') or {}
    system_prompt = (config.get('systemPrompt') or '').strip()
    if system_prompt:
        prompt = f"{system_prompt}\n\n{prompt}"

    session_id = request.headers.get('X-Agent-Session-Id', make_turn_id())

    try:
        agent = get_agent_instance(config)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

    # --- Session-scoped adata (Phase 3) ---
    # Get or create session; use session adata if available, else global.
    session = session_manager.get_or_create(session_id, base_adata=state.current_adata)
    current_adata = session.adata if session.adata is not None else state.current_adata

    # Record user message in session history
    session.add_message("user", prompt)

    def _commit_adata(ctx):
        """Called in the producer thread after the agent finishes.

        Commits adata to both session state and global state.
        Only commits when the turn completed without error.
        """
        if (ctx.get('data_updated')
                and ctx.get('result_adata') is not None
                and not ctx.get('error')):
            # Commit to session (copy-on-write)
            session_manager.commit_session_adata(session_id, ctx['result_adata'])
            # Also update global state for backward compatibility
            state.current_adata = ctx['result_adata']
            sync_adaptor_with_adata()
            try:
                state.kernel_executor.sync_adata(state.current_adata)
            except Exception:
                pass

        if ctx.get('trace_id'):
            session.register_trace(ctx['trace_id'])

        # Record assistant response in session history.
        # Prefer the accumulated LLM text (covers Q&A and analysis turns),
        # fall back to the done-event summary, then to a data-update note.
        reply_text = ctx.get('llm_text', '') or ctx.get('summary', '')
        if not reply_text and ctx.get('data_updated'):
            shape = ctx.get('result_shape')
            reply_text = f"Data updated: {shape[0]}x{shape[1]}" if shape else "Data updated"
        if reply_text:
            session.add_message("assistant", reply_text)

    def _cleanup_turn(ctx):
        """Always runs — clear active turn tracking even on cancel."""
        session.clear_turn()

    # Build conversation history for multi-turn context.
    # Exclude the current user message (just added above) — it's passed as `prompt`.
    prior_history = session.get_history_dicts()[:-1]  # all except last (current)

    handle = stream_agent_events(
        agent, prompt, current_adata,
        session_id=session_id,
        history=prior_history,
        on_complete=_commit_adata,
        on_finally=_cleanup_turn,
    )

    # Register active turn in session for cancel support
    session.register_turn(handle.turn_id, handle.agent_cancel)

    resp = Response(stream_with_context(handle), mimetype='text/event-stream')
    resp.headers['Cache-Control'] = 'no-cache'
    resp.headers['X-Accel-Buffering'] = 'no'
    resp.headers['X-Agent-Turn-Id'] = handle.turn_id
    return resp


@app.route('/api/agent/chat/turn/<turn_id>', methods=['GET'])
def agent_chat_turn_replay(turn_id):
    """Return buffered events for a turn (for reconnection).

    Clients that lose the SSE connection mid-turn can GET this endpoint to
    retrieve all events emitted so far, then reconnect.
    """
    events = get_turn_buffer(turn_id)
    return jsonify({'turn_id': turn_id, 'events': events})


@app.route('/api/agent/harness/initialize', methods=['GET', 'POST'])
def agent_harness_initialize():
    """Return the web/client handshake payload for the harness layer."""
    payload = request.json if request.is_json else {}
    session_id = (payload.get('session_id') or request.headers.get('X-Agent-Session-Id') or '').strip()
    return jsonify(build_harness_initialize_payload(session_id))


@app.route('/api/agent/trace/<trace_id>', methods=['GET'])
def agent_trace_replay(trace_id):
    """Return a persisted harness trace by id."""
    trace = load_trace(trace_id)
    if trace is None:
        return jsonify({'error': 'Trace not found'}), 404
    return jsonify(trace)


# ---------------------------------------------------------------------------
# Session management endpoints (Phase 3)
# ---------------------------------------------------------------------------

@app.route('/api/agent/session', methods=['POST'])
def agent_session_create():
    """Create or reset a chat session.

    Request body::

        { "session_id": "optional-custom-id" }

    If ``session_id`` is omitted, a new one is generated.
    If the session already exists, it is reset (history cleared, adata reset).
    """
    payload = request.json if request.json else {}
    session_id = (payload.get('session_id') or '').strip() or make_turn_id()
    reset = payload.get('reset', False)

    if reset:
        session_manager.delete_session(session_id)

    session = session_manager.create_session(session_id, base_adata=state.current_adata)
    return jsonify(session.to_summary())


@app.route('/api/agent/session/<session_id>/history', methods=['GET'])
def agent_session_history(session_id):
    """Return chat history for a session."""
    session = session_manager.get_session(session_id)
    if session is None:
        # Try loading from persistent storage
        history = session_manager.load_history(session_id)
        if history:
            return jsonify({'session_id': session_id, 'history': history, 'source': 'file'})
        return jsonify({'error': 'Session not found'}), 404

    return jsonify({
        'session_id': session_id,
        'history': session.get_history_dicts(),
        'trace_ids': list(session.trace_ids),
        'runtime': session.get_runtime_state(),
        'source': 'memory',
    })


@app.route('/api/agent/session/<session_id>/approvals', methods=['GET'])
def agent_session_approvals(session_id):
    """Return pending approvals for a session."""
    session = session_manager.get_session(session_id)
    if session is None:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({
        'session_id': session_id,
        'approvals': session.get_pending_approvals(),
    })


@app.route('/api/agent/session/<session_id>/questions', methods=['GET'])
def agent_session_questions(session_id):
    """Return pending questions for a session."""
    session = session_manager.get_session(session_id)
    if session is None:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({
        'session_id': session_id,
        'questions': session.get_pending_questions(),
    })


@app.route('/api/agent/session/<session_id>/tasks', methods=['GET'])
def agent_session_tasks(session_id):
    """Return recent runtime tasks for a session."""
    session = session_manager.get_session(session_id)
    if session is None:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({
        'session_id': session_id,
        'tasks': session.list_tasks(),
        'runtime': session.get_runtime_state(),
    })


@app.route('/api/agent/session/<session_id>', methods=['DELETE'])
def agent_session_delete(session_id):
    """Delete a session and cancel any active turn."""
    deleted = session_manager.delete_session(session_id)
    if not deleted:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({'deleted': True, 'session_id': session_id})


@app.route('/api/agent/sessions', methods=['GET'])
def agent_sessions_list():
    """List all active sessions."""
    sessions = session_manager.list_sessions()
    return jsonify({'sessions': sessions})


@app.route('/api/agent/chat/cancel', methods=['POST'])
def agent_chat_cancel():
    """Cancel the active agent turn for a session.

    Request body::

        { "session_id": "...", "turn_id": "..." }

    Either ``session_id`` or ``turn_id`` can be provided.
    If ``session_id`` is given, the active turn for that session is cancelled.
    If ``turn_id`` is given directly, that specific turn is cancelled.
    """
    payload = request.json if request.json else {}
    turn_id = (payload.get('turn_id') or '').strip()
    session_id = (payload.get('session_id') or '').strip()

    if not turn_id and session_id:
        # Look up active turn for this session
        turn_id = get_active_turn_for_session(session_id)
        if not turn_id:
            # Also try via session manager
            cancelled = session_manager.cancel_turn(session_id)
            if cancelled:
                return jsonify({'cancelled': True, 'session_id': session_id})
            return jsonify({'error': 'No active turn for session'}), 404

    if not turn_id:
        return jsonify({'error': 'Provide session_id or turn_id'}), 400

    cancelled = cancel_active_turn(turn_id)
    if not cancelled:
        return jsonify({'error': 'Turn not found or already finished'}), 404

    return jsonify({'cancelled': True, 'turn_id': turn_id})


@app.route('/api/agent/chat/approval', methods=['POST'])
def agent_chat_approval():
    """Record an approval response for a pending agent turn.

    The decision is persisted in session state and also applied to any
    currently blocked runtime approval broker for the matching turn.
    """
    payload = request.json if request.json else {}
    session_id = (payload.get('session_id') or '').strip()
    approval_id = (payload.get('approval_id') or '').strip()
    decision = (payload.get('decision') or '').strip().lower()

    if not session_id or not approval_id:
        return jsonify({'error': 'Provide session_id and approval_id'}), 400
    if decision not in {'approve', 'deny'}:
        return jsonify({'error': 'decision must be approve or deny'}), 400

    resolved = session_manager.resolve_approval(session_id, approval_id, decision)
    if resolved is None:
        return jsonify({'error': 'Approval not found'}), 404

    applied = resolve_pending_approval(approval_id, decision == 'approve')

    return jsonify({
        'recorded': True,
        'applied': applied,
        'approval': resolved,
    })


@app.route('/api/agent/chat/question', methods=['POST'])
def agent_chat_question():
    """Record an answer for a pending agent question."""
    payload = request.json if request.json else {}
    session_id = (payload.get('session_id') or '').strip()
    question_id = (payload.get('question_id') or '').strip()
    answer = str(payload.get('answer') or '').strip()

    if not session_id or not question_id:
        return jsonify({'error': 'Provide session_id and question_id'}), 400
    if not answer:
        return jsonify({'error': 'Provide a non-empty answer'}), 400

    resolved = session_manager.resolve_question(session_id, question_id, answer)
    if resolved is None:
        return jsonify({'error': 'Question not found'}), 404

    applied = resolve_pending_question(question_id, answer)

    return jsonify({
        'recorded': True,
        'applied': applied,
        'question': resolved,
    })


# ============================================================================
# Static File Routes
# ============================================================================

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    max_bytes = app.config.get('MAX_CONTENT_LENGTH')
    max_mb = int(max_bytes / (1024 * 1024)) if isinstance(max_bytes, (int, float)) and max_bytes else None
    payload = {'error': 'File too large'}
    if max_mb:
        payload['max_size_mb'] = max_mb
    return jsonify(payload), 413


@app.route('/')
def index():
    """Serve the OmicVerse homepage."""
    return send_from_directory(app.root_path, 'index.html')


@app.route('/analysis')
def analysis():
    """Serve single cell analysis interface."""
    return send_from_directory(app.root_path, 'single_cell_analysis_standalone.html')


@app.route('/legacy')
def legacy_index():
    """Serve legacy UI."""
    cra_build = os.path.join(app.root_path, 'design_ref', 'horizon-ui-chakra', 'build')
    cra_index = os.path.join(cra_build, 'index.html')
    if os.path.exists(cra_index):
        return send_from_directory(cra_build, 'index.html')

    ui_dist = os.path.join(app.root_path, 'ui', 'dist')
    ui_index = os.path.join(ui_dist, 'index.html')
    if os.path.exists(ui_index):
        return send_from_directory(ui_dist, 'index.html')

    return send_from_directory(app.root_path, 'index.html')


@app.route('/<path:path>')
def static_proxy(path):
    """Serve static files with SPA fallback."""
    cra_build = os.path.join(app.root_path, 'design_ref', 'horizon-ui-chakra', 'build')
    cra_file = os.path.join(cra_build, path)
    if os.path.exists(cra_file):
        return send_from_directory(cra_build, path)

    ui_dist = os.path.join(app.root_path, 'ui', 'dist')
    ui_file = os.path.join(ui_dist, path)
    if os.path.exists(ui_file):
        return send_from_directory(ui_dist, path)

    if path.startswith('static/'):
        rel = path[len('static/'):]
        return send_from_directory(app.static_folder, rel)

    full_path = os.path.join(app.root_path, path)
    if os.path.isfile(full_path):
        return send_from_directory(app.root_path, path)

    if os.path.exists(os.path.join(cra_build, 'index.html')):
        return send_from_directory(cra_build, 'index.html')
    if os.path.exists(os.path.join(ui_dist, 'index.html')):
        return send_from_directory(ui_dist, 'index.html')
    return send_from_directory(app.root_path, 'index.html')


# ============================================================================
# GPU (deck.gl) – color-only update (faster: no position fetch/encode)
# ============================================================================

@app.route('/api/plot_gpu_colors', methods=['POST'])
def plot_gpu_colors():
    """Return ONLY the colour + hover arrays for deck.gl (position unchanged).

    Use this endpoint when the embedding hasn't changed so we avoid re-fetching
    and re-encoding the large position FlatBuffers.

    Binary layout (all little-endian):
        [0-3]              uint32  n_cells
        [4-7]              uint32  json_len
        [8 .. 8+json_len)  UTF-8 JSON metadata (same schema as /api/plot_gpu)
        [padded to 4 bytes] uint8[n*4]   r,g,b,a colours
        [above + n*4]       float32[n]   hover_values
    """
    if state.current_adaptor is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        import struct as _struct, json as _json
        from server.common.fbs.matrix import decode_matrix_fbs
        import numpy as _np

        req       = request.json or {}
        embedding = req.get('embedding', '')
        x_axis_str = req.get('x_axis', '')
        y_axis_str = req.get('y_axis', '')
        color_by  = req.get('color_by', '')
        palette   = req.get('palette', 'viridis')
        cat_pal   = req.get('category_palette', None)
        vmin_req  = req.get('vmin', None)
        vmax_req  = req.get('vmax', None)
        density_adjust = _parse_density_adjust(req.get('density_adjust', 1.0))
        density_active = bool(req.get('density_active', False))
        n         = req.get('n_cells', None)   # supplied by frontend

        # If n_cells unknown, fetch from adaptor
        if n is None:
            n = state.current_adaptor.n_obs

        x_all = None
        y_all = None
        try:
            if x_axis_str and y_axis_str:
                adata = state.current_adata
                x_arr, _ = _resolve_axis(adata, x_axis_str)
                y_arr, _ = _resolve_axis(adata, y_axis_str)
                x_all = _np.asarray(x_arr, dtype=_np.float64)
                y_all = _np.asarray(y_arr, dtype=_np.float64)
            elif embedding:
                fbs_data = state.current_adaptor.get_embedding_fbs(embedding)
                coords_df = decode_matrix_fbs(fbs_data)
                x_all = _np.asarray(coords_df['x'], dtype=_np.float64)
                y_all = _np.asarray(coords_df['y'], dtype=_np.float64)
        except Exception as _coord_exc:
            logging.warning(f'plot_gpu_colors resolve coords failed: {_coord_exc}')
            x_all = None
            y_all = None

        colors_rgba  = _np.full((n, 4), [100, 149, 237, 200], dtype=_np.uint8)
        hover_values = _np.full(n, _np.nan, dtype=_np.float32)

        meta = {
            'n_total':         n,
            'color_label':     '',
            'is_categorical':  False,
            'category_labels': None,
            'colorscale':      None,
            'density_enabled': False,
            'density_applied': False,
            'density_adjust':  density_adjust,
            'density_message': 'Density adjust is off',
        }

        if color_by.startswith('obs:'):
            col_name = color_by[4:]
            obs_fbs  = state.current_adaptor.get_obs_fbs([col_name])
            obs_df   = decode_matrix_fbs(obs_fbs)
            if col_name in obs_df.columns:
                vals = obs_df[col_name]
                meta['color_label'] = col_name
                if pd.api.types.is_numeric_dtype(vals):
                    v    = vals.to_numpy(dtype=_np.float64)
                    if density_active and x_all is not None and y_all is not None and len(x_all) == len(v):
                        v, density_applied = _smooth_values_by_density(x_all, y_all, v, density_adjust)
                    else:
                        density_applied = False
                    vlo  = float(vmin_req) if vmin_req is not None else float(v.min())
                    vhi  = float(vmax_req) if vmax_req is not None else float(v.max())
                    span = max(vhi - vlo, 1e-10)
                    t    = _np.clip((v - vlo) / span, 0.0, 1.0)
                    from matplotlib import colormaps as _cmaps
                    cmap        = _cmaps.get_cmap(palette.lower() if palette else 'viridis')
                    rgba_f      = cmap(t)
                    colors_rgba = (_np.clip(rgba_f, 0, 1) * 255).astype(_np.uint8)
                    hover_values = v.astype(_np.float32)
                    meta['colorscale'] = palette or 'viridis'
                    meta['vmin'] = round(vlo, 6)
                    meta['vmax'] = round(vhi, 6)
                    meta['density_enabled'] = True
                    meta['density_applied'] = bool(density_applied)
                    meta['density_message'] = '' if density_active else 'Density adjust is off'
                else:
                    cats   = vals.astype('category')
                    labels = cats.cat.categories.tolist()
                    codes  = cats.cat.codes.to_numpy(dtype=_np.int32)
                    n_cats = len(labels)
                    if cat_pal:
                        disc = get_discrete_colors(n_cats, cat_pal)
                    else:
                        disc = (
                            get_uns_colors_for_labels(state.current_adata, col_name, labels)
                            or get_discrete_colors(n_cats, None)
                        )
                    cat_hex_list = []
                    for code, hex_c in enumerate(disc):
                        mask = codes == code
                        try:
                            r = int(hex_c[1:3], 16)
                            g = int(hex_c[3:5], 16)
                            b = int(hex_c[5:7], 16)
                        except Exception:
                            r, g, b = 100, 149, 237
                            hex_c = '#6495ed'
                        if mask.any():
                            colors_rgba[mask] = [r, g, b, 200]
                        cat_hex_list.append(hex_c)
                    hover_values = codes.astype(_np.float32)
                    meta['is_categorical']    = True
                    meta['category_labels']   = labels
                    meta['category_colors']   = cat_hex_list
                    meta['density_enabled'] = False
                    meta['density_applied'] = False
                    meta['density_message'] = 'Density adjust is disabled for non-numeric features'

        elif color_by.startswith('gene:'):
            gene_name = color_by[5:]
            meta['color_label'] = f'{gene_name} expression'
            try:
                expr_fbs = state.current_adaptor.get_expression_fbs([gene_name])
                expr_df  = decode_matrix_fbs(expr_fbs)
                if gene_name in expr_df.columns:
                    v    = expr_df[gene_name].to_numpy(dtype=_np.float64)
                    if density_active and x_all is not None and y_all is not None and len(x_all) == len(v):
                        v, density_applied = _smooth_values_by_density(x_all, y_all, v, density_adjust)
                    else:
                        density_applied = False
                    vlo  = float(vmin_req) if vmin_req is not None else float(v.min())
                    vhi  = float(vmax_req) if vmax_req is not None else float(v.max())
                    span = max(vhi - vlo, 1e-10)
                    t    = _np.clip((v - vlo) / span, 0.0, 1.0)
                    from matplotlib import colormaps as _cmaps
                    cmap        = _cmaps.get_cmap(palette.lower() if palette else 'viridis')
                    rgba_f      = cmap(t)
                    colors_rgba = (_np.clip(rgba_f, 0, 1) * 255).astype(_np.uint8)
                    hover_values = v.astype(_np.float32)
                    meta['colorscale'] = palette
                    meta['density_enabled'] = True
                    meta['density_applied'] = bool(density_applied)
                    meta['density_message'] = '' if density_active else 'Density adjust is off'
            except Exception as _ge:
                logging.warning(f'plot_gpu_colors gene {gene_name}: {_ge}')

        # ── pack binary: header + json + colors + hover ───────────────────────
        meta_json = _json.dumps(meta).encode('utf-8')
        json_len  = len(meta_json)
        total_hdr = 8 + json_len
        pad       = (4 - (total_hdr % 4)) % 4   # align to 4 bytes (uint8/float32)

        buf = (
            _struct.pack('<II', n, json_len)
            + meta_json
            + b'\x00' * pad
            + colors_rgba.tobytes()   # n * 4 bytes
            + hover_values.tobytes()  # n * 4 bytes
        )

        from flask import Response as _Resp
        return _Resp(buf, mimetype='application/octet-stream',
                     headers={'Cache-Control': 'no-cache'})

    except Exception as e:
        logging.error(f'plot_gpu_colors failed: {e}')
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Initialize
# ============================================================================

# Ensure default notebook exists
ensure_default_notebook(state.file_root)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(debug=True, host='0.0.0.0', port=port)

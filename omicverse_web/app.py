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
from services.agent_service import get_agent_instance, run_agent_stream, run_agent_chat
from utils.notebook_helpers import ensure_default_notebook

# Import blueprints
from routes import kernel, files, data, notebooks

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

# Global state container (for easier blueprint access)
class AppState:
    """Container for global application state."""
    def __init__(self):
        self.current_adaptor = None
        self.current_adata = None
        self.current_filename = None
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


def get_discrete_colors(n_categories, palette_name=None):
    """Get discrete color palette for categorical data."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colors import ListedColormap

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


# ============================================================================
# Code Execution Routes (not in blueprints due to complexity)
# ============================================================================

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

        result = execution.get('result')
        if result is not None:
            result = str(result)

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
                'embeddings': [emb.replace('X_', '') for emb in state.current_adata.obsm.keys()],
                'obs_columns': list(state.current_adata.obs.columns),
                'var_columns': list(state.current_adata.var.columns)
            }

        return jsonify({
            'output': output,
            'result': result,
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
                            'embeddings': [emb.replace('X_', '') for emb in state.current_adata.obsm.keys()],
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
                # Convert result to string for JSON serialization
                result_value = None
                if execution_result['result']:
                    raw_result = execution_result['result'].get('result')
                    if raw_result is not None:
                        result_value = str(raw_result)

                result_data = {
                    'type': 'complete',
                    'output': execution_result['result'].get('output', '') if execution_result['result'] else '',
                    'error': execution_result['result'].get('error') if execution_result['result'] else None,
                    'result': result_value,
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

def _analyze_data_state(adata):
    """Heuristically analyze the preprocessing state of an AnnData object.

    Returns a dict with:
        x_max, x_min, is_int, is_log1p, is_normalized,
        estimated_target_sum, is_scaled
    """
    import numpy as _np
    import scipy.sparse as _sp

    try:
        X = adata.X
        # Sample up to 5000 cells for speed
        n_sample = min(adata.n_obs, 5000)
        if adata.n_obs > n_sample:
            idx = _np.random.choice(adata.n_obs, n_sample, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        # Convert to dense only if not too large
        if _sp.issparse(X_sample):
            x_max_val = float(X_sample.max())
            x_min_val = float(X_sample.min())
            # Check for integer values using a small slice of nonzero entries
            coo = X_sample.tocoo()
            nz = coo.data[:500] if len(coo.data) > 500 else coo.data
            X_dense_sample = X_sample[:200].toarray() if X_sample.shape[0] >= 200 else X_sample.toarray()
        else:
            X_sample_arr = _np.asarray(X_sample)
            x_max_val = float(_np.nanmax(X_sample_arr))
            x_min_val = float(_np.nanmin(X_sample_arr))
            nz_mask = X_sample_arr != 0
            nz = X_sample_arr[nz_mask][:500]
            X_dense_sample = X_sample_arr[:200]

        # is_int: are nonzero values all integers?
        is_int = bool(len(nz) > 0 and _np.all(_np.abs(nz - _np.round(nz)) < 1e-4))

        # is_log1p: reliable check = 'log1p' key in uns
        is_log1p_uns = 'log1p' in adata.uns
        # Fallback heuristic: max < 20 and not integer
        is_log1p_heuristic = (not is_int) and (x_max_val < 30)
        is_log1p = bool(is_log1p_uns or is_log1p_heuristic)

        has_negative = bool(x_min_val < 0)

        # is_scaled: data has negative values and max ≤ 50 (typical post-scale range)
        is_scaled = bool(has_negative and x_max_val <= 50)

        # is_normalized: check coefficient of variation of row sums
        row_sums = _np.asarray(X_dense_sample.sum(axis=1)).ravel().astype(float)
        row_sums = row_sums[row_sums > 0]
        if len(row_sums) >= 5:
            cv = float(_np.std(row_sums) / _np.mean(row_sums)) if _np.mean(row_sums) > 0 else 1.0
            is_normalized = bool(cv < 0.05)
        else:
            is_normalized = False

        # estimated_target_sum — only computable for non-log1p normalized data
        estimated_target_sum = None
        _COMMON_TS = [500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
        if is_normalized and not is_scaled and not is_log1p:
            row_sums_all = _np.asarray(X_dense_sample.sum(axis=1)).ravel()
            row_sums_all = row_sums_all[row_sums_all > 0]
            if len(row_sums_all) > 0:
                median_ts = float(_np.median(row_sums_all))
                closest = min(_COMMON_TS, key=lambda v: abs(v - median_ts))
                estimated_target_sum = closest if abs(closest - median_ts) / (median_ts + 1e-9) < 0.5 else None

        return {
            'x_max': round(x_max_val, 4),
            'x_min': round(x_min_val, 4),
            'is_int': is_int,
            'is_log1p': is_log1p,
            'is_normalized': is_normalized,
            'is_scaled': is_scaled,
            'estimated_target_sum': estimated_target_sum,
        }
    except Exception as _e:
        logging.warning(f"_analyze_data_state failed: {_e}")
        return {}


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
    return jsonify({
        'loaded':      True,
        'filename':    state.current_filename or 'data.h5ad',
        'n_cells':     state.current_adata.n_obs,
        'n_genes':     state.current_adata.n_vars,
        'embeddings':  [k.replace('X_', '') for k in state.current_adata.obsm.keys()],
        'obs_columns': list(state.current_adata.obs.columns),
        'var_columns': list(state.current_adata.var.columns),
        'uns_keys':    list(state.current_adata.uns.keys()),
        'layers':      list(state.current_adata.layers.keys()),
        'data_state':  _analyze_data_state(state.current_adata),
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

        # Build expression matrix
        if layer and layer in adata.layers:
            X = adata.layers[layer]
        else:
            X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        else:
            X = _np.asarray(X)

        # Gene indices
        var_names = list(adata.var_names)
        gene_idx  = [var_names.index(g) for g in genes]
        expr      = X[:, gene_idx]           # (n_cells, n_genes)

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


@app.route('/api/plot', methods=['POST'])
def plot_data_legacy():
    """Legacy plot endpoint - returns JSON data for Plotly."""
    if state.current_adaptor is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        from server.common.fbs.matrix import decode_matrix_fbs

        data_req = request.json
        embedding = data_req.get('embedding', '')
        color_by = data_req.get('color_by', '')
        palette = data_req.get('palette', None)
        category_palette = data_req.get('category_palette', None)
        vmin = data_req.get('vmin', None)
        vmax = data_req.get('vmax', None)

        if not embedding:
            return jsonify({'error': 'No embedding specified'}), 400

        # Get embedding data
        fbs_data = state.current_adaptor.get_embedding_fbs(embedding)
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
                obs_fbs = state.current_adaptor.get_obs_fbs([col_name])
                obs_df = decode_matrix_fbs(obs_fbs)

                if col_name in obs_df.columns:
                    values = obs_df[col_name]
                    if pd.api.types.is_numeric_dtype(values):
                        colors_array = values.values
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
                        categories = values.astype('category') if not pd.api.types.is_categorical_dtype(values) else values
                        plot_data['color_label'] = col_name
                        plot_data['category_labels'] = categories.cat.categories.tolist()
                        plot_data['category_codes'] = categories.cat.codes.tolist()
                        n_categories = len(categories.cat.categories)
                        discrete_colors = get_discrete_colors(n_categories, category_palette)
                        plot_data['colors'] = [discrete_colors[code] for code in categories.cat.codes]
                        plot_data['discrete_colors'] = discrete_colors

            elif color_by.startswith('gene:'):
                gene_name = color_by.replace('gene:', '')
                try:
                    expr_fbs = state.current_adaptor.get_expression_fbs([gene_name])
                    expr_df = decode_matrix_fbs(expr_fbs)
                    if gene_name in expr_df.columns:
                        expression = expr_df[gene_name].values
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

    try:
        agent = get_agent_instance(config)
        if state.current_adata is None:
            reply = run_agent_chat(agent, prompt)
            return jsonify({
                'reply': reply,
                'code': None,
                'data_updated': False,
                'data_info': None
            })

        result = run_agent_stream(agent, prompt, state.current_adata)
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
# Initialize
# ============================================================================

# Ensure default notebook exists
ensure_default_notebook(state.file_root)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(debug=True, host='0.0.0.0', port=port)

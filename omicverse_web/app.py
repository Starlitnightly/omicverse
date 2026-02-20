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
    except Exception:
        pass


# Helper function for discrete colors
def get_discrete_colors(n_categories, palette_name=None):
    """Get discrete color palette for categorical data."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if palette_name and palette_name in plt.colormaps():
        cmap = plt.get_cmap(palette_name)
        colors = [mcolors.to_hex(cmap(i / max(1, n_categories - 1))) for i in range(n_categories)]
    else:
        # Default: use tab20 or Set3
        if n_categories <= 20:
            cmap = plt.get_cmap('tab20')
            colors = [mcolors.to_hex(cmap(i)) for i in range(n_categories)]
        else:
            cmap = plt.get_cmap('hsv')
            colors = [mcolors.to_hex(cmap(i / n_categories)) for i in range(n_categories)]

    return colors


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

def _capture_tool_output(func):
    """Run func() while capturing stdout, stderr, and scanpy/root logging into one string."""
    import contextlib

    output_buf = io.StringIO()

    # Handler that writes into our buffer
    log_handler = logging.StreamHandler(output_buf)
    log_handler.setLevel(logging.DEBUG)
    log_handler.setFormatter(logging.Formatter('%(message)s'))

    sc_logger = logging.getLogger('scanpy')
    root_logger = logging.getLogger()

    # Temporarily lower scanpy logger level so info/hint messages are emitted
    prev_sc_level = sc_logger.level
    prev_root_level = root_logger.level
    sc_logger.setLevel(logging.DEBUG)
    # Don't lower root logger – it's too noisy; Flask/werkzeug use it
    sc_logger.addHandler(log_handler)
    root_logger.addHandler(log_handler)

    # Also temporarily raise scanpy verbosity so logg.info / logg.hint fire
    try:
        import scanpy as _sc
        prev_verbosity = _sc.settings.verbosity
        _sc.settings.verbosity = 3  # hint level: info + hints
    except Exception:
        prev_verbosity = None

    try:
        with contextlib.redirect_stdout(output_buf), \
             contextlib.redirect_stderr(output_buf):
            func()
    finally:
        sc_logger.removeHandler(log_handler)
        root_logger.removeHandler(log_handler)
        sc_logger.setLevel(prev_sc_level)
        root_logger.setLevel(prev_root_level)
        if prev_verbosity is not None:
            try:
                _sc.settings.verbosity = prev_verbosity
            except Exception:
                pass

    return output_buf.getvalue().strip()


@app.route('/api/tools/<tool>', methods=['POST'])
def run_tool(tool):
    """Run single-cell analysis tools, capturing all output for the analysis log."""
    if state.current_adata is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        params = request.json if request.json else {}
        captured = ''

        # ── Preprocessing ────────────────────────────────────────────────────
        if tool == 'normalize':
            target_sum = params.get('target_sum', 1e4)
            def _run(): sc.pp.normalize_total(state.current_adata, target_sum=target_sum)
            captured = _capture_tool_output(_run)

        elif tool == 'log1p':
            def _run(): sc.pp.log1p(state.current_adata)
            captured = _capture_tool_output(_run)

        elif tool == 'scale':
            max_value = params.get('max_value', 10)
            def _run(): sc.pp.scale(state.current_adata, max_value=max_value)
            captured = _capture_tool_output(_run)

        elif tool == 'hvg':
            n_total = state.current_adata.n_vars
            n_genes = params.get('n_genes', 2000)
            method  = params.get('method', 'seurat')
            def _run(): sc.pp.highly_variable_genes(state.current_adata, flavor=method, n_top_genes=n_genes)
            captured = _capture_tool_output(_run)
            state.current_adata = state.current_adata[:, state.current_adata.var.highly_variable].copy()
            n_hvg = state.current_adata.n_vars
            prefix = (captured + '\n') if captured else ''
            captured = prefix + f'hvg: selected {n_hvg} highly variable genes (from {n_total})'


        # ── QC ───────────────────────────────────────────────────────────────
        elif tool == 'filter_cells':
            before = state.current_adata.n_obs
            for key in ('min_counts', 'min_genes', 'max_counts', 'max_genes'):
                v = params.get(key, None)
                if v is None or v == '':
                    continue
                try:
                    v = int(v)
                except Exception:
                    continue
                if key == 'min_counts':
                    sc.pp.filter_cells(state.current_adata, min_counts=v)
                elif key == 'min_genes':
                    sc.pp.filter_cells(state.current_adata, min_genes=v)
                elif key == 'max_counts':
                    sc.pp.filter_cells(state.current_adata, max_counts=v)
                elif key == 'max_genes':
                    sc.pp.filter_cells(state.current_adata, max_genes=v)
            after = state.current_adata.n_obs
            captured = (f'filter_cells: {before} → {after} cells '
                        f'(removed {before - after})')

        elif tool == 'filter_genes':
            import numpy as _np
            before = state.current_adata.n_vars
            min_cells   = params.get('min_cells', None)
            g_min_counts = params.get('min_counts', params.get('g_min_counts', None))
            max_cells   = params.get('max_cells', None)
            g_max_counts = params.get('max_counts', params.get('g_max_counts', None))
            if g_min_counts is not None and g_min_counts != '':
                sc.pp.filter_genes(state.current_adata, min_counts=int(g_min_counts))
            if min_cells is not None and min_cells != '':
                sc.pp.filter_genes(state.current_adata, min_cells=int(min_cells))
            X = (state.current_adata.X.toarray()
                 if hasattr(state.current_adata.X, 'toarray')
                 else state.current_adata.X)
            if max_cells is not None:
                expr_cells = (X > 0).sum(axis=0)
                expr_cells = (_np.asarray(expr_cells).A1
                              if hasattr(expr_cells, 'A1') else _np.asarray(expr_cells))
                state.current_adata._inplace_subset_var(expr_cells <= int(max_cells))
                X = (state.current_adata.X.toarray()
                     if hasattr(state.current_adata.X, 'toarray')
                     else state.current_adata.X)
            if g_max_counts is not None:
                sums = _np.asarray(X).sum(axis=0)
                sums = sums.A1 if hasattr(sums, 'A1') else sums
                state.current_adata._inplace_subset_var(sums <= float(g_max_counts))
            after = state.current_adata.n_vars
            captured = (f'filter_genes: {before} → {after} genes '
                        f'(removed {before - after})')

        elif tool == 'filter_outliers':
            import numpy as _np, re as _re
            before = state.current_adata.n_obs
            # Auto-detect mitochondrial gene prefix
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
                state.current_adata, qc_vars=['mt', 'ribo', 'hb'],
                inplace=True, log1p=True)
            keep = _np.ones(state.current_adata.n_obs, dtype=bool)
            for pct_param, col_hint in [
                ('max_mt_percent', 'mt'),
                ('max_ribo_percent', 'ribo'),
                ('max_hb_percent', 'hb'),
            ]:
                threshold = params.get(pct_param)
                if threshold is None:
                    continue
                pct_col = next(
                    (c for c in state.current_adata.obs.columns
                     if 'pct' in c.lower() and col_hint in c.lower()), None)
                if pct_col:
                    keep &= (state.current_adata.obs[pct_col].astype(float)
                             <= float(threshold)).values
            if keep.sum() != state.current_adata.n_obs:
                state.current_adata._inplace_subset_obs(keep)
            after = state.current_adata.n_obs
            n_mt = int(mt_mask.sum())
            captured = (f'filter_outliers: {before} → {after} cells '
                        f'(removed {before - after})\n'
                        f'  detected {n_mt} mitochondrial genes '
                        f'(prefixes: {", ".join(mt_prefixes)})')

        elif tool == 'doublets':
            before = state.current_adata.n_obs
            batch_key = params.get('batch_key') or None
            scrublet_kwargs = dict(
                sim_doublet_ratio=float(params.get('sim_doublet_ratio', 2.0)),
                expected_doublet_rate=float(params.get('expected_doublet_rate', 0.05)),
                stdev_doublet_rate=float(params.get('stdev_doublet_rate', 0.02)),
                synthetic_doublet_umi_subsampling=float(
                    params.get('synthetic_doublet_umi_subsampling', 1.0)),
                knn_dist_metric=params.get('knn_dist_metric', 'euclidean'),
                normalize_variance=bool(params.get('normalize_variance', True)),
                log_transform=bool(params.get('log_transform', False)),
                mean_center=bool(params.get('mean_center', True)),
                n_prin_comps=int(params.get('n_prin_comps', 30))
            )
            try:
                sc.pp.scrublet(state.current_adata, batch_key=batch_key,
                               **scrublet_kwargs)
            except Exception as e:
                return jsonify({'error': f'Scrublet 执行失败: {e}'}), 400
            pred_col = next((c for c in ('predicted_doublet', 'predicted_doublets')
                             if c in state.current_adata.obs.columns), None)
            score_col = next((c for c in ('doublet_score', 'doublet_scores')
                              if c in state.current_adata.obs.columns), None)
            n_doublets = 0
            if pred_col:
                n_doublets = int(state.current_adata.obs[pred_col].astype(bool).sum())
                keep = ~state.current_adata.obs[pred_col].astype(bool)
                state.current_adata._inplace_subset_obs(keep.values)
            after = state.current_adata.n_obs
            score_info = (f', median score={state.current_adata.obs[score_col].median():.3f}'
                          if score_col and after > 0 else '')
            captured = (f'doublets: {n_doublets} doublets removed, '
                        f'{before} → {after} cells{score_info}')

        # ── Dimensionality reduction ──────────────────────────────────────────
        elif tool == 'pca':
            n_comps = params.get('n_comps', 50)
            def _run():
                sc.pp.pca(state.current_adata,
                          n_comps=min(n_comps, min(state.current_adata.shape) - 1))
            captured = _capture_tool_output(_run)

        elif tool == 'neighbors':
            n_neighbors = params.get('n_neighbors', 15)
            n_pcs       = params.get('n_pcs', 50)
            def _run():
                sc.pp.neighbors(state.current_adata,
                                n_neighbors=n_neighbors, n_pcs=n_pcs)
            captured = _capture_tool_output(_run)

        elif tool == 'umap':
            min_dist = params.get('min_dist', 0.5)
            def _run():
                if 'neighbors' not in state.current_adata.uns:
                    sc.pp.neighbors(state.current_adata, n_neighbors=15,
                                    n_pcs=30, use_rep='X_pca')
                sc.tl.umap(state.current_adata, min_dist=min_dist)
            captured = _capture_tool_output(_run)

        elif tool == 'tsne':
            perplexity = params.get('perplexity', 30)
            def _run(): sc.tl.tsne(state.current_adata, perplexity=perplexity)
            captured = _capture_tool_output(_run)

        # ── Clustering ───────────────────────────────────────────────────────
        elif tool == 'leiden':
            resolution = params.get('resolution', 1.0)
            def _run(): sc.tl.leiden(state.current_adata, resolution=resolution)
            captured = _capture_tool_output(_run)

        elif tool == 'louvain':
            resolution = params.get('resolution', 1.0)
            def _run(): sc.tl.louvain(state.current_adata, resolution=resolution)
            captured = _capture_tool_output(_run)

        else:
            return jsonify({'error': f'Unknown tool: {tool}'}), 400

        # Sync adaptor
        sync_adaptor_with_adata()

        return jsonify({
            'success': True,
            'n_cells': state.current_adata.n_obs,
            'n_genes': state.current_adata.n_vars,
            'embeddings': [emb.replace('X_', '') for emb in state.current_adata.obsm.keys()],
            'obs_columns': list(state.current_adata.obs.columns),
            'stdout': captured
        })

    except Exception as e:
        logging.error(f"Tool {tool} failed: {e}")
        return jsonify({'error': str(e)}), 500


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

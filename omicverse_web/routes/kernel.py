"""
Kernel Routes - IPython Kernel API Endpoints
=============================================
Flask blueprint for kernel management and code execution endpoints.
"""

import json
import logging
import time
import queue
import io
import threading
from flask import Blueprint, request, jsonify, Response, stream_with_context

from services.kernel_service import (
    normalize_kernel_id,
    get_kernel_context,
    get_execution_state,
    request_interrupt as request_kernel_interrupt,
    execution_state_lock,
    execution_state,
)
from utils.variable_helpers import summarize_var, resolve_var_path
from utils.memory_helpers import estimate_var_size, get_process_memory_mb
from utils.adata_helpers import canonical_embedding_keys as _canonical_embedding_keys


# Create blueprint
bp = Blueprint('kernel', __name__)


@bp.route('/stats', methods=['GET'])
def kernel_stats():
    """Get kernel memory statistics and top variables."""
    try:
        kernel_id = request.args.get('kernel_id')
        executor, ns = get_kernel_context(kernel_id, bp.state.kernel_executor, bp.state.kernel_sessions)
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

        memory_mb = get_process_memory_mb()
        return jsonify({
            'memory_mb': memory_mb if memory_mb is not None else 0,
            'vars': vars_info[:10],
            'kernel_id': normalize_kernel_id(kernel_id)
        })
    except Exception as e:
        logging.error(f"Kernel stats failed: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/vars', methods=['GET'])
def kernel_vars():
    """Get list of all variables in kernel namespace."""
    try:
        kernel_id = request.args.get('kernel_id')
        executor, ns = get_kernel_context(kernel_id, bp.state.kernel_executor, bp.state.kernel_sessions)
        executor._ensure_kernel()

        vars_info = []
        for name, value in ns.items():
            if name.startswith('_'):
                continue
            # Skip IPython internal variables
            if name in ['In', 'Out', 'get_ipython', 'exit', 'quit']:
                continue
            # Skip callable objects (functions, classes)
            if callable(value):
                continue
            # Skip modules
            if hasattr(value, '__name__') and hasattr(value, '__file__'):
                continue
            # Check if it's a module type
            if type(value).__name__ == 'module':
                continue
            module_name = getattr(value, '__module__', '')
            if module_name.startswith('builtins'):
                continue

            vars_info.append(summarize_var(name, value))

            # Add AnnData sub-attributes
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

        return jsonify({
            'vars': vars_info[:50],
            'kernel_id': normalize_kernel_id(kernel_id)
        })
    except Exception as e:
        logging.error(f"Kernel vars failed: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/var_detail', methods=['GET'])
def kernel_var_detail():
    """Get detailed information about a specific variable."""
    name = request.args.get('name', '')
    try:
        kernel_id = request.args.get('kernel_id')
        executor, ns = get_kernel_context(kernel_id, bp.state.kernel_executor, bp.state.kernel_sessions)
        executor._ensure_kernel()
        value = resolve_var_path(name, ns)

        # Try to convert to DataFrame for display
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
            elif isinstance(value, pd.Series):
                df = value.to_frame().iloc[:50, :1]
                return jsonify({
                    'type': 'dataframe',
                    'name': name,
                    'shape': [len(value), 1],
                    'table': df.to_dict(orient='split')
                })
        except Exception:
            pass

        # Try to handle AnnData
        try:
            if value.__class__.__name__ == 'AnnData':
                return jsonify({
                    'type': 'anndata',
                    'name': name,
                    'summary': {
                        'shape': list(value.shape),
                        'obs_columns': list(value.obs.columns),
                        'var_columns': list(value.var.columns),
                        'obsm_keys': list(value.obsm.keys()) if value.obsm else [],
                        'layers': list(value.layers.keys()) if value.layers else []
                    }
                })
        except Exception:
            pass

        # Fallback to string representation
        return jsonify({
            'name': name,
            'type': type(value).__name__,
            'content': str(value)[:10000]
        })

    except Exception as e:
        logging.error(f"Variable detail failed: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/adata_slot', methods=['GET'])
def kernel_adata_slot():
    """Get detail for a specific slot/key within an AnnData variable."""
    var_name = request.args.get('var_name', '')
    slot = request.args.get('slot', '')
    key = request.args.get('key', '')
    try:
        kernel_id = request.args.get('kernel_id')
        executor, ns = get_kernel_context(kernel_id, bp.state.kernel_executor, bp.state.kernel_sessions)
        executor._ensure_kernel()

        if not var_name or var_name not in ns:
            return jsonify({'error': f'Variable "{var_name}" not found'}), 404

        adata = ns[var_name]
        if adata.__class__.__name__ != 'AnnData':
            return jsonify({'error': f'"{var_name}" is not an AnnData object'}), 400

        import pandas as pd
        import numpy as np

        slot_obj = getattr(adata, slot, None)
        if slot_obj is None:
            return jsonify({'error': f'Slot "{slot}" not found on {var_name}'}), 404

        if slot in ('obs', 'var'):
            if key:
                series = slot_obj[key]
                df = series.to_frame().iloc[:100]
                return jsonify({
                    'type': 'dataframe',
                    'name': f'{var_name}.{slot}["{key}"]',
                    'shape': [len(slot_obj), 1],
                    'table': df.to_dict(orient='split')
                })
            else:
                df = slot_obj.iloc[:50, :20]
                return jsonify({
                    'type': 'dataframe',
                    'name': f'{var_name}.{slot}',
                    'shape': list(slot_obj.shape),
                    'table': df.to_dict(orient='split')
                })

        elif slot in ('obsm', 'varm', 'obsp', 'varp'):
            arr = slot_obj[key]
            shape = list(arr.shape) if hasattr(arr, 'shape') else []
            try:
                if hasattr(arr, 'toarray'):
                    preview_arr = arr[:10, :10].toarray()
                elif isinstance(arr, np.ndarray):
                    preview_arr = arr[:10, :10]
                else:
                    preview_arr = None
                if preview_arr is not None:
                    cols = [str(i) for i in range(preview_arr.shape[1])]
                    idx = [str(i) for i in range(preview_arr.shape[0])]
                    return jsonify({
                        'type': 'dataframe',
                        'name': f'{var_name}.{slot}["{key}"]',
                        'shape': shape,
                        'table': {'columns': cols, 'index': idx, 'data': preview_arr.tolist()}
                    })
            except Exception:
                pass
            return jsonify({
                'type': 'content',
                'name': f'{var_name}.{slot}["{key}"]',
                'content': f'shape={shape}\n{str(arr)[:2000]}'
            })

        elif slot == 'layers':
            layer = slot_obj[key]
            shape = list(layer.shape) if hasattr(layer, 'shape') else []
            try:
                if hasattr(layer, 'toarray'):
                    preview_arr = layer[:10, :10].toarray()
                elif isinstance(layer, np.ndarray):
                    preview_arr = layer[:10, :10]
                else:
                    preview_arr = None
                if preview_arr is not None:
                    cols = [str(i) for i in range(preview_arr.shape[1])]
                    idx = [str(i) for i in range(preview_arr.shape[0])]
                    return jsonify({
                        'type': 'dataframe',
                        'name': f'{var_name}.layers["{key}"]',
                        'shape': shape,
                        'table': {'columns': cols, 'index': idx, 'data': preview_arr.tolist()}
                    })
            except Exception:
                pass
            return jsonify({
                'type': 'content',
                'name': f'{var_name}.layers["{key}"]',
                'content': f'shape={shape}'
            })

        elif slot == 'uns':
            val = slot_obj[key]
            if isinstance(val, pd.DataFrame):
                df = val.iloc[:50, :20]
                return jsonify({
                    'type': 'dataframe',
                    'name': f'{var_name}.uns["{key}"]',
                    'shape': list(val.shape),
                    'table': df.to_dict(orient='split')
                })
            return jsonify({
                'type': 'content',
                'name': f'{var_name}.uns["{key}"]',
                'content': str(val)[:5000]
            })

        else:
            return jsonify({'error': f'Unsupported slot: {slot}'}), 400

    except Exception as e:
        logging.error(f'adata_slot failed: {e}')
        return jsonify({'error': str(e)}), 500


@bp.route('/interrupt', methods=['POST'])
def kernel_interrupt():
    """Request interrupt of current code execution."""
    try:
        result = request_kernel_interrupt()
        if not result['success']:
            return jsonify(result), 400

        # Give it a moment to interrupt gracefully
        time.sleep(0.1)

        return jsonify(result)
    except Exception as e:
        logging.error(f"Interrupt failed: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/restart', methods=['POST'])
def kernel_restart():
    """Restart kernel - clear all variables and reset state."""
    try:
        payload = request.get_json(silent=True) or {}
        kernel_id = normalize_kernel_id(payload.get('kernel_id', 'default.ipynb'))

        logging.info(f"Restarting kernel: {kernel_id}")

        # Get the kernel executor
        executor, ns = get_kernel_context(kernel_id, bp.state.kernel_executor, bp.state.kernel_sessions)

        # Clear the namespace (remove all user variables)
        # Keep built-in variables and imports
        user_vars = [
            key for key in list(ns.keys())
            if not key.startswith('_') and key not in ['In', 'Out', 'get_ipython', 'exit', 'quit']
        ]
        for var in user_vars:
            del ns[var]

        # Clear execution state
        with execution_state_lock:
            execution_state['is_executing'] = False
            execution_state['interrupt_requested'] = False
            execution_state['execution_id'] = None
            execution_state['start_time'] = None

        # Clear global current_adata if this is the default kernel
        if kernel_id == 'default.ipynb':
            bp.state.current_adata = None
            bp.state.current_adaptor = None
            bp.state.current_filename = None

        logging.info(f"Kernel {kernel_id} restarted successfully")

        return jsonify({
            'success': True,
            'message': f'Kernel {kernel_id} restarted',
            'cleared_vars': len(user_vars)
        })

    except Exception as e:
        import traceback
        logging.error(f"Kernel restart failed: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@bp.route('/status', methods=['GET'])
def kernel_execution_status():
    """Get current code execution status."""
    try:
        status = get_execution_state()

        # Add elapsed time if executing
        if status['start_time']:
            status['elapsed_seconds'] = time.time() - status['start_time']
            # Remove start_time from response (internal detail)
            del status['start_time']

        return jsonify(status)
    except Exception as e:
        logging.error(f"Status check failed: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/list', methods=['GET'])
def kernel_list():
    """List available kernels."""
    return jsonify({
        'kernels': list(bp.state.kernel_names.keys()),
        'current': bp.state.current_kernel_name
    })


@bp.route('/select', methods=['POST'])
def kernel_select():
    """Select active kernel."""
    payload = request.get_json(silent=True) or {}
    name = payload.get('name', 'python3')
    bp.state.current_kernel_name = name
    return jsonify({'success': True, 'current': name})


@bp.route('/load_adata', methods=['POST'])
def kernel_load_adata():
    """Load an AnnData object from kernel variables to visualization."""
    try:
        payload = request.get_json(silent=True) or {}
        var_name = payload.get('var_name')
        kernel_id = normalize_kernel_id(payload.get('kernel_id'))

        logging.info(f"Loading AnnData: var_name={var_name}, kernel_id={kernel_id}")

        if not var_name:
            return jsonify({'error': 'Missing variable name'}), 400

        # Get kernel context and retrieve the variable
        executor, ns = get_kernel_context(kernel_id, bp.state.kernel_executor, bp.state.kernel_sessions)
        executor._ensure_kernel()

        # Resolve the variable (handle paths like 'obj.adata')
        value = resolve_var_path(var_name, ns)

        # Check if it's an AnnData object
        if value is None:
            return jsonify({'error': f'Variable "{var_name}" not found'}), 404

        if value.__class__.__name__ != 'AnnData':
            return jsonify({'error': f'Variable "{var_name}" is not an AnnData object (type: {type(value).__name__})'}), 400

        # Load the AnnData to current_adata
        bp.state.current_adata = value
        bp.state.current_filename = f'{var_name} (from kernel)'
        logging.info(f"Loaded AnnData from variable '{var_name}': {bp.state.current_adata.shape}")

        # Create adaptor if needed (similar to upload endpoint)
        try:
            from server.data_adaptor.anndata_adaptor import HighPerformanceAnndataAdaptor
            bp.state.current_adaptor = HighPerformanceAnndataAdaptor(bp.state.current_adata)
            logging.info("Created adaptor successfully")
        except Exception as e:
            logging.error(f"Failed to create adaptor: {e}")
            # Return error if adaptor creation failed
            return jsonify({'error': f'Failed to create data adaptor: {str(e)}'}), 500

        # Sync back to kernel namespace if needed
        try:
            if kernel_id == 'default.ipynb':
                bp.state.kernel_executor.sync_adata(bp.state.current_adata)
        except Exception:
            pass

        # Return data info
        from utils.adata_helpers import analyze_data_state as _analyze_data_state
        adata = bp.state.current_adata
        data_info = {
            'filename': bp.state.current_filename,
            'n_cells': adata.n_obs,
            'n_genes': adata.n_vars,
            'embeddings': _canonical_embedding_keys(adata),
            'obs_columns': list(adata.obs.columns),
            'var_columns': list(adata.var.columns),
            'uns_keys':    list(adata.uns.keys()),
            'layers':      list(adata.layers.keys()),
            'data_state':  _analyze_data_state(adata),
        }

        return jsonify({
            'success': True,
            'message': f'Loaded {var_name} to visualization',
            'data_info': data_info
        })

    except Exception as e:
        import traceback
        logging.error(f"Load AnnData failed: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@bp.route('/sync_odata', methods=['POST'])
def sync_odata():
    """Sync current visualization adata into kernel namespace as both 'adata' and 'odata'."""
    if bp.state.current_adata is None:
        return jsonify({'success': False, 'message': 'No data loaded'})
    try:
        bp.state.kernel_executor.sync_adata(bp.state.current_adata)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# Initialize blueprint with dependencies (will be set by app.py)
bp.state = None

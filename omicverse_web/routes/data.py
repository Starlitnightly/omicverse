"""
Data Routes - Data Upload and Query API Endpoints
==================================================
Flask blueprint for AnnData file upload and data retrieval endpoints.
"""

import os
import logging
import numpy as np
from flask import Blueprint, request, jsonify, send_file, make_response
from werkzeug.utils import secure_filename

from server.data_adaptor.anndata_adaptor import HighPerformanceAnndataAdaptor


# Create blueprint
bp = Blueprint('data', __name__)


@bp.route('/upload', methods=['POST'])
def upload_file():
    """Upload h5ad file and create data adaptor."""
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
        filepath = os.path.join(bp.upload_folder, filename)
        file.save(filepath)

        # Create high-performance data adaptor
        bp.state.current_adaptor = HighPerformanceAnndataAdaptor(filepath)
        # Expose underlying AnnData for tool endpoints
        bp.state.current_adata = bp.state.current_adaptor.adata
        bp.state.current_filename = filename

        try:
            bp.state.kernel_executor.sync_adata(bp.state.current_adata)
        except Exception:
            pass

        # Get schema and summary
        schema = bp.state.current_adaptor.get_schema()
        summary = bp.state.current_adaptor.get_data_summary()
        chunk_info = bp.state.current_adaptor.get_chunk_info()

        # Build response compatible with legacy UI expectations
        response_data = {
            'filename': filename,
            # Legacy/UI fields used by single-cell.js
            'n_cells': summary.get('n_obs', bp.state.current_adaptor.n_obs),
            'n_genes': summary.get('n_vars', bp.state.current_adaptor.n_vars),
            'embeddings': summary.get('embeddings', []),
            'obs_columns': list(bp.state.current_adaptor.adata.obs.columns),
            'var_columns': list(bp.state.current_adaptor.adata.var.columns),
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


@bp.route('/schema', methods=['GET'])
def get_schema():
    """Get data schema."""
    if bp.state.current_adaptor is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        schema = bp.state.current_adaptor.get_schema()
        return jsonify({'schema': schema})
    except Exception as e:
        logging.error(f"Schema retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/data/obs', methods=['GET'])
def get_obs_data():
    """Get observation annotations as FlatBuffers."""
    if bp.state.current_adaptor is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        columns = request.args.getlist('columns')
        chunk_index = request.args.get('chunk', 0, type=int)

        if chunk_index > 0:
            # Chunked data request
            fbs_data = bp.state.current_adaptor.get_chunked_data('obs', chunk_index, columns=columns if columns else None)
        else:
            # Full data request
            fbs_data = bp.state.current_adaptor.get_obs_fbs(columns if columns else None)

        response = make_response(fbs_data)
        response.headers['Content-Type'] = 'application/octet-stream'
        response.headers['Cache-Control'] = 'public, max-age=3600'
        return response

    except Exception as e:
        logging.error(f"Obs data retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/data/embedding/<embedding_name>', methods=['GET'])
def get_embedding_data(embedding_name):
    """Get embedding coordinates as FlatBuffers."""
    if bp.state.current_adaptor is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        chunk_index = request.args.get('chunk', 0, type=int)
        out_format = request.args.get('format', 'fbs')

        if chunk_index > 0:
            if out_format == 'json':
                # JSON slice response
                if embedding_name == 'random' or embedding_name == 'X_random':
                    coords = bp.state.current_adaptor._get_random_embedding()
                else:
                    embedding_key = f'X_{embedding_name}' if not embedding_name.startswith('X_') else embedding_name
                    coords = bp.state.current_adaptor.adata.obsm[embedding_key]
                start = chunk_index * bp.state.current_adaptor.chunk_size
                end = min(start + bp.state.current_adaptor.chunk_size, bp.state.current_adaptor.n_obs)
                return jsonify({'x': coords[start:end, 0].tolist(), 'y': coords[start:end, 1].tolist()})
            else:
                # FBS slice
                fbs_data = bp.state.current_adaptor.get_chunked_data('embedding', chunk_index, embedding_name=embedding_name)
                response = make_response(fbs_data)
                response.headers['Content-Type'] = 'application/octet-stream'
                response.headers['Cache-Control'] = 'public, max-age=3600'
                return response
        else:
            if out_format == 'json':
                if embedding_name == 'random' or embedding_name == 'X_random':
                    coords = bp.state.current_adaptor._get_random_embedding()
                else:
                    embedding_key = f'X_{embedding_name}' if not embedding_name.startswith('X_') else embedding_name
                    coords = bp.state.current_adaptor.adata.obsm[embedding_key]
                return jsonify({'x': coords[:, 0].tolist(), 'y': coords[:, 1].tolist()})
            else:
                fbs_data = bp.state.current_adaptor.get_embedding_fbs(embedding_name)
                response = make_response(fbs_data)
                response.headers['Content-Type'] = 'application/octet-stream'
                response.headers['Cache-Control'] = 'public, max-age=3600'
                return response

    except KeyError as e:
        return jsonify({'error': f'Embedding not found: {embedding_name}'}), 404
    except Exception as e:
        logging.error(f"Embedding data retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/data/expression', methods=['POST'])
def get_expression_data():
    """Get gene expression data as FlatBuffers."""
    if bp.state.current_adaptor is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        data = request.json
        gene_names = data.get('genes', [])
        cell_indices = data.get('cell_indices', None)

        if not gene_names:
            return jsonify({'error': 'No genes specified'}), 400

        fbs_data = bp.state.current_adaptor.get_expression_fbs(gene_names, cell_indices)

        response = make_response(fbs_data)
        response.headers['Content-Type'] = 'application/octet-stream'
        return response

    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logging.error(f"Expression data retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/save', methods=['POST'])
def save_data():
    """Save processed data as h5ad file."""
    if bp.state.current_adata is None:
        return jsonify({'error': 'No data to save'}), 400

    try:
        # Save to temporary file
        output_path = os.path.join(bp.upload_folder, f'processed_{bp.state.current_filename}')
        bp.state.current_adata.write_h5ad(output_path)

        return send_file(output_path, as_attachment=True,
                        download_name=f'processed_{bp.state.current_filename}')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/data_info', methods=['GET'])
def get_data_info():
    """Get detailed information about the loaded data."""
    if bp.state.current_adata is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        info = {
            'shape': list(bp.state.current_adata.shape),
            'obs_keys': list(bp.state.current_adata.obs.keys()),
            'var_keys': list(bp.state.current_adata.var.keys()),
            'obsm_keys': list(bp.state.current_adata.obsm.keys()) if bp.state.current_adata.obsm else [],
            'uns_keys': list(bp.state.current_adata.uns.keys()) if bp.state.current_adata.uns else [],
            'layers': list(bp.state.current_adata.layers.keys()) if bp.state.current_adata.layers else []
        }

        # Add some statistics
        if hasattr(bp.state.current_adata.X, 'toarray'):
            X = bp.state.current_adata.X.toarray()
        else:
            X = bp.state.current_adata.X

        info['statistics'] = {
            'mean_genes_per_cell': float(np.mean(np.sum(X > 0, axis=1))),
            'mean_counts_per_cell': float(np.mean(np.sum(X, axis=1))),
            'total_counts': float(np.sum(X))
        }

        return jsonify(info)

    except Exception as e:
        logging.error(f"Data info retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/data_status', methods=['GET'])
def get_data_status():
    """Get current data status (basic). Use /api/status for full UI-compatible response."""
    return jsonify({
        'loaded': bp.state.current_adata is not None,
        'filename': bp.state.current_filename,
        'cells': bp.state.current_adata.n_obs if bp.state.current_adata else 0,
        'genes': bp.state.current_adata.n_vars if bp.state.current_adata else 0,
    })


@bp.route('/genes', methods=['GET'])
def get_genes():
    """Get all gene names."""
    if bp.state.current_adata is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        genes = list(bp.state.current_adata.var_names)
        return jsonify({'genes': genes})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/gene_search', methods=['POST'])
def search_genes():
    """Search for genes by name pattern."""
    if bp.state.current_adata is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        data = request.json
        pattern = data.get('pattern', '').upper()

        if not pattern:
            return jsonify({'genes': []})

        # Search genes that match the pattern
        matching_genes = [gene for gene in bp.state.current_adata.var_names if pattern in gene.upper()]

        # Limit results to avoid too many matches
        matching_genes = matching_genes[:20]

        return jsonify({'genes': matching_genes})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/qc_prefixes', methods=['GET'])
def get_qc_prefixes():
    """Detect common QC gene prefixes (e.g., mitochondrial) from var_names."""
    if bp.state.current_adata is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        import re
        var_names = bp.state.current_adata.var_names.astype(str)
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


@bp.route('/filter', methods=['POST'])
def filter_cells():
    """Filter cells based on criteria."""
    if bp.state.current_adaptor is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        filters = request.json
        indices = bp.state.current_adaptor.filter_cells(filters)

        return jsonify({
            'filtered_indices': indices,
            'count': len(indices)
        })

    except Exception as e:
        logging.error(f"Cell filtering failed: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/differential_expression', methods=['POST'])
def compute_differential_expression():
    """Compute differential expression between groups."""
    if bp.state.current_adaptor is None:
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
        future = bp.state.thread_pool.submit(
            bp.state.current_adaptor.get_differential_expression,
            group1_indices, group2_indices, method, n_genes
        )

        result = future.result(timeout=30)  # 30 second timeout

        return jsonify(result)

    except Exception as e:
        logging.error(f"Differential expression failed: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/export_plot_data', methods=['POST'])
def export_plot_data():
    """Export current plot data as CSV."""
    if bp.state.current_adata is None:
        return jsonify({'error': 'No data loaded'}), 400

    try:
        data = request.json
        embedding = data.get('embedding', '')
        color_by = data.get('color_by', '')

        # Create CSV with embedding coordinates and color data
        import pandas as pd
        import io

        # Get embedding coordinates
        embedding_key = f'X_{embedding}' if not embedding.startswith('X_') else embedding
        if embedding_key not in bp.state.current_adata.obsm:
            return jsonify({'error': f'Embedding {embedding} not found'}), 404

        coords = bp.state.current_adata.obsm[embedding_key]
        df = pd.DataFrame({
            f'{embedding}_1': coords[:, 0],
            f'{embedding}_2': coords[:, 1]
        })

        # Add color data if specified
        if color_by and color_by in bp.state.current_adata.obs.columns:
            df[color_by] = bp.state.current_adata.obs[color_by].values

        # Convert to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=True)
        csv_buffer.seek(0)

        from flask import Response
        return Response(
            csv_buffer.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename=plot_data_{embedding}.csv'}
        )

    except Exception as e:
        logging.error(f"Export plot data failed: {e}")
        return jsonify({'error': str(e)}), 500


# Initialize blueprint with dependencies (will be set by app.py)
bp.state = None
bp.upload_folder = None

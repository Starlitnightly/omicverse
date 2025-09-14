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
                        plot_data['colors'] = values.tolist()
                        plot_data['colorscale'] = 'Viridis'
                        plot_data['color_label'] = col_name
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
                        discrete_colors = get_discrete_colors(n_categories)
                        plot_data['colors'] = [discrete_colors[code] for code in categories.cat.codes]
                        plot_data['discrete_colors'] = discrete_colors

            elif color_by.startswith('gene:'):
                gene_name = color_by.replace('gene:', '')
                try:
                    expr_fbs = current_adaptor.get_expression_fbs([gene_name])
                    expr_df = decode_matrix_fbs(expr_fbs)

                    if gene_name in expr_df.columns:
                        expression = expr_df[gene_name]
                        plot_data['colors'] = expression.tolist()
                        plot_data['colorscale'] = 'Viridis'
                        plot_data['color_label'] = f'{gene_name} expression'
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
            # Quality control: filter cells by gene counts/UMI/mt% if available
            min_genes = params.get('min_genes', None)
            min_counts = params.get('min_counts', None)
            max_mt_percent = params.get('max_mt_percent', None)
            if min_genes is not None:
                sc.pp.filter_cells(current_adata, min_genes=int(min_genes))
            if min_counts is not None and min_counts > 0:
                sc.pp.filter_cells(current_adata, min_counts=int(min_counts))
            # Remove cells by mitochondrial percent if present
            if max_mt_percent is not None:
                mt_col = None
                for c in current_adata.obs.columns:
                    lc = c.lower()
                    if 'mt' in lc and '%' in lc or 'pct_counts_mt' in lc or 'percent_mt' in lc:
                        mt_col = c
                        break
                if mt_col is not None:
                    keep = current_adata.obs[mt_col] <= float(max_mt_percent)
                    current_adata._inplace_subset_obs(keep.values)

        elif tool == 'filter_genes':
            min_cells = params.get('min_cells', None)
            if min_cells is not None:
                sc.pp.filter_genes(current_adata, min_cells=int(min_cells))
            # Optional min_mean filter
            min_mean = params.get('min_mean', None)
            if min_mean is not None and hasattr(current_adata, 'X'):
                import numpy as _np
                X = current_adata.X.toarray() if hasattr(current_adata.X, 'toarray') else current_adata.X
                gene_means = _np.asarray(X).mean(axis=0).A1 if hasattr(_np.asarray(X), 'A1') else _np.asarray(X).mean(axis=0)
                keep = gene_means >= float(min_mean)
                current_adata._inplace_subset_var(keep)

        elif tool == 'doublets':
            return jsonify({'error': 'Doublet removal not implemented yet'}), 400

        else:
            return jsonify({'error': f'Unknown tool: {tool}'}), 400

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
        return jsonify({'error': str(e)}), 500

def get_discrete_colors(n_categories):
    """获取离散分类颜色"""
    # 预定义的颜色调色板
    color_palettes = {
        'set1': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999'],
        'set2': ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'],
        'set3': ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd'],
        'paired': ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'],
        'plotly3': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    }
    
    # 根据类别数量选择合适的调色板
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

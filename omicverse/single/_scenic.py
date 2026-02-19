import numpy as np
import anndata as ad
import pandas as pd


class SCENIC:

    def __init__(
            self,
            adata,
            db_glob,
            motif_path,
            n_jobs=8):
        
        from .._settings import Colors, EMOJI, print_gpu_usage_color
        from ..external.ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
        self.adata = adata
        self.db_glob = db_glob
        self.motif_path = motif_path
        self.n_jobs = n_jobs

        import glob
        import os

        # Print SCENIC initialization information with colors
        print(f"{Colors.HEADER}{Colors.BOLD}{EMOJI['start']} SCENIC Analysis Initialization:{Colors.ENDC}")
        print(f"   {Colors.CYAN}Input data shape: {Colors.BOLD}{adata.shape[0]} cells × {adata.shape[1]} genes{Colors.ENDC}")
        
        # Check data type and quality
        total_counts = adata.X.sum()
        mean_genes_per_cell = (adata.X > 0).sum(axis=1).mean()
        print(f"   {Colors.BLUE}Total UMI counts: {Colors.BOLD}{total_counts:,.0f}{Colors.ENDC}")
        print(f"   {Colors.BLUE}Mean genes per cell: {Colors.BOLD}{mean_genes_per_cell:.1f}{Colors.ENDC}")
        
        # Database information
        db_fnames = glob.glob(db_glob)  # e.g. your feather files
        print(f"   {Colors.GREEN}Ranking databases found: {Colors.BOLD}{len(db_fnames)}{Colors.ENDC}")
        
        if len(db_fnames) == 0:
            print(f"   {Colors.FAIL}{EMOJI['error']} No ranking databases found at: {Colors.BOLD}{db_glob}{Colors.ENDC}")
            raise ValueError(f"No ranking databases found at: {db_glob}")
        
        # Show database files
        for i, fname in enumerate(db_fnames[:5]):  # Show first 5
            basename = os.path.basename(fname)
            print(f"   {Colors.BLUE}  └─ [{i+1}] {Colors.BOLD}{basename}{Colors.ENDC}")
        if len(db_fnames) > 5:
            print(f"   {Colors.BLUE}  └─ ... and {Colors.BOLD}{len(db_fnames)-5}{Colors.ENDC}{Colors.BLUE} more{Colors.ENDC}")
        
        # Motif annotations
        print(f"   {Colors.GREEN}Motif annotations: {Colors.BOLD}{motif_path}{Colors.ENDC}")
        if not os.path.exists(motif_path):
            print(f"   {Colors.FAIL}{EMOJI['error']} Motif annotation file not found!{Colors.ENDC}")
            raise FileNotFoundError(f"Motif annotation file not found: {motif_path}")
        
        # Computational settings
        print(f"\n{Colors.HEADER}{Colors.BOLD}⚙️  Computational Settings:{Colors.ENDC}")
        print(f"   {Colors.BLUE}Number of workers: {Colors.BOLD}{n_jobs}{Colors.ENDC}")
        
        # GPU usage information
        print(f"\n{Colors.HEADER}{Colors.BOLD}{EMOJI['gpu']} GPU Usage Information:{Colors.ENDC}")
        try:
            print_gpu_usage_color(bar_length=25)
        except Exception as e:
            print(f"   {Colors.WARNING}Could not get GPU info: {e}{Colors.ENDC}")
        
        # Performance recommendations
        print(f"\n{Colors.HEADER}{Colors.BOLD}💡 Performance Recommendations:{Colors.ENDC}")
        if adata.shape[0] > 50000:
            print(f"   {Colors.WARNING}▶ Large dataset detected ({adata.shape[0]} cells):{Colors.ENDC}")
            print(f"     {Colors.CYAN}Consider increasing n_jobs to {Colors.BOLD}{min(n_jobs*2, 16)}{Colors.ENDC}{Colors.CYAN} for faster processing{Colors.ENDC}")
        
        if adata.shape[1] > 30000:
            print(f"   {Colors.WARNING}▶ Many genes detected ({adata.shape[1]} genes):{Colors.ENDC}")
            print(f"     {Colors.CYAN}Consider filtering low-expression genes before SCENIC analysis{Colors.ENDC}")
        
        if len(db_fnames) > 10:
            print(f"   {Colors.BLUE}▶ Multiple databases available ({len(db_fnames)}):{Colors.ENDC}")
            print(f"     {Colors.GREEN}Great! More databases can improve regulon discovery{Colors.ENDC}")
        
        dbs = [ RankingDatabase(fname, name=os.path.splitext(os.path.basename(fname))[0])
        for fname in db_fnames ]

        self.dbs = dbs
        
        print(f"\n{Colors.GREEN}{EMOJI['done']} SCENIC initialization completed successfully!{Colors.ENDC}")
        print(f"{Colors.CYAN}{'─' * 60}{Colors.ENDC}")


    def cal_grn(
        self,method='regdiffusion',layer='counts',tf_names=None,**kwargs):
        r"""
        Initialize and Train a RegDiffusion model.

        For architecture and training details, please refer to our paper.

        > From noise to knowledge: probabilistic diffusion-based neural inference

        You can access the model through `RegDiffusionTrainer.model`. 
        
        Arguments:
            exp_array (np.ndarray): 2D numpy array. If used on single-cell RNAseq, 
                the rows are cells and the columns are genes. Data should be log 
                transformed. You may also want to remove all non expressed genes.
            cell_types (np.ndarray): (Optional) 1D integer array for cell type. If 
                you have labels in your cell type, you need to convert them to 
                interge. Default is None. 
            T (int): Total number of diffusion steps. Default: 5,000
            start_noise (float): Minimal noise level (beta) to be added. Default: 
                0.0001
            end_noise (float): Maximal noise level (beta) to be added. Default: 
                0.02
            time_dim (int): Dimension size for the time embedding. Default: 64.
            celltype_dim (int): Dimension size for the cell type embedding. 
                Default: 4.
            hidden_dim (list): Dimension sizes for the feature learning layers. We 
                use the size of the first layer as the dimension for gene embeddings
                as well. Default: [16, 16, 16].
            init_coef (int): A coefficent to control the value to initialize the 
                adjacency matrix. Here we define regulatory norm as 1 over (number 
                of genes - 1). The value which we use to initialize the model is 
                `init_coef` times of the regulatory norm. Default: 5. 
            lr_nn (float): Learning rate for the rest of the neural networks except
                the adjacency matrix. Default: 0.001
            lr_adj (float): Learning rate for the adjacency matrix. By default, it 
                equals to 0.02 * gene regulatory norm, which equals 1/(n_gene-1). 
            weight_decay_nn (float): L2 regularization coef on the rest of the 
                neural networks. Default: 0.1.
            weight_decay_adj (float): L2 regularization coef on the adj matrix.
                Default: 0.01.
            sparse_loss_coef (float): L1 regularization coef on the adj matrix. 
                Default: 0.25.
            adj_dropout (float): Probability of an edge to be zeroed. Default: 0.3.
            batch_size (int): Batch size for training. Default: 128.
            n_steps (int): Total number of training iterations. Default: 1000.
            train_split (float): Train partition. Default: 1.0.
            train_split_seed (int): Random seed for train/val partition. 
                Default: 123
            device (str or torch.device): Device where the model is running. For 
                example, "cpu", "cuda", "cuda:1", and etc. You are not recommended 
                to run this model on Apple's MPS chips. Default is "cuda" but if 
                you only has CPU, it will switch back to CPU.
            compile (boolean): Whether to compile the model before training. 
                Compile the model is a good idea on large dataset and ofter improves
                inference speed when it works. For smaller dataset, eager execution 
                if often good enough. 
            evaluator (GRNEvaluator): (Optional) A defined GRNEvaluator if ground 
                truth data is available. Evaluation will be done every 100 steps by 
                default but you can change this setting through the eval_on_n_steps 
                option. Default is None
            eval_on_n_steps (int): If an evaluator is provided, the trainer will 
                run evaluation every `eval_on_n_steps` steps. Default: 100.
            logger (LightLogger): (Optional) A LightLogger to log training process. 
                The only situation when you need to provide this is when you want 
                to save logs from different trainers into the same logger. Default 
                is None. 
        """
        from .._settings import Colors, EMOJI, print_gpu_usage_color
        
        
        print(f"\n{Colors.HEADER}{Colors.BOLD}🧬 Gene Regulatory Network (GRN) Inference:{Colors.ENDC}")
        print(f"   {Colors.CYAN}Method: {Colors.BOLD}{method}{Colors.ENDC}")
        print(f"   {Colors.BLUE}Data layer: {Colors.BOLD}'{layer}'{Colors.ENDC}")
        
        # Data preparation
        if layer in self.adata.layers:
            if self.adata.layers[layer].max()<np.log1p(1e4):
                raise ValueError(f"Layer '{layer}' has been log normalized, please use a layer with raw counts data")
            x = self.adata.layers[layer].toarray()
            x = np.log(x+1.0)
            print(f"   {Colors.GREEN}✓ Using existing '{layer}' layer{Colors.ENDC}")
        else:
            print(f"   {Colors.WARNING}⚠️  Layer '{layer}' not found, recovering counts...{Colors.ENDC}")
            raise ValueError(f"Layer '{layer}' not found")

        

        # Display data statistics
        print(f"\n{Colors.HEADER}{Colors.BOLD}📊 Data Statistics:{Colors.ENDC}")
        print(f"   {Colors.BLUE}Expression matrix shape: {Colors.BOLD}{x.shape}{Colors.ENDC}")
        print(f"   {Colors.BLUE}Mean expression: {Colors.BOLD}{x.mean():.3f}{Colors.ENDC}")
        print(f"   {Colors.BLUE}Sparsity: {Colors.BOLD}{(x == 0).sum() / x.size * 100:.1f}%{Colors.ENDC}")
        
        # Display training parameters
        if method=='regdiffusion':
            import regdiffusion as rd
            print(f"\n{Colors.HEADER}{Colors.BOLD}⚙️  Training Parameters:{Colors.ENDC}")
            default_params = {
                'T': 5000, 'start_noise': 0.0001, 'end_noise': 0.02,
                'time_dim': 64, 'celltype_dim': 4, 'hidden_dim': [16, 16, 16],
                'init_coef': 5, 'lr_nn': 0.001, 'lr_adj': None,
                'weight_decay_nn': 0.1, 'weight_decay_adj': 0.01,
                'sparse_loss_coef': 0.25, 'adj_dropout': 0.3,
                'batch_size': 128, 'n_steps': 1000, 'train_split': 1.0,
                'device': 'cuda'
            }
            
            # Show key parameters
            key_params = ['n_steps', 'batch_size', 'device', 'lr_nn', 'sparse_loss_coef']
            for param in key_params:
                value = kwargs.get(param, default_params.get(param, 'Default'))
                if param == 'device':
                    color = Colors.GREEN if 'cuda' in str(value) else Colors.BLUE
                else:
                    color = Colors.BLUE
                print(f"   {color}{param}: {Colors.BOLD}{value}{Colors.ENDC}")
            
            # GPU status check
            if kwargs.get('device', 'cuda') != 'cpu':
                print(f"\n{Colors.HEADER}{Colors.BOLD}{EMOJI['gpu']} GPU Training Status:{Colors.ENDC}")
                try:
                    print_gpu_usage_color(bar_length=20)
                except:
                    print(f"   {Colors.WARNING}GPU status check failed{Colors.ENDC}")
            
            # Training time estimation
            estimated_time = x.shape[0] * kwargs.get('n_steps', 1000) / 10000  # rough estimate
            if estimated_time > 60:
                print(f"\n{Colors.HEADER}{Colors.BOLD}⏱️  Estimated Training Time:{Colors.ENDC}")
                print(f"   {Colors.CYAN}Approximate: {Colors.BOLD}{estimated_time/60:.1f} minutes{Colors.ENDC}")
                if estimated_time > 300:  # 5 minutes
                    print(f"   {Colors.WARNING}💡 This may take a while. Consider reducing n_steps or using GPU{Colors.ENDC}")

            print(f"\n{Colors.GREEN}{EMOJI['start']} Starting RegDiffusion training...{Colors.ENDC}")
            print(f"{Colors.CYAN}{'─' * 60}{Colors.ENDC}")

            rd_trainer = rd.RegDiffusionTrainer(x,**kwargs)
            rd_trainer.train()
            grn = rd_trainer.get_grn(self.adata.var_names, top_gene_percentile=50)

            # Here for each gene, we are going to extract all edges
            edgelist = grn.extract_edgelist(k=-1, workers=self.n_jobs)
            edgelist.columns = ['TF', 'target', 'importance']
            self.edgelist = edgelist
            
            self.edgelist['importance']=self.edgelist['importance'].astype(np.float32)
            self.adjacencies = self.edgelist
        elif method=='grnboost2': 
            from ..external.single.arboreto.algo import grnboost2
            edgelist = grnboost2(expression_data=x,
                    tf_names=tf_names)
            self.edgelist = edgelist
            self.edgelist['importance']=self.edgelist['importance'].astype(np.float32)
            self.adjacencies = self.edgelist
        elif method=='genie3':
            from ..external.single.arboreto.algo import genie3
            edgelist = genie3(expression_data=x,
                    tf_names=tf_names)
            self.edgelist = edgelist
            self.edgelist['importance']=self.edgelist['importance'].astype(np.float32)
            self.adjacencies = self.edgelist
        else:
            raise ValueError(f"Method '{method}' not supported")


        
        # Display results
        print(f"\n{Colors.GREEN}{EMOJI['done']} GRN inference completed!{Colors.ENDC}")
        print(f"   {Colors.BLUE}Total edges detected: {Colors.BOLD}{len(self.edgelist):,}{Colors.ENDC}")
        print(f"   {Colors.BLUE}Unique TFs: {Colors.BOLD}{self.edgelist['TF'].nunique()}{Colors.ENDC}")
        print(f"   {Colors.BLUE}Unique targets: {Colors.BOLD}{self.edgelist['target'].nunique()}{Colors.ENDC}")
        print(f"   {Colors.BLUE}Mean importance: {Colors.BOLD}{self.edgelist['importance'].mean():.4f}{Colors.ENDC}")
        
        return edgelist
    
    def cal_regulons(
            self,
            rho_mask_dropouts=True,
            seed=42,**kwargs
        ):
        from .._settings import Colors, EMOJI, print_gpu_usage_color
        
        from ..external.pyscenic.utils import modules_from_adjacencies
        from ..external.pyscenic.prune import prune2df, df2regulons

        print(f"\n{Colors.HEADER}{Colors.BOLD}🎯 Regulon Calculation and Activity Scoring:{Colors.ENDC}")
        print(f"   {Colors.CYAN}Input edges: {Colors.BOLD}{len(self.edgelist):,}{Colors.ENDC}")
        print(f"   {Colors.BLUE}Databases: {Colors.BOLD}{len(self.dbs)}{Colors.ENDC}")
        print(f"   {Colors.BLUE}Workers: {Colors.BOLD}{self.n_jobs}{Colors.ENDC}")

        expr_mtx=self.adata.to_df()
        
        print(f"\n{Colors.HEADER}{Colors.BOLD}📊 Expression Matrix Info:{Colors.ENDC}")
        print(f"   {Colors.BLUE}Shape: {Colors.BOLD}{expr_mtx.shape}{Colors.ENDC}")
        print(f"   {Colors.BLUE}Missing values: {Colors.BOLD}{expr_mtx.isnull().sum().sum()}{Colors.ENDC}")
        
        # Parameters display
        print(f"\n{Colors.HEADER}{Colors.BOLD}⚙️  Regulon Parameters:{Colors.ENDC}")
        print(f"   {Colors.BLUE}rho_mask_dropouts: {Colors.BOLD}{rho_mask_dropouts}{Colors.ENDC}")
        print(f"   {Colors.BLUE}random_seed: {Colors.BOLD}{seed}{Colors.ENDC}")
        
        # Show additional kwargs if provided
        if kwargs:
            print(f"   {Colors.BLUE}Additional parameters:{Colors.ENDC}")
            for key, value in kwargs.items():
                print(f"   {Colors.BLUE}  {key}: {Colors.BOLD}{value}{Colors.ENDC}")

        print(f"\n{Colors.GREEN}{EMOJI['start']} Step 1: Building co-expression modules...{Colors.ENDC}")
        
        modules = list(
            modules_from_adjacencies(
                self.edgelist, 
                expr_mtx,
                rho_mask_dropouts=rho_mask_dropouts,
                **kwargs
            )
        )
        self.modules = modules
        
        print(f"   {Colors.GREEN}✓ Modules created: {Colors.BOLD}{len(modules)}{Colors.ENDC}")
        if len(modules) > 0:
            module_sizes = [len(module) for module in modules]
            print(f"   {Colors.BLUE}  Mean module size: {Colors.BOLD}{np.mean(module_sizes):.1f} genes{Colors.ENDC}")
            print(f"   {Colors.BLUE}  Module size range: {Colors.BOLD}{min(module_sizes)} - {max(module_sizes)} genes{Colors.ENDC}")

        print(f"\n{Colors.GREEN}{EMOJI['start']} Step 2: Pruning modules with cisTarget databases...{Colors.ENDC}")
        
        # Calculate enriched motifs and build regulons
        df = prune2df(self.dbs, modules, self.motif_path,num_workers=self.n_jobs,
                        client_or_address='custom_multiprocessing')
        regulons = df2regulons(df)
        self.regulons = regulons
        
        print(f"   {Colors.GREEN}✓ Regulons created: {Colors.BOLD}{len(regulons)}{Colors.ENDC}")
        if len(regulons) > 0:
            regulon_sizes = [len(regulon.gene2weight) for regulon in regulons]
            print(f"   {Colors.BLUE}  Mean regulon size: {Colors.BOLD}{np.mean(regulon_sizes):.1f} genes{Colors.ENDC}")
            print(f"   {Colors.BLUE}  Regulon size range: {Colors.BOLD}{min(regulon_sizes)} - {max(regulon_sizes)} genes{Colors.ENDC}")

        print(f"\n{Colors.GREEN}{EMOJI['start']} Step 3: Calculating AUCell scores...{Colors.ENDC}")
        
        from ..single._aucell import aucell

        auc_mtx = aucell(
            expr_mtx,
            regulons,
            num_workers=self.n_jobs,   # parallelism
            seed=seed                    # for reproducibility
        )
        self.auc_mtx = auc_mtx

        import anndata as ad

        ad_auc_mtx = ad.AnnData(auc_mtx)
        ad_auc_mtx.obs=self.adata.obs.loc[ad_auc_mtx.obs.index]
        self.ad_auc_mtx=ad_auc_mtx
        
        # Final results summary
        print(f"\n{Colors.GREEN}{EMOJI['done']} Regulon analysis completed successfully!{Colors.ENDC}")
        print(f"\n{Colors.HEADER}{Colors.BOLD}📈 Final Results Summary:{Colors.ENDC}")
        print(f"   {Colors.GREEN}✓ Input modules: {Colors.BOLD}{len(modules)}{Colors.ENDC}")
        print(f"   {Colors.GREEN}✓ Final regulons: {Colors.BOLD}{len(regulons)}{Colors.ENDC}")
        print(f"   {Colors.GREEN}✓ AUC matrix shape: {Colors.BOLD}{auc_mtx.shape}{Colors.ENDC}")
        print(f"   {Colors.BLUE}Mean AUC value: {Colors.BOLD}{auc_mtx.values.mean():.4f}{Colors.ENDC}")
        print(f"   {Colors.BLUE}AUC range: {Colors.BOLD}{auc_mtx.values.min():.4f} - {auc_mtx.values.max():.4f}{Colors.ENDC}")
        
        # Success rate
        success_rate = len(regulons) / len(modules) * 100 if len(modules) > 0 else 0
        if success_rate >= 50:
            color = Colors.GREEN
        elif success_rate >= 25:
            color = Colors.WARNING
        else:
            color = Colors.FAIL
        print(f"   {color}Module→Regulon success rate: {Colors.BOLD}{success_rate:.1f}%{Colors.ENDC}")
        
        # Recommendations
        print(f"\n{Colors.HEADER}{Colors.BOLD}💡 Next Steps Recommendations:{Colors.ENDC}")
        if len(regulons) > 0:
            print(f"   {Colors.GREEN}✓ Analysis successful! You can now:{Colors.ENDC}")
            print(f"     {Colors.CYAN}• Use scenic.ad_auc_mtx for downstream analysis{Colors.ENDC}")
            print(f"     {Colors.CYAN}• Visualize regulon activity with: sc.pl.heatmap(scenic.ad_auc_mtx, ...){Colors.ENDC}")
            print(f"     {Colors.CYAN}• Calculate regulon specificity scores{Colors.ENDC}")
        else:
            print(f"   {Colors.WARNING}⚠️  No regulons detected. Consider:{Colors.ENDC}")
            print(f"     {Colors.CYAN}• Relaxing thresholds in modules_from_adjacencies{Colors.ENDC}")
            print(f"     {Colors.CYAN}• Checking motif annotation compatibility{Colors.ENDC}")
            print(f"     {Colors.CYAN}• Using different ranking databases{Colors.ENDC}")
        
        print(f"\n{Colors.CYAN}{'─' * 60}{Colors.ENDC}")

        return ad_auc_mtx

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler

def calculate_gene_temporal_center(rna_data, cell_indices, gene_name, 
                                 pseudotime_col='palantir_pseudotime', 
                                 method='weighted_mean'):
    """
    Calculate the temporal center position of a gene
    
    Parameters:
    -----------
    rna_data : AnnData object or similar structure
        Data object containing gene expression and cell metadata
    cell_indices : array-like
        Cell indices to analyze
    gene_name : str
        Gene name
    pseudotime_col : str
        Pseudotime column name
    method : str
        Calculation method: 'weighted_mean', 'peak', 'median'
    
    Returns:
    --------
    dict : Contains temporal center position and related statistical information
    """
    
    # Get gene expression data
    gene_expression = rna_data[cell_indices, gene_name].to_df()
    if isinstance(gene_expression, pd.DataFrame):
        gene_expression = gene_expression.iloc[:, 0]  # If DataFrame, take first column
    
    # Get corresponding pseudotime
    pseudotime = rna_data.obs.loc[cell_indices, pseudotime_col]
    
    # Ensure data alignment
    common_indices = gene_expression.index.intersection(pseudotime.index)
    gene_expr = gene_expression.loc[common_indices]
    pseudo_time = pseudotime.loc[common_indices]
    
    # Remove missing values
    valid_mask = ~(np.isnan(gene_expr) | np.isnan(pseudo_time))
    gene_expr = gene_expr[valid_mask]
    pseudo_time = pseudo_time[valid_mask]
    
    results = {}
    
    if method == 'weighted_mean':
        # Method 1: Expression level weighted average pseudotime
        weights = gene_expr + 1e-6  # Avoid division by zero
        temporal_center = np.average(pseudo_time, weights=weights)
        
    elif method == 'peak':
        # Method 2: Find pseudotime corresponding to expression peak
        # Use sliding window smoothing
        df = pd.DataFrame({'pseudotime': pseudo_time, 'expression': gene_expr})
        df_sorted = df.sort_values('pseudotime')
        
        # Calculate moving average
        window_size = max(10, len(df_sorted) // 20)
        df_sorted['expr_smooth'] = df_sorted['expression'].rolling(
            window=window_size, center=True).mean()
        
        # Find peak
        peak_idx = df_sorted['expr_smooth'].idxmax()
        temporal_center = df_sorted.loc[peak_idx, 'pseudotime']
        
    elif method == 'median':
        # Method 3: Pseudotime corresponding to expression median
        expr_threshold = np.median(gene_expr[gene_expr > 0])
        high_expr_cells = gene_expr >= expr_threshold
        temporal_center = np.median(pseudo_time[high_expr_cells])
    
    # Calculate additional statistical information
    results['temporal_center'] = temporal_center
    results['expression_range'] = (gene_expr.min(), gene_expr.max())
    results['pseudotime_range'] = (pseudo_time.min(), pseudo_time.max())
    results['correlation'] = stats.pearsonr(pseudo_time, gene_expr)[0]
    results['n_cells'] = len(gene_expr)
    
    # Calculate temporal distribution width of expression
    expr_weights = gene_expr / gene_expr.sum()
    variance = np.average((pseudo_time - temporal_center)**2, weights=expr_weights)
    results['temporal_width'] = np.sqrt(variance)
    
    return results

def plot_gene_temporal_pattern(rna_data, cell_indices, gene_name, 
                             pseudotime_col='palantir_pseudotime',
                             temporal_center=None):
    """
    Plot gene temporal expression pattern
    """
    # Get data
    gene_expression = rna_data[cell_indices, gene_name].to_df()
    if isinstance(gene_expression, pd.DataFrame):
        gene_expression = gene_expression.iloc[:, 0]
    
    pseudotime = rna_data.obs.loc[cell_indices, pseudotime_col]
    
    # Data alignment and cleaning
    common_indices = gene_expression.index.intersection(pseudotime.index)
    gene_expr = gene_expression.loc[common_indices]
    pseudo_time = pseudotime.loc[common_indices]
    
    valid_mask = ~(np.isnan(gene_expr) | np.isnan(pseudo_time))
    gene_expr = gene_expr[valid_mask]
    pseudo_time = pseudo_time[valid_mask]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Scatter plot
    ax1.scatter(pseudo_time, gene_expr, alpha=0.6, s=20)
    ax1.set_xlabel('Pseudotime')
    ax1.set_ylabel(f'{gene_name} Expression')
    ax1.set_title(f'{gene_name} Expression along Pseudotime')
    
    # Add temporal center line
    if temporal_center is not None:
        ax1.axvline(x=temporal_center, color='red', linestyle='--', 
                   label=f'Temporal Center: {temporal_center:.3f}')
        ax1.legend()
    
    # Smooth curve
    df = pd.DataFrame({'pseudotime': pseudo_time, 'expression': gene_expr})
    df_sorted = df.sort_values('pseudotime')
    window_size = max(10, len(df_sorted) // 20)
    df_sorted['expr_smooth'] = df_sorted['expression'].rolling(
        window=window_size, center=True).mean()
    
    ax1.plot(df_sorted['pseudotime'], df_sorted['expr_smooth'], 
            color='orange', linewidth=2, label='Smoothed')
    ax1.legend()
    
    # Expression distribution histogram
    ax2.hist(gene_expr, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel(f'{gene_name} Expression')
    ax2.set_ylabel('Cell Count')
    ax2.set_title(f'{gene_name} Expression Distribution')
    
    plt.tight_layout()
    return fig

# Usage example
def analyze_gene_temporal_center(rna_margin, axial_cell_idx, gene_name='gata5'):
    """
    Complete gene temporal analysis workflow
    """
    print(f"Analyzing temporal center position of gene {gene_name}...")
    
    # Calculate temporal center using different methods
    methods = ['weighted_mean', 'peak', 'median']
    results = {}
    
    for method in methods:
        try:
            result = calculate_gene_temporal_center(
                rna_margin, axial_cell_idx, gene_name, method=method
            )
            results[method] = result
            print(f"\n{method} method:")
            print(f"  Temporal center position: {result['temporal_center']:.4f}")
            print(f"  Temporal distribution width: {result['temporal_width']:.4f}")
            print(f"  Expression-time correlation: {result['correlation']:.4f}")
            print(f"  Number of cells: {result['n_cells']}")
            
        except Exception as e:
            print(f"{method} method calculation failed: {e}")
    
    # Plot figure
    if 'weighted_mean' in results:
        fig = plot_gene_temporal_pattern(
            rna_margin, axial_cell_idx, gene_name,
            temporal_center=results['weighted_mean']['temporal_center']
        )
        plt.show()
    
    return results

def batch_calculate_gene_temporal_centers(rna_data, cell_indices, gene_list, 
                                        pseudotime_col='palantir_pseudotime',
                                        methods=['weighted_mean', 'peak', 'median'],
                                        min_expression=0.1, min_cells=10):
    """
    Batch calculate temporal center positions for multiple genes
    
    Parameters:
    -----------
    rna_data : AnnData object
        Data object containing gene expression and cell metadata
    cell_indices : array-like
        Cell indices to analyze
    gene_list : list
        List of genes to analyze
    pseudotime_col : str
        Pseudotime column name
    methods : list
        List of calculation methods
    min_expression : float
        Minimum expression threshold
    min_cells : int
        Minimum number of expressing cells
    
    Returns:
    --------
    pd.DataFrame : Table containing temporal center information for all genes
    """
    
    results_list = []
    failed_genes = []
    
    print(f"Starting batch calculation of temporal center positions for {len(gene_list)} genes...")
    
    for i, gene_name in enumerate(gene_list):
        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{len(gene_list)} genes")
        
        try:
            # Get gene expression data for pre-filtering
            gene_expression = rna_data[cell_indices, gene_name].to_df()
            if isinstance(gene_expression, pd.DataFrame):
                gene_expression = gene_expression.iloc[:, 0]
            
            # Check if gene meets filtering criteria
            expressing_cells = (gene_expression > min_expression).sum()
            max_expression = gene_expression.max()
            
            if expressing_cells < min_cells or max_expression < min_expression:
                print(f"Skipping gene {gene_name}: expressing cells={expressing_cells}, max expression={max_expression:.3f}")
                continue
            
            gene_result = {'gene': gene_name}
            
            # Calculate temporal center for each method
            for method in methods:
                try:
                    result = calculate_gene_temporal_center(
                        rna_data, cell_indices, gene_name, 
                        pseudotime_col=pseudotime_col, method=method
                    )
                    
                    # Add method-specific column names
                    gene_result[f'{method}_temporal_center'] = result['temporal_center']
                    gene_result[f'{method}_temporal_width'] = result['temporal_width']
                    
                    # Only add general information on first method (avoid duplication)
                    if method == methods[0]:
                        gene_result['expression_min'] = result['expression_range'][0]
                        gene_result['expression_max'] = result['expression_range'][1]
                        gene_result['pseudotime_min'] = result['pseudotime_range'][0]
                        gene_result['pseudotime_max'] = result['pseudotime_range'][1]
                        gene_result['expr_time_correlation'] = result['correlation']
                        gene_result['n_cells_analyzed'] = result['n_cells']
                        gene_result['expressing_cells'] = expressing_cells
                        
                except Exception as e:
                    print(f"Gene {gene_name} method {method} calculation failed: {e}")
                    gene_result[f'{method}_temporal_center'] = np.nan
                    gene_result[f'{method}_temporal_width'] = np.nan
            
            results_list.append(gene_result)
            
        except Exception as e:
            failed_genes.append(gene_name)
            print(f"Gene {gene_name} completely failed calculation: {e}")
    
    # Convert to DataFrame
    if results_list:
        df_results = pd.DataFrame(results_list)
        
        # Add some derived metrics
        if len(methods) > 1:
            # Calculate consistency between different methods
            temporal_cols = [f'{method}_temporal_center' for method in methods]
            df_results['temporal_center_std'] = df_results[temporal_cols].std(axis=1)
            df_results['temporal_center_range'] = (df_results[temporal_cols].max(axis=1) - 
                                                 df_results[temporal_cols].min(axis=1))
        
        # Calculate expression dynamic range
        df_results['expression_dynamic_range'] = (df_results['expression_max'] - 
                                                df_results['expression_min'])
        
        # Calculate expression density
        df_results['expression_density'] = (df_results['expressing_cells'] / 
                                          df_results['n_cells_analyzed'])
        
        # Sort by temporal center of first method
        main_method = methods[0]
        df_results = df_results.sort_values(f'{main_method}_temporal_center')
        df_results = df_results.reset_index(drop=True)
        
        print(f"\nBatch calculation completed!")
        print(f"Successfully analyzed: {len(df_results)} genes")
        print(f"Calculation failed: {len(failed_genes)} genes")
        
        return df_results, failed_genes
    
    else:
        print("No genes successfully calculated!")
        return pd.DataFrame(), failed_genes

def get_temporal_gene_clusters(df_results, method='weighted_mean', n_clusters=5):
    """
    Cluster genes based on temporal center positions
    """
    from sklearn.cluster import KMeans
    
    temporal_col = f'{method}_temporal_center'
    valid_data = df_results.dropna(subset=[temporal_col])
    
    if len(valid_data) < n_clusters:
        print(f"Valid genes ({len(valid_data)}) less than cluster number ({n_clusters})")
        return df_results
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(valid_data[[temporal_col]])
    
    # Add clustering results
    df_results_copy = df_results.copy()
    df_results_copy['temporal_cluster'] = np.nan
    df_results_copy.loc[valid_data.index, 'temporal_cluster'] = clusters
    
    # Calculate statistics for each cluster
    cluster_stats = []
    for cluster_id in range(n_clusters):
        cluster_genes = df_results_copy[df_results_copy['temporal_cluster'] == cluster_id]
        if len(cluster_genes) > 0:
            stats = {
                'cluster': cluster_id,
                'n_genes': len(cluster_genes),
                'temporal_center_mean': cluster_genes[temporal_col].mean(),
                'temporal_center_std': cluster_genes[temporal_col].std(),
                'temporal_center_min': cluster_genes[temporal_col].min(),
                'temporal_center_max': cluster_genes[temporal_col].max()
            }
            cluster_stats.append(stats)
    
    cluster_stats_df = pd.DataFrame(cluster_stats)
    print("\nTemporal clustering statistics:")
    print(cluster_stats_df.to_string(index=False))
    
    return df_results_copy

def save_temporal_analysis_results(df_results, output_prefix='gene_temporal_analysis'):
    """
    Save analysis results to multiple files
    """
    # Save complete results
    df_results.to_csv(f'{output_prefix}_complete.csv', index=False)
    
    # Save simplified version (main metrics only)
    main_cols = ['gene', 'weighted_mean_temporal_center', 'weighted_mean_temporal_width',
                'expression_max', 'expr_time_correlation', 'expressing_cells', 'expression_density']
    available_cols = [col for col in main_cols if col in df_results.columns]
    df_summary = df_results[available_cols].copy()
    df_summary.to_csv(f'{output_prefix}_summary.csv', index=False)
    
    # Save early, middle, late gene lists
    if 'weighted_mean_temporal_center' in df_results.columns:
        temporal_col = 'weighted_mean_temporal_center'
        valid_results = df_results.dropna(subset=[temporal_col])
        
        # Divide into three parts by temporal center
        quantiles = valid_results[temporal_col].quantile([0.33, 0.67])
        
        early_genes = valid_results[valid_results[temporal_col] <= quantiles[0.33]]['gene'].tolist()
        middle_genes = valid_results[(valid_results[temporal_col] > quantiles[0.33]) & 
                                   (valid_results[temporal_col] <= quantiles[0.67])]['gene'].tolist()
        late_genes = valid_results[valid_results[temporal_col] > quantiles[0.67]]['gene'].tolist()
        
        # Save gene lists
        with open(f'{output_prefix}_early_genes.txt', 'w') as f:
            f.write('\n'.join(early_genes))
        with open(f'{output_prefix}_middle_genes.txt', 'w') as f:
            f.write('\n'.join(middle_genes))
        with open(f'{output_prefix}_late_genes.txt', 'w') as f:
            f.write('\n'.join(late_genes))
        
        print(f"\nTemporal gene classification:")
        print(f"Early genes ({len(early_genes)}): temporal center <= {quantiles[0.33]:.3f}")
        print(f"Middle genes ({len(middle_genes)}): {quantiles[0.33]:.3f} < temporal center <= {quantiles[0.67]:.3f}")
        print(f"Late genes ({len(late_genes)}): temporal center > {quantiles[0.67]:.3f}")
    
    print(f"\nResults saved to {output_prefix}_*.csv and {output_prefix}_*_genes.txt")

# Complete batch analysis workflow
def run_batch_temporal_analysis(rna_margin, axial_cell_idx, gene_list, 
                               output_prefix='gene_temporal_analysis'):
    """
    Run complete batch temporal analysis workflow
    
    Parameters:
    -----------
    rna_margin : AnnData object
        Data object
    axial_cell_idx : array-like  
        Cell indices
    gene_list : list
        Gene list
    output_prefix : str
        Output file prefix
    
    Returns:
    --------
    df_results : pd.DataFrame
        Analysis results table
    """
    
    # Batch calculation
    df_results, failed_genes = batch_calculate_gene_temporal_centers(
        rna_margin, axial_cell_idx, gene_list
    )
    
    if len(df_results) == 0:
        print("No genes successfully analyzed!")
        return None
    
    # Add clustering information
    df_results = get_temporal_gene_clusters(df_results)
    
    # Save results
    save_temporal_analysis_results(df_results, output_prefix)
    
    # Display results summary
    print(f"\n=== Analysis Results Summary ===")
    print(f"Total genes: {len(gene_list)}")
    print(f"Successfully analyzed: {len(df_results)}")
    print(f"Failed genes: {len(failed_genes)}")
    
    if len(df_results) > 0:
        print(f"\nTemporal center position statistics:")
        temporal_col = 'weighted_mean_temporal_center'
        if temporal_col in df_results.columns:
            print(f"Minimum: {df_results[temporal_col].min():.4f}")
            print(f"Maximum: {df_results[temporal_col].max():.4f}")
            print(f"Mean: {df_results[temporal_col].mean():.4f}")
            print(f"Median: {df_results[temporal_col].median():.4f}")
    
    return df_results

# Usage example:
"""
# Prepare gene list
gene_list = ['gata5', 'gata6', 'sox2', 'nanog', 'oct4']  # your gene list

# Run batch analysis
df_results = run_batch_temporal_analysis(
    rna_margin, 
    axial_cell_idx, 
    gene_list,
    output_prefix='axial_genes_temporal'
)

# View results
print(df_results.head())

# Filter genes in specific temporal range
early_genes_df = df_results[df_results['weighted_mean_temporal_center'] < 0.3]
print("Early genes:", early_genes_df['gene'].tolist())
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import umap
from sklearn.neighbors import NearestNeighbors

def build_correlation_network_umap_layout(embedding_df, correlation_threshold=0.7, umap_neighbors=15, min_dist=0.1):
    """
    UMAP layout + correlation network
    
    Parameters:
    -----------
    embedding_df : pd.DataFrame
        Gene embedding data
    correlation_threshold : float
        Correlation threshold, control gene connection
    umap_neighbors : int
        UMAP neighbors parameter
    min_dist : float
        UMAP minimum distance parameter
    
    Returns:
    --------
    G : nx.Graph
        Network graph (based on correlation connection)
    pos : dict
        UMAP layout position
    correlation_matrix : pd.DataFrame
        Correlation matrix
    """
    
    # 1. Calculate correlation matrix
    from sklearn.metrics.pairwise import cosine_similarity
    correlation_matrix = cosine_similarity(embedding_df.values)
    correlation_df = pd.DataFrame(correlation_matrix, 
                                index=embedding_df.index, 
                                columns=embedding_df.index)
    
    # 2. UMAP dimensionality reduction to get layout position
    reducer = umap.UMAP(n_neighbors=umap_neighbors, min_dist=min_dist, random_state=42)
    umap_coords = reducer.fit_transform(embedding_df.values)
    
    # 3. Create network (based on correlation threshold)
    G = nx.Graph()
    genes = embedding_df.index.tolist()
    G.add_nodes_from(genes)
    
    # Add edges: based on correlation threshold
    for i, gene1 in enumerate(genes):
        for j, gene2 in enumerate(genes):
            if i < j:  # 避免重复
                correlation = float(correlation_df.loc[gene1, gene2])
                if abs(correlation) >= correlation_threshold:
                    G.add_edge(gene1, gene2, weight=abs(correlation))
    
    # 4. Create UMAP layout dictionary
    pos = {gene: umap_coords[i] for i, gene in enumerate(genes)}
    
    print(f"Network built successfully:")
    print(f"  Node number: {G.number_of_nodes()}")
    print(f"  Edge number: {G.number_of_edges()}")
    print(f"  Correlation threshold: {correlation_threshold}")
    
    return G, pos, correlation_df

def plot_umap_network(G, pos, node_colors=None, figsize=(12, 8)):
    """Simple plot of UMAP layout network"""
    
    plt.figure(figsize=figsize)
    
    # Plot edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, edge_color='gray')
    
    # Plot nodes
    if node_colors is None:
        node_colors = 'lightblue'
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=50, alpha=0.8)
    
    # Optional: add labels (if nodes are not many)
    if len(G.nodes()) < 50:
        nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title('Gene Network with UMAP Layout')
    plt.axis('off')
    return plt.gcf()

def add_tf_regulation(G, tf_gene_dict):
    """
    Add TF-Gene regulation relationship to network
    
    Parameters:
    -----------
    G : nx.Graph
        Existing network
    tf_gene_dict : dict
        TF-Gene regulation dictionary {TF: [target_genes]}
    
    Returns:
    --------
    G : nx.Graph
        Network with regulation relationship
    tf_genes : list
        TF gene list
    """
    
    tf_genes = []
    regulation_edges = 0
    
    for tf, targets in tf_gene_dict.items():
        if tf in G.nodes():
            tf_genes.append(tf)
            
        for target in targets:
            if tf in G.nodes() and target in G.nodes():
                # Add regulation edge, marked as regulation type
                G.add_edge(tf, target, edge_type='regulation', weight=1.0)
                regulation_edges += 1
    
    print(f"Add regulation relationship:")
    print(f"  TF gene number: {len(tf_genes)}")
    print(f"  Regulation edge number: {regulation_edges}")
    
    return G, tf_genes

def plot_umap_network_with_regulation(G, pos, temporal_df, tf_genes, figsize=(15, 10)):
    """Plot network with regulation relationship"""
    
    plt.figure(figsize=figsize)
    
    # Separate different types of edges
    correlation_edges = [(u, v) for u, v, d in G.edges(data=True) 
                        if d.get('edge_type', 'correlation') == 'correlation']
    regulation_edges = [(u, v) for u, v, d in G.edges(data=True) 
                       if d.get('edge_type') == 'regulation']
    
    # Plot correlation edges (gray)
    nx.draw_networkx_edges(G, pos, edgelist=correlation_edges, 
                          alpha=0.3, width=0.5, edge_color='gray')
    
    # Plot regulation edges (red)
    nx.draw_networkx_edges(G, pos, edgelist=regulation_edges,
                          alpha=0.7, width=1.5, edge_color='red', 
                          style='dashed')
    
    # Get node colors (peak_temporal_center)
    node_colors = []
    for gene in G.nodes():
        if gene in temporal_df['gene'].values:
            temporal_val = float(temporal_df[temporal_df['gene'] == gene]['peak_temporal_center'].values[0])
            node_colors.append(temporal_val)
        else:
            node_colors.append(0)
    
    # Plot all nodes
    scatter = nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                   node_size=80, alpha=0.8, 
                                   cmap='viridis', vmin=min(node_colors), vmax=max(node_colors))
    
    # Only show TF gene labels
    tf_pos = {gene: pos[gene] for gene in tf_genes if gene in pos}
    nx.draw_networkx_labels(G, tf_pos, font_size=8, font_color='black', 
                           font_weight='bold')
    
    # Add colorbar and legend
    plt.colorbar(scatter, label='Peak Temporal Center', shrink=0.8)
    
    # Add edge type legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='gray', lw=1, alpha=0.5, label='Correlation'),
                      Line2D([0], [0], color='red', lw=2, alpha=0.7, linestyle='--', label='TF Regulation')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title('Gene Network: UMAP Layout + Correlation + TF Regulation')
    plt.axis('off')
    return plt.gcf()


def plot_grn(G, pos, tf_list,temporal_df, tf_gene_dict,
             figsize=(6,6),top_tf_target_num=5,title='GRN',ax=None,
             fontsize=12,cmap='RdBu_r',
             ):
    # Plot curved correlation edges
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Calculate PageRank
    pagerank = nx.pagerank(G)
        
    # Separate different types of edges
    correlation_edges = [(u, v) for u, v, d in G.edges(data=True) 
                        if d.get('edge_type', 'correlation') == 'correlation']
    regulation_edges = [(u, v) for u, v, d in G.edges(data=True) 
                    if d.get('edge_type') == 'regulation']

    from matplotlib.patches import FancyArrowPatch
    '''
    for edge in correlation_edges:
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]
        
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            connectionstyle="arc3,rad=0.1",  # 弯曲程度
                            arrowstyle="-", 
                            alpha=0.3, color='gray', linewidth=0.5)
        ax.add_patch(arrow)
    '''

    # Plot curved regulation edges
    for edge in regulation_edges:
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]
        
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            connectionstyle="arc3,rad=0.2",  # More curved
                            arrowstyle="-", 
                            alpha=0.7, color='#c2c2c2', linewidth=1.5,
                            #linestyle='dashed'
                            )
        ax.add_patch(arrow)

    # Get node colors
    node_colors = []
    for gene in G.nodes():
        if gene in temporal_df.index.values:
            temporal_val = float(temporal_df.loc[gene,'peak_temporal_center'])
            node_colors.append(temporal_val)
        else:
            node_colors.append(0)

    # Get node size (based on PageRank)
    node_sizes = []
    for gene in G.nodes():
        pr_value = pagerank[gene]
        # Scale PageRank value to appropriate node size range (50-500)
        size = 50 + (pr_value - min(pagerank.values())) / (max(pagerank.values()) - min(pagerank.values())) * 450
        node_sizes.append(size)

    # Plot nodes
    x_coords = [pos[node][0] for node in G.nodes()]
    y_coords = [pos[node][1] for node in G.nodes()]

    scatter = ax.scatter(x_coords, y_coords, c=node_colors, s=node_sizes, alpha=0.8, 
                        cmap=cmap, vmin=min(node_colors), vmax=max(node_colors))


    # Only show TF labels
    texts=[]
    from ..pl._palette import sc_color
    for tf,colors in zip(tf_list,sc_color[11:]):
        if tf in pos:
            # Add white border
            import matplotlib.patheffects as path_effects
            
            text = ax.text(pos[tf][0], pos[tf][1], tf, fontsize=10, 
            fontweight='normal', ha='center', va='center', color='white')
            text.set_path_effects([path_effects.Stroke(linewidth=3, foreground=colors),
                            path_effects.Normal()])
            texts.append(text)
            target_genes=tf_gene_dict[tf]
            for tf_g in target_genes[:top_tf_target_num]:
                if tf_g in pos:
                    text = ax.text(pos[tf_g][0], pos[tf_g][1], tf_g, fontsize=10, 
                    fontweight='normal', ha='center', va='center', color=colors)
                    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                                    path_effects.Normal()])
                    texts.append(text)
            
            
            
    from adjustText import adjust_text
    adjust_text(texts,only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
                        arrowprops=dict(arrowstyle='->', color='#a51616'))

    # Add colorbar and legend
    plt.colorbar(scatter, label='Mean\nExpression', shrink=0.5)

    #from matplotlib.lines import Line2D
    #legend_elements = [
    #    Line2D([0], [0], color='gray', lw=1, alpha=0.5, label='Correlation'),
    #    Line2D([0], [0], color='red', lw=2, alpha=0.7, linestyle='--', label='TF Regulation')
    #]
    #ax.legend(handles=legend_elements, loc='upper right')

    ax.set_title(title,fontsize=fontsize+1)
    ax.axis('off')
    return ax


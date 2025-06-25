#!/usr/bin/env python3
"""
Process all commot entries and create CellChat-style AnnData object
with obs=cell type pairs, var=ligand-receptor pairs, layers=['pvalues', 'means']
"""

import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
from itertools import combinations_with_replacement

# Get the optimized function
#from omicverse.externel.commot.tools._spatial_communication import summarize_cluster_gpu
from ..externel.commot.tools._spatial_communication import summarize_cluster_gpu

def create_communication_anndata(adata, clustering_column, n_permutations=100):
    """
    Create AnnData object with communication results in CellChat format
    
    Parameters
    ----------
    adata : AnnData
        Input spatial data with commot results
    clustering_column : str
        Column name for cell type clustering
    n_permutations : int
        Number of permutations for p-value calculation
        
    Returns
    -------
    comm_adata : AnnData
        Communication results with structure:
        - obs: cell type pairs ('celltype1|celltype2')  
        - var: ligand-receptor pairs with metadata
        - layers: 'pvalues' and 'means'
    """
    
    # Get cluster info
    celltypes = list(adata.obs[clustering_column].unique())
    celltypes.sort()
    clusterid = np.array(adata.obs[clustering_column], str)
    
    # Find all commot keys and categorize them
    commot_keys = [key for key in adata.obsp.keys() if key.startswith('commot')]
    print(f"Found {len(commot_keys)} commot entries")
    
    # Extract ligand-receptor information from commot database info
    lr_pairs = []
    lr_metadata = []
    
    # Find database info
    database_info_keys = [key for key in adata.uns.keys() if key.startswith('commot-') and key.endswith('-info')]
    
    if database_info_keys:
        # Use database info to get L-R pair details
        db_key = database_info_keys[0]  # Use first database
        df_ligrec = adata.uns[db_key]['df_ligrec']
        
        for _, row in df_ligrec.iterrows():
            ligand = row['ligand']
            receptor = row['receptor'] 
            pathway = row['pathway']
            
            lr_pair_name = f"{ligand}-{receptor}"
            lr_pairs.append(lr_pair_name)
            
            lr_metadata.append({
                'interacting_pair': lr_pair_name,
                'partner_a': ligand,
                'partner_b': receptor,
                'gene_a': ligand,
                'gene_b': receptor,
                'classification': pathway,
                'secreted': True,  # Default assumption
                'receptor_a': False,
                'receptor_b': True,
                'annotation_strategy': 'commot',
                'is_integrin': False,
                'directionality': 'ligand-receptor',
                'id_cp_interaction': f"commot_{ligand}_{receptor}"
            })
    else:
        # Fallback: extract from commot key names
        for key in commot_keys:
            if key.count('-') >= 3:  # Format: commot-database-ligand-receptor
                parts = key.split('-')
                if len(parts) >= 4:
                    ligand = parts[2]
                    receptor = parts[3]
                    
                    lr_pair_name = f"{ligand}-{receptor}"
                    if lr_pair_name not in lr_pairs:
                        lr_pairs.append(lr_pair_name)
                        
                        lr_metadata.append({
                            'interacting_pair': lr_pair_name,
                            'partner_a': ligand,
                            'partner_b': receptor,
                            'gene_a': ligand,
                            'gene_b': receptor,
                            'classification': 'Unknown',
                            'secreted': True,
                            'receptor_a': False,
                            'receptor_b': True,
                            'annotation_strategy': 'commot',
                            'is_integrin': False,
                            'directionality': 'ligand-receptor',
                            'id_cp_interaction': f"commot_{ligand}_{receptor}"
                        })
    
    # Create cell type pairs (sender|receiver format)
    cell_pairs = []
    for sender in celltypes:
        for receiver in celltypes:
            cell_pairs.append(f"{sender}|{receiver}")
    
    print(f"Processing {len(cell_pairs)} cell type pairs × {len(commot_keys)} pathways")
    
    # Initialize data matrices
    n_pairs = len(cell_pairs)
    n_interactions = len(commot_keys)
    
    means_matrix = np.zeros((n_pairs, n_interactions))
    pvalues_matrix = np.ones((n_pairs, n_interactions))  # Default p=1
    
    # Create interaction metadata for each commot key
    interaction_metadata = []
    
    # Process each commot entry
    for j, key in enumerate(tqdm(commot_keys, desc="Processing pathways")):
        S = adata.obsp[key]
        df_cluster, df_p_value = summarize_cluster_gpu(S, clusterid, celltypes, n_permutations,scale_factor='sum')
        
        # Extract pathway/interaction info from key
        if key.startswith('commot-cellchat-'):
            pathway_part = key.replace('commot-cellchat-', '')
            if '-' in pathway_part and not pathway_part.endswith('-total'):
                # Individual L-R pair
                ligand, receptor = pathway_part.split('-', 1)
                interaction_name = f"{ligand}-{receptor}"
                classification = "Individual_LR"
            else:
                # Pathway or total
                interaction_name = pathway_part
                classification = "Pathway" if pathway_part != "total-total" else "Total"
        else:
            interaction_name = key
            classification = "Unknown"
        
        # Store interaction metadata
        interaction_metadata.append({
            'interacting_pair': interaction_name,
            'partner_a': interaction_name.split('-')[0] if '-' in interaction_name else interaction_name,
            'partner_b': interaction_name.split('-')[1] if '-' in interaction_name and len(interaction_name.split('-')) > 1 else interaction_name,
            'gene_a': interaction_name.split('-')[0] if '-' in interaction_name else interaction_name,
            'gene_b': interaction_name.split('-')[1] if '-' in interaction_name and len(interaction_name.split('-')) > 1 else interaction_name,
            'classification': classification,
            'secreted': True,
            'receptor_a': False,
            'receptor_b': True,
            'annotation_strategy': 'commot',
            'is_integrin': False,
            'directionality': 'ligand-receptor',
            'id_cp_interaction': f"commot_{interaction_name.replace('-', '_')}"
        })
        
        # Fill data matrices
        for i, (sender, receiver) in enumerate([(pair.split('|')[0], pair.split('|')[1]) for pair in cell_pairs]):
            if sender in df_cluster.index and receiver in df_cluster.columns:
                means_matrix[i, j] = df_cluster.loc[sender, receiver]
                pvalues_matrix[i, j] = df_p_value.loc[sender, receiver]
    
    # Create observation metadata
    obs_data = []
    for pair in cell_pairs:
        sender, receiver = pair.split('|')
        obs_data.append({
            'sender': sender,
            'receiver': receiver,
            'cell_type_pair': pair
        })
    
    obs_df = pd.DataFrame(obs_data, index=cell_pairs)
    var_df = pd.DataFrame(interaction_metadata, index=commot_keys)
    
    # Create AnnData object
    comm_adata = ad.AnnData(
        X=means_matrix,  # Default layer
        obs=obs_df,
        var=var_df
    )
    
    # Add layers
    comm_adata.layers['means'] = means_matrix
    comm_adata.layers['pvalues'] = pvalues_matrix
    
    print(f"✅ Created AnnData: {comm_adata.n_obs} obs × {comm_adata.n_vars} vars")
    print(f"   obs (cell pairs): {list(comm_adata.obs.columns)}")
    print(f"   var (interactions): {list(comm_adata.var.columns)}")
    print(f"   layers: {list(comm_adata.layers.keys())}")
    
    return comm_adata

def quick_demo(adata, clustering_column, max_pathways=5):
    """
    Quick demo with subset of pathways
    """
    print("🚀 Running quick demo with subset of pathways...")
    
    # Temporarily limit pathways for demo
    all_keys = [key for key in adata.obsp.keys() if key.startswith('commot')]
    demo_keys = all_keys[:max_pathways]
    
    # Create temporary adata with subset
    adata_demo = adata.copy()
    for key in all_keys:
        if key not in demo_keys:
            del adata_demo.obsp[key]
    
    return create_communication_anndata(adata_demo, clustering_column, n_permutations=10)

def process_all_commot(adata, clustering_column, n_permutations=100, return_format='anndata'):
    """
    Simple wrapper that maintains backwards compatibility
    
    Parameters
    ----------
    adata : AnnData
        Input spatial data with commot results
    clustering_column : str  
        Column name for cell type clustering
    n_permutations : int
        Number of permutations for p-value calculation
    return_format : str
        'anndata' for CellChat-style AnnData (recommended)
        'dict' for old dictionary format
        
    Returns
    -------
    results : AnnData or dict
        Communication results in requested format
    """
    
    if return_format == 'anndata':
        return create_communication_anndata(adata, clustering_column, n_permutations)
    
    elif return_format == 'dict':
        # Old format for backwards compatibility
        celltypes = list(adata.obs[clustering_column].unique())
        celltypes.sort()
        clusterid = np.array(adata.obs[clustering_column], str)
        
        commot_keys = [key for key in adata.obsp.keys() if key.startswith('commot')]
        print(f"Found {len(commot_keys)} commot entries to process")
        
        results = {}
        for key in tqdm(commot_keys, desc="Processing"):
            S = adata.obsp[key]
            df_cluster, df_p_value = summarize_cluster_gpu(S, clusterid, celltypes, n_permutations,scale_factor='sum')
            results[key] = {'communication': df_cluster, 'pvalue': df_p_value}
        
        print(f"✅ Done! Processed {len(results)} pathways")
        return results
    
    else:
        raise ValueError("return_format must be 'anndata' or 'dict'")

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_usage():
    """
    Example usage patterns
    """
    print("📝 USAGE EXAMPLES:")
    print()
    print("# 1. Create CellChat-style AnnData (RECOMMENDED)")
    print("comm_adata = create_communication_anndata(adata, 'cell_type')")
    print("# OR using wrapper:")
    print("comm_adata = process_all_commot(adata, 'cell_type')")
    print()
    print("# 2. Quick demo with subset")  
    print("demo_adata = quick_demo(adata, 'cell_type', max_pathways=5)")
    print()
    print("# 3. Use with CellChatViz")
    print("from omicverse.pl._cpdbviz import CellChatViz")
    print("viz = CellChatViz(comm_adata)")
    print("viz.netVisual_circle(viz.compute_aggregated_network()[1])")
    print()
    print("# 4. Access structured results")
    print("print(f'Shape: {comm_adata.shape}')")
    print("print(f'Cell pairs: {comm_adata.obs.sender.unique()}')")
    print("print(f'Pathways: {comm_adata.var.classification.unique()}')")
    print("print(f'Max communication: {comm_adata.layers[\"means\"].max():.4f}')")
    print()
    print("# 5. Backwards compatibility (old dict format)")
    print("results_dict = process_all_commot(adata, 'cell_type', return_format='dict')")
    print("print(results_dict['commot-cellchat-total-total']['communication'])")
    print()
    print("# 6. Export to CSV")
    print("# Communication matrix")
    print("pd.DataFrame(comm_adata.layers['means'], ")
    print("            index=comm_adata.obs.index, ")
    print("            columns=comm_adata.var.index).to_csv('communication_means.csv')")
    print("# P-values")
    print("pd.DataFrame(comm_adata.layers['pvalues'], ")
    print("            index=comm_adata.obs.index, ")
    print("            columns=comm_adata.var.index).to_csv('communication_pvalues.csv')")


import pandas as pd
def update_classification_from_database(comm_adata, adata_with_db):
    """
    Update classification in comm_adata using database info from original adata
    
    Parameters
    ----------
    comm_adata : AnnData
        Communication AnnData object to update
    adata_with_db : AnnData
        Original adata containing commot database info
        
    Returns
    -------
    comm_adata : AnnData
        Updated communication AnnData with proper classifications
    """
    
    # Find database info
    database_info_keys = [key for key in adata_with_db.uns.keys() if key.startswith('commot-') and key.endswith('-info')]
    
    if not database_info_keys:
        print("❌ No commot database info found in adata_with_db.uns")
        return comm_adata
    
    # Use the first available database info
    db_key = database_info_keys[0]
    df_ligrec = adata_with_db.uns[db_key]['df_ligrec']
    
    print(f"📖 Found database info: {db_key}")
    print(f"   Database contains {len(df_ligrec)} ligand-receptor pairs")
    print(f"   Pathways: {sorted(df_ligrec['pathway'].unique())}")
    
    # Create mapping from ligand-receptor pairs to pathways
    lr_to_pathway = {}
    for _, row in df_ligrec.iterrows():
        ligand = row['ligand']
        receptor = row['receptor']
        pathway = row['pathway']
        
        # Store L-R pair to pathway mapping
        lr_key = f"{ligand}-{receptor}"
        lr_to_pathway[lr_key] = {
            'ligand': ligand,
            'receptor': receptor,
            'pathway': pathway
        }
    
    # Update classifications in comm_adata
    updated_classifications = []
    updated_gene_a = []
    updated_gene_b = []
    updated_partner_a = []
    updated_partner_b = []
    
    print(f"\n🔄 Updating classifications for {len(comm_adata.var)} interactions...")
    
    for i, var_name in enumerate(comm_adata.var.index):
        current_pair = comm_adata.var.loc[var_name, 'interacting_pair']
        
        if var_name.startswith('commot-cellchat-'):
            pathway_part = var_name.replace('commot-cellchat-', '')
            
            if pathway_part == 'total-total':
                # Total across all pathways
                updated_classifications.append('Total')
                updated_gene_a.append('total')
                updated_gene_b.append('total')
                updated_partner_a.append('total')
                updated_partner_b.append('total')
                
            elif '-' in pathway_part and not pathway_part.endswith('-total'):
                # Individual ligand-receptor pair
                if pathway_part.count('-') >= 2:
                    # Handle complex receptors like TGFBR1_TGFBR2
                    parts = pathway_part.split('-')
                    ligand = parts[0]
                    receptor = '-'.join(parts[1:])  # Join remaining parts
                else:
                    # Simple ligand-receptor
                    ligand, receptor = pathway_part.split('-', 1)
                
                lr_key = f"{ligand}-{receptor}"
                
                if lr_key in lr_to_pathway:
                    # Found in database - use database info
                    info = lr_to_pathway[lr_key]
                    updated_classifications.append(info['pathway'])
                    updated_gene_a.append(info['ligand'])
                    updated_gene_b.append(info['receptor'])
                    updated_partner_a.append(info['ligand'])
                    updated_partner_b.append(info['receptor'])
                    print(f"   ✓ {lr_key} → {info['pathway']}")
                else:
                    # Not found in database
                    updated_classifications.append('Individual_LR')
                    updated_gene_a.append(ligand)
                    updated_gene_b.append(receptor)
                    updated_partner_a.append(ligand)
                    updated_partner_b.append(receptor)
                    print(f"   ⚠️  {lr_key} → Individual_LR (not in database)")
            
            else:
                # Pathway summary (e.g., WNT, TGFb, SEMA3)
                pathway_name = pathway_part
                updated_classifications.append(pathway_name)
                updated_gene_a.append(pathway_name)
                updated_gene_b.append(pathway_name)
                updated_partner_a.append(pathway_name)
                updated_partner_b.append(pathway_name)
                print(f"   ✓ {pathway_name} → {pathway_name} (pathway summary)")
        
        else:
            # Unknown format - keep original
            updated_classifications.append('Unknown')
            updated_gene_a.append(current_pair)
            updated_gene_b.append(current_pair)
            updated_partner_a.append(current_pair)
            updated_partner_b.append(current_pair)
            print(f"   ❓ {var_name} → Unknown")
    
    # Update the var DataFrame
    comm_adata.var['classification'] = updated_classifications
    comm_adata.var['gene_a'] = updated_gene_a
    comm_adata.var['gene_b'] = updated_gene_b
    comm_adata.var['partner_a'] = updated_partner_a
    comm_adata.var['partner_b'] = updated_partner_b
    
    # Print summary
    pathway_counts = pd.Series(updated_classifications).value_counts()
    print(f"\n📊 Updated pathway breakdown:")
    for pathway, count in pathway_counts.items():
        print(f"   {pathway}: {count} interactions")
    
    print(f"\n✅ Successfully updated classifications in comm_adata!")
    
    return comm_adata

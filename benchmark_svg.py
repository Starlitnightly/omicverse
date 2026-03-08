"""
SVG 方法速度 & 准确率基准测试
比较：SOMDE (n_jobs=1 vs -1) 和 SpatialDE (n_jobs=1 vs -1)
使用模拟数据，含已知 SVG 标签 → 计算 AUROC 评估准确率
"""
import time
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. 模拟数据生成
# ─────────────────────────────────────────────
def simulate_spatial(n_cells=2000, n_svgs=30, n_total=150, seed=42):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, (n_cells, 2)).astype(np.float32)

    # SVG: 以随机高斯斑块编码空间模式
    svg_expr = np.zeros((n_cells, n_svgs), dtype=np.float32)
    for i in range(n_svgs):
        cx, cy = rng.uniform(10, 90, 2)
        sigma  = rng.uniform(8, 25)
        dist2  = (coords[:, 0] - cx)**2 + (coords[:, 1] - cy)**2
        pattern = np.exp(-dist2 / (2 * sigma**2))
        base = rng.negative_binomial(3, 0.4, n_cells).astype(float)
        svg_expr[:, i] = base * (1 + 8 * pattern)

    # 非 SVG: 纯随机
    n_non = n_total - n_svgs
    non_expr = rng.negative_binomial(3, 0.4, (n_cells, n_non)).astype(float)

    X = np.hstack([svg_expr, non_expr])
    gene_names = [f'SVG_{i}' for i in range(n_svgs)] + [f'nonSVG_{i}' for i in range(n_non)]
    is_svg = [True]*n_svgs + [False]*n_non

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f'cell_{i}' for i in range(n_cells)]),
        var=pd.DataFrame({'is_svg': is_svg}, index=gene_names),
    )
    adata.obsm['spatial'] = coords
    adata.layers['counts'] = X.copy()
    return adata

def auroc(scores, labels):
    """AUROC（正例 = SVG）"""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels, scores)

# ─────────────────────────────────────────────
# 2. 运行单次测试
# ─────────────────────────────────────────────
def run_test(adata, mode, n_jobs, **extra):
    import omicverse as ov
    a = adata.copy()
    t0 = time.perf_counter()
    a = ov.space.svg(a, mode=mode, n_jobs=n_jobs, **extra)
    elapsed = time.perf_counter() - t0

    # 准确率：用 LLR 排名做 AUROC（LLR 越大 = 越可能是 SVG）
    col_map = {'somde': 'somde_LLR', 'spatialde': 'spatialde_LLR'}
    llr_col = col_map.get(mode)
    truth = a.var['is_svg'].values
    if llr_col and llr_col in a.var.columns:
        scores = a.var[llr_col].fillna(0).values
        auc = auroc(scores, truth)
    else:
        auc = float('nan')

    n_svg = int(a.var['space_variable_features'].sum())
    return elapsed, auc, n_svg

# ─────────────────────────────────────────────
# 3. 主流程
# ─────────────────────────────────────────────
if __name__ == '__main__':
    import multiprocessing, os
    ncpu = os.cpu_count() or 4
    print(f'可用 CPU: {ncpu}')

    results = []

    for n_cells in [1000, 3000]:
        print(f'\n{"="*60}')
        print(f'模拟数据: {n_cells} 细胞 × 150 基因 (30 SVG)')
        print('='*60)
        adata = simulate_spatial(n_cells=n_cells, n_svgs=30, n_total=150)

        for mode in ['somde', 'spatialde']:
            extra = {'k': 20} if mode == 'somde' else {}
            for n_jobs in [1, -1]:
                label = f'{mode} n_jobs={n_jobs}'
                print(f'\n  ▶ {label} ...', flush=True)
                try:
                    elapsed, auc, n_svg = run_test(adata, mode, n_jobs, **extra)
                    print(f'    ✓ 时间: {elapsed:.1f}s  AUROC: {auc:.3f}  SVG数: {n_svg}')
                    results.append({'n_cells': n_cells, 'mode': mode,
                                    'n_jobs': n_jobs, 'time_s': round(elapsed, 2),
                                    'AUROC': round(auc, 4), 'n_svg': n_svg})
                except Exception as e:
                    print(f'    ✗ 出错: {e}')
                    results.append({'n_cells': n_cells, 'mode': mode,
                                    'n_jobs': n_jobs, 'time_s': None,
                                    'AUROC': None, 'n_svg': None})

    # ─── 汇总表格 ───
    print('\n\n' + '='*60)
    print('汇总结果')
    print('='*60)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # 加速比
    print('\n加速比 (n_jobs=1 → n_jobs=-1):')
    for key, g in df.groupby(['n_cells', 'mode']):
        row1 = g[g['n_jobs'] == 1]['time_s'].values
        rowN = g[g['n_jobs'] == -1]['time_s'].values
        if len(row1) and len(rowN) and row1[0] and rowN[0]:
            speedup = row1[0] / rowN[0]
            print(f'  {key[1]} ({key[0]} cells): {row1[0]:.1f}s → {rowN[0]:.1f}s  '
                  f'({speedup:.1f}x 加速)')

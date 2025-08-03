import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from .forceatlas2 import ForceAtlas2

def test_multithreading():
    """Test and benchmark the multithreaded implementation of ForceAtlas2"""
    # Create a random graph
    n_nodes = 500
    n_edges = 2000
    G = nx.gnm_random_graph(n_nodes, n_edges)
    
    # Convert to adjacency matrix
    A = nx.to_scipy_sparse_array(G)
    
    # Single-threaded run
    print("Running single-threaded...")
    start_time = time.time()
    forceatlas2 = ForceAtlas2(
        # Behavior
        outboundAttractionDistribution=True,
        edgeWeightInfluence=1.0,
        
        # Performance
        jitterTolerance=1.0,
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        multiThreaded=False,
        
        # Tuning
        scalingRatio=2.0,
        strongGravityMode=False,
        gravity=1.0,
        
        # Log
        verbose=True
    )
    pos_single = forceatlas2.forceatlas2(A, iterations=50)
    single_time = time.time() - start_time
    print(f"Single-threaded time: {single_time:.2f} seconds")
    
    # Multi-threaded run
    print("Running multi-threaded...")
    start_time = time.time()
    forceatlas2 = ForceAtlas2(
        # Behavior
        outboundAttractionDistribution=True,
        edgeWeightInfluence=1.0,
        
        # Performance
        jitterTolerance=1.0,
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        multiThreaded=True,
        
        # Tuning
        scalingRatio=2.0,
        strongGravityMode=False,
        gravity=1.0,
        
        # Log
        verbose=True
    )
    pos_multi = forceatlas2.forceatlas2(A, iterations=50)
    multi_time = time.time() - start_time
    print(f"Multi-threaded time: {multi_time:.2f} seconds")
    
    # Calculate speedup
    speedup = single_time / multi_time
    print(f"Speedup: {speedup:.2f}x")
    
    # Compare final positions to verify correctness
    pos_single_arr = np.array(pos_single)
    pos_multi_arr = np.array(pos_multi)
    
    # The layouts might be different due to thread execution order,
    # but should have similar statistical properties
    print("Single-threaded stats:")
    print(f"  Mean position: ({np.mean(pos_single_arr[:, 0]):.3f}, {np.mean(pos_single_arr[:, 1]):.3f})")
    print(f"  Standard deviation: ({np.std(pos_single_arr[:, 0]):.3f}, {np.std(pos_single_arr[:, 1]):.3f})")
    
    print("Multi-threaded stats:")
    print(f"  Mean position: ({np.mean(pos_multi_arr[:, 0]):.3f}, {np.mean(pos_multi_arr[:, 1]):.3f})")
    print(f"  Standard deviation: ({np.std(pos_multi_arr[:, 0]):.3f}, {np.std(pos_multi_arr[:, 1]):.3f})")
    
    # Visualize the layouts
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.title("Single-threaded")
    nx.draw(G, pos=dict(zip(range(n_nodes), pos_single)), node_size=20, edge_color='gray', alpha=0.6)
    
    plt.subplot(122)
    plt.title("Multi-threaded")
    nx.draw(G, pos=dict(zip(range(n_nodes), pos_multi)), node_size=20, edge_color='gray', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("forceatlas2_comparison.png")
    plt.close()
    
    print("Visualization saved to forceatlas2_comparison.png")

if __name__ == "__main__":
    test_multithreading() 
#!/usr/bin/env python3
"""Quick test for GPT-5 code extraction with debug logging."""

import os
os.environ['OVAGENT_DEBUG'] = '1'  # Enable debug logging

import omicverse as ov
import scanpy as sc

print("=" * 60)
print("Quick GPT-5 Test with Debug Logging")
print("=" * 60)

# Initialize GPT-5 agent
print("\n1. Initializing GPT-5 agent...")
agent = ov.Agent(
    model='gpt-5',
    api_key=os.getenv('OPENAI_API_KEY')
)
print("   ✓ Agent initialized")

# Load data
print("\n2. Loading PBMC3k data...")
adata = sc.datasets.pbmc3k()
print(f"   ✓ Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

# Try a simple request
print("\n3. Running quality control request...")
print("   Request: 'quality control with nUMI>500, mito<0.2'")
print("   (Watch for debug output below)")
print("-" * 60)

try:
    result = agent.run('quality control with nUMI>500, mito<0.2', adata)
    print("-" * 60)
    print(f"\n✅ SUCCESS! Cells: {adata.n_obs} → {result.n_obs}")
    print(f"   Filtered: {adata.n_obs - result.n_obs} cells removed")
except Exception as e:
    print("-" * 60)
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

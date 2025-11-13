#!/usr/bin/env python3
"""
OVAgent Simple Test - Workaround for Prompt Length Issue

This simplified test bypasses the function registry prompt length issue
by testing the agent with shorter, more direct requests.

Bug Found: Function registry generates 124K char prompts (max: 100K)
Issue: agent_backend.py:331-333 has hardcoded 100,000 char limit
"""

import os
import sys
from datetime import datetime

print("="*80)
print("OVAGENT SIMPLIFIED TEST - OpenAI and Gemini")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Check prerequisites
try:
    import omicverse as ov
    import scanpy as sc
    print(f"✅ OmicVerse: {ov.__version__}")
    print(f"✅ Scanpy: {sc.__version__}")
except ImportError as e:
    print(f"❌ Missing package: {e}")
    sys.exit(1)

# Check API keys
openai_key = os.getenv('OPENAI_API_KEY')
gemini_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')

print(f"\nAPI Keys:")
print(f"  {'✅' if openai_key else '❌'} OPENAI_API_KEY")
print(f"  {'✅' if gemini_key else '❌'} GEMINI_API_KEY")

if not (openai_key or gemini_key):
    print("\n❌ No API keys found. Set at least one:")
    print("   export OPENAI_API_KEY='your-key'")
    print("   export GEMINI_API_KEY='your-key'")
    sys.exit(1)

# Load PBMC3k
print("\n" + "-"*80)
print("Loading PBMC3k Data")
print("-"*80)

try:
    adata = sc.datasets.pbmc3k()
    print(f"✅ Loaded PBMC3k: {adata.n_obs} cells × {adata.n_vars} genes\n")
except Exception as e:
    print(f"❌ Failed to load data: {e}")
    sys.exit(1)

# Test function
def test_agent_simple(model, api_key, adata):
    """Test agent with simple initialization only"""
    print("="*80)
    print(f"Testing: {model}")
    print("="*80)

    results = {
        'model': model,
        'initialization': False,
        'error': None
    }

    try:
        # Test 1: Initialization
        print("\n🧪 Test 1: Agent Initialization")
        agent = ov.Agent(model=model, api_key=api_key)
        print(f"   ✅ Agent created successfully")
        results['initialization'] = True

        # Test 2: Try simple request (will likely fail due to prompt length)
        print("\n🧪 Test 2: Simple QC Request")
        print("   NOTE: This may fail due to bug - prompt too long")

        try:
            result = agent.run('quality control with nUMI>500, mito<0.2', adata.copy())
            print(f"   ✅ QC succeeded: {adata.n_obs} → {result.n_obs} cells")
            results['qc'] = True
        except ValueError as e:
            if 'user_prompt too long' in str(e):
                print(f"   ⚠️  KNOWN BUG: {e}")
                print(f"   → Function registry generates 124K chars (max: 100K)")
                results['qc'] = False
                results['error'] = 'prompt_too_long'
            else:
                raise

    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['error'] = str(e)

    print()
    return results

# Run tests
all_results = []

# Test OpenAI
if openai_key:
    print("\n" + "="*80)
    print("TESTING OPENAI MODELS")
    print("="*80 + "\n")

    for model in ['gpt-4o-mini']:  # Just test one model
        result = test_agent_simple(model, openai_key, adata)
        all_results.append(result)

# Test Gemini
if gemini_key:
    print("\n" + "="*80)
    print("TESTING GEMINI MODELS")
    print("="*80 + "\n")

    for model in ['gemini/gemini-2.5-flash']:  # Just test one model
        result = test_agent_simple(model, gemini_key, adata)
        all_results.append(result)

# Summary
print("="*80)
print("TEST SUMMARY")
print("="*80)

for r in all_results:
    status = "✅ INIT OK" if r['initialization'] else "❌ FAILED"
    print(f"\n{r['model']}: {status}")
    if r.get('error') == 'prompt_too_long':
        print(f"  ⚠️  Hit known bug: prompt length limit")
    elif r.get('error'):
        print(f"  Error: {r['error']}")

# Bug report
print("\n" + "="*80)
print("BUG REPORT")
print("="*80)
print("""
🐛 CRITICAL BUG FOUND: Prompt Length Limit Exceeded

Location: omicverse/utils/agent_backend.py:331-333
Issue: Function registry generates ~124,000 character prompts
Limit: Hardcoded 100,000 character maximum
Impact: ALL requests fail, even simple ones

Error Message:
  ValueError: user_prompt too long (124017 chars, max 100000)

Affected:
  - Priority 1 workflow: 124,304 chars
  - Priority 2 workflow: 124,017 chars
  - All providers (OpenAI, Gemini, etc.)

Root Cause:
  The function registry (110 functions in 7 categories) is being
  serialized into the prompt, causing it to exceed limits.

Suggested Fixes:
  1. Increase limit to 150,000+ chars (quick fix)
  2. Reduce function registry size in prompts
  3. Use progressive loading for functions (like skills)
  4. Compress or summarize function descriptions

Workaround:
  Currently no workaround for users. Bug must be fixed in code.
""")

print("="*80)
print(f"Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

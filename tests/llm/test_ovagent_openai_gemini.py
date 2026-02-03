#!/usr/bin/env python3
"""
OVAgent Comprehensive Test Suite - OpenAI and Gemini Providers
Based on OVAGENT_PBMC3K_TESTING_PLAN.md

Run this script directly (not via pytest) in an environment with omicverse and scanpy installed.
Requires OPENAI_API_KEY and/or GEMINI_API_KEY environment variables.

Usage:
    python tests/llm/test_ovagent_openai_gemini.py

Note: This is a standalone test script, not a pytest test.
"""

import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
import pytest

# Skip this file when running under pytest - it's designed to run as a standalone script
pytestmark = pytest.mark.skip(reason="Standalone script - run directly with: python tests/llm/test_ovagent_openai_gemini.py")

# Configuration
TEST_OPENAI = True  # Set to False to skip OpenAI tests
TEST_GEMINI = True  # Set to False to skip Gemini tests

# Models to test
OPENAI_MODELS = [
    'gpt-5',           # Latest GPT-5
    'gpt-4o-mini',     # Fast and cost-effective
]

GEMINI_MODELS = [
    'gemini/gemini-2.5-pro',    # Most capable
    'gemini/gemini-2.5-flash',  # Fast
]

# Test results storage
test_results = {
    'start_time': datetime.now(),
    'tests_run': 0,
    'tests_passed': 0,
    'tests_failed': 0,
    'tests_skipped': 0,
    'provider_results': {},
    'errors': []
}


def print_header(text, char='='):
    """Print formatted section header"""
    print(f"\n{char * 80}")
    print(f"{text}")
    print(f"{char * 80}\n")


def print_test(test_name):
    """Print test name"""
    print(f"🧪 TEST: {test_name}")


def print_result(passed, message=""):
    """Print test result"""
    global test_results
    test_results['tests_run'] += 1
    if passed:
        test_results['tests_passed'] += 1
        print(f"✅ PASSED {f'- {message}' if message else ''}")
    else:
        test_results['tests_failed'] += 1
        print(f"❌ FAILED {f'- {message}' if message else ''}")
    print()


def print_skip(message=""):
    """Print skipped test"""
    global test_results
    test_results['tests_skipped'] += 1
    print(f"⏭️  SKIPPED {f'- {message}' if message else ''}\n")


def check_prerequisites():
    """Check if all prerequisites are met"""
    print_header("CHECKING PREREQUISITES")

    all_good = True

    # Check Python version
    print(f"Python version: {sys.version}")

    # Check omicverse
    try:
        import omicverse as ov
        print(f"✅ OmicVerse version: {ov.__version__}")
        print(f"   Location: {ov.__file__}")
    except ImportError as e:
        print(f"❌ OmicVerse not installed: {e}")
        all_good = False
        return False

    # Check scanpy
    try:
        import scanpy as sc
        print(f"✅ Scanpy version: {sc.__version__}")
    except ImportError as e:
        print(f"❌ Scanpy not installed: {e}")
        all_good = False
        return False

    # Check API keys
    print("\nAPI Key Status:")
    openai_key = os.getenv('OPENAI_API_KEY')
    gemini_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')

    if openai_key:
        print(f"  ✅ OPENAI_API_KEY: Set (length={len(openai_key)})")
    else:
        print(f"  ❌ OPENAI_API_KEY: Not set")
        if TEST_OPENAI:
            all_good = False

    if gemini_key:
        print(f"  ✅ GEMINI_API_KEY: Set (length={len(gemini_key)})")
    else:
        print(f"  ❌ GEMINI_API_KEY: Not set")
        if TEST_GEMINI:
            all_good = False

    # Check skill directory
    pkg_root = Path(ov.__file__).resolve().parents[1]
    builtin_skill_path = pkg_root / 'omicverse' / '.claude' / 'skills'

    print(f"\nSkill Directory:")
    if builtin_skill_path.exists():
        skill_count = len(list(builtin_skill_path.glob('*/SKILL.md')))
        print(f"  ✅ Built-in skills: {builtin_skill_path}")
        print(f"     {skill_count} skills found")
    else:
        print(f"  ⚠️  Built-in skills directory not found: {builtin_skill_path}")

    return all_good


def load_pbmc3k_data():
    """Load PBMC3k dataset"""
    import scanpy as sc

    print_header("LOADING PBMC3K DATA", '-')

    adata = None
    local_path = os.environ.get('PBMC3K_PATH')

    if local_path and os.path.exists(local_path):
        adata = sc.read_h5ad(local_path)
        print(f'✅ Loaded local PBMC3k from: {local_path}')
    else:
        try:
            adata = sc.datasets.pbmc3k()
            print('✅ Loaded Scanpy pbmc3k dataset')
        except Exception as e:
            print(f'⚠️  pbmc3k not available: {e}')
            try:
                adata = sc.datasets.pbmc68k_reduced()
                print('✅ Loaded fallback pbmc68k_reduced dataset')
            except Exception as e2:
                print(f'❌ Could not load any PBMC dataset: {e2}')
                return None

    print(f"   Dataset: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def test_agent_initialization(model, api_key):
    """Test Case 1.1.1: Initialize agent"""
    print_test(f"Agent Initialization - {model}")

    try:
        import omicverse as ov
        agent = ov.Agent(model=model, api_key=api_key)

        # Verify agent created
        assert agent is not None, "Agent is None"
        print(f"   Agent initialized successfully")
        print(f"   Model: {model}")

        print_result(True, "Agent initialized")
        return agent

    except Exception as e:
        print(f"   Error: {e}")
        traceback.print_exc()
        print_result(False, str(e))
        test_results['errors'].append({
            'test': 'Agent Initialization',
            'model': model,
            'error': str(e)
        })
        return None


def test_quality_control(agent, adata, model):
    """Test Case 1.2.1: Basic QC with cell filtering"""
    print_test(f"Quality Control - {model}")

    try:
        start_time = time.time()

        # Run QC
        result = agent.run('quality control with nUMI>500, mito<0.2', adata.copy())

        elapsed = time.time() - start_time

        # Verify filtering happened
        assert result.n_obs < adata.n_obs, "No cells were filtered"

        print(f"   QC completed in {elapsed:.2f}s")
        print(f"   Cells: {adata.n_obs} → {result.n_obs} ({result.n_obs/adata.n_obs*100:.1f}% retained)")

        print_result(True, f"{result.n_obs} cells retained")
        return result

    except Exception as e:
        print(f"   Error: {e}")
        traceback.print_exc()
        print_result(False, str(e))
        test_results['errors'].append({
            'test': 'Quality Control',
            'model': model,
            'error': str(e)
        })
        return None


def test_preprocessing(agent, adata, model):
    """Test Case 1.3.1: Standard preprocessing"""
    print_test(f"Preprocessing with HVG Selection - {model}")

    try:
        start_time = time.time()

        # Run preprocessing
        result = agent.run(
            'preprocess with 2000 highly variable genes using shiftlog|pearson',
            adata.copy()
        )

        elapsed = time.time() - start_time

        # Verify HVGs computed (check for both possible column names)
        hvg_column = None
        if 'highly_variable' in result.var.columns:
            hvg_column = 'highly_variable'
        elif 'highly_variable_features' in result.var.columns:
            hvg_column = 'highly_variable_features'

        assert hvg_column is not None, "HVGs not computed (neither 'highly_variable' nor 'highly_variable_features' found)"
        hvg_count = result.var[hvg_column].sum()

        print(f"   Preprocessing completed in {elapsed:.2f}s")
        print(f"   HVGs identified: {hvg_count}")

        # Allow some tolerance (should be ~2000)
        assert 1500 <= hvg_count <= 2500, f"Unexpected HVG count: {hvg_count}"

        print_result(True, f"{hvg_count} HVGs identified")
        return result

    except Exception as e:
        print(f"   Error: {e}")
        traceback.print_exc()
        print_result(False, str(e))
        test_results['errors'].append({
            'test': 'Preprocessing',
            'model': model,
            'error': str(e)
        })
        return None


def test_clustering(agent, adata, model):
    """Test Case 1.4.1: Leiden clustering"""
    print_test(f"Leiden Clustering - {model}")

    try:
        start_time = time.time()

        # Run clustering
        result = agent.run('leiden clustering resolution=1.0', adata.copy())

        elapsed = time.time() - start_time

        # Verify clustering performed
        assert 'leiden' in result.obs.columns, "Leiden clustering not performed"
        n_clusters = result.obs['leiden'].nunique()

        print(f"   Clustering completed in {elapsed:.2f}s")
        print(f"   Clusters identified: {n_clusters}")

        assert n_clusters > 1, "Only one cluster found"

        print_result(True, f"{n_clusters} clusters identified")
        return result

    except Exception as e:
        print(f"   Error: {e}")
        traceback.print_exc()
        print_result(False, str(e))
        test_results['errors'].append({
            'test': 'Clustering',
            'model': model,
            'error': str(e)
        })
        return None


def test_umap_visualization(agent, adata, model):
    """Test Case 1.5.1: UMAP computation"""
    print_test(f"UMAP Visualization - {model}")

    try:
        start_time = time.time()

        # Run UMAP
        result = agent.run('compute umap and plot colored by leiden', adata.copy())

        elapsed = time.time() - start_time

        # Verify UMAP computed
        assert 'X_umap' in result.obsm.keys(), "UMAP not computed"
        umap_shape = result.obsm['X_umap'].shape

        print(f"   UMAP completed in {elapsed:.2f}s")
        print(f"   UMAP shape: {umap_shape}")

        assert umap_shape[1] == 2, "UMAP should have 2 dimensions"

        print_result(True, f"UMAP shape {umap_shape}")
        return result

    except Exception as e:
        print(f"   Error: {e}")
        traceback.print_exc()
        print_result(False, str(e))
        test_results['errors'].append({
            'test': 'UMAP Visualization',
            'model': model,
            'error': str(e)
        })
        return None


def test_semantic_matching(agent, adata, model):
    """Test Case 2.1.2: Semantic matching accuracy"""
    print_test(f"Semantic Skill Matching - {model}")

    test_cases = [
        ("QC my data", "should handle preprocessing"),
        ("filter low quality cells", "should handle QC"),
        ("cluster cells", "should handle clustering"),
    ]

    try:
        passed = 0
        for request, expected in test_cases:
            try:
                print(f"   Testing: '{request}'")
                result = agent.run(request, adata.copy())
                print(f"   ✓ '{request}' → executed successfully")
                passed += 1
            except Exception as e:
                print(f"   ✗ '{request}' → failed: {e}")

        success_rate = passed / len(test_cases)
        print(f"   Success rate: {passed}/{len(test_cases)} ({success_rate*100:.0f}%)")

        print_result(success_rate >= 0.7, f"{passed}/{len(test_cases)} semantic variations handled")

    except Exception as e:
        print(f"   Error: {e}")
        traceback.print_exc()
        print_result(False, str(e))
        test_results['errors'].append({
            'test': 'Semantic Matching',
            'model': model,
            'error': str(e)
        })


def test_end_to_end_pipeline(agent, adata, model):
    """Test Case 1.6.1: Full workflow in sequence"""
    print_test(f"End-to-End Pipeline - {model}")

    try:
        start_time = time.time()

        print("   Step 1: Quality Control")
        adata_work = agent.run('quality control with nUMI>500, mito<0.2', adata.copy())
        print(f"      → {adata_work.n_obs} cells after QC")

        print("   Step 2: Preprocessing")
        adata_work = agent.run('preprocess with 2000 highly variable genes using shiftlog|pearson', adata_work)
        # Check for both possible HVG column names
        hvg_col = 'highly_variable' if 'highly_variable' in adata_work.var.columns else 'highly_variable_features'
        hvg_count = adata_work.var[hvg_col].sum()
        print(f"      → {hvg_count} HVGs selected")

        print("   Step 3: Clustering")
        adata_work = agent.run('leiden clustering resolution=1.0', adata_work)
        n_clusters = adata_work.obs['leiden'].nunique()
        print(f"      → {n_clusters} clusters identified")

        print("   Step 4: UMAP Visualization")
        adata_work = agent.run('compute umap and plot colored by leiden', adata_work)
        print(f"      → UMAP shape {adata_work.obsm['X_umap'].shape}")

        elapsed = time.time() - start_time

        # Validate final state
        checks = [
            ('QC applied', adata_work.n_obs < adata.n_obs),
            ('HVGs computed', 'highly_variable' in adata_work.var.columns or 'highly_variable_features' in adata_work.var.columns),
            ('Clustering done', 'leiden' in adata_work.obs.columns),
            ('UMAP computed', 'X_umap' in adata_work.obsm.keys()),
        ]

        all_passed = all(check[1] for check in checks)

        print(f"\n   Pipeline completed in {elapsed:.2f}s")
        print("   Final validation:")
        for check_name, check_result in checks:
            status = "✓" if check_result else "✗"
            print(f"      {status} {check_name}")

        print_result(all_passed, f"Complete pipeline in {elapsed:.2f}s")
        return adata_work

    except Exception as e:
        print(f"   Error: {e}")
        traceback.print_exc()
        print_result(False, str(e))
        test_results['errors'].append({
            'test': 'End-to-End Pipeline',
            'model': model,
            'error': str(e)
        })
        return None


def test_provider(provider_name, models, api_key, adata):
    """Test all models for a given provider"""
    print_header(f"TESTING {provider_name.upper()} PROVIDER")

    provider_results = {
        'models_tested': 0,
        'models_passed': 0,
        'models_failed': 0,
        'model_details': {}
    }

    for model in models:
        print_header(f"Testing Model: {model}", '-')

        model_start = time.time()
        model_results = {
            'tests': {},
            'total_time': 0,
            'success': False
        }

        # Test 1: Agent Initialization
        agent = test_agent_initialization(model, api_key)
        model_results['tests']['initialization'] = agent is not None

        if agent is None:
            provider_results['models_failed'] += 1
            provider_results['model_details'][model] = model_results
            continue

        # Test 2: Quality Control
        adata_qc = test_quality_control(agent, adata, model)
        model_results['tests']['quality_control'] = adata_qc is not None

        if adata_qc is None:
            provider_results['models_failed'] += 1
            provider_results['model_details'][model] = model_results
            continue

        # Test 3: Preprocessing
        adata_prep = test_preprocessing(agent, adata_qc, model)
        model_results['tests']['preprocessing'] = adata_prep is not None

        # Test 4: Clustering
        if adata_prep is not None:
            adata_clust = test_clustering(agent, adata_prep, model)
            model_results['tests']['clustering'] = adata_clust is not None

            # Test 5: UMAP
            if adata_clust is not None:
                adata_umap = test_umap_visualization(agent, adata_clust, model)
                model_results['tests']['umap'] = adata_umap is not None

        # Test 6: Semantic Matching
        test_semantic_matching(agent, adata, model)

        # Test 7: End-to-End Pipeline
        adata_final = test_end_to_end_pipeline(agent, adata, model)
        model_results['tests']['end_to_end'] = adata_final is not None

        # Calculate results
        model_elapsed = time.time() - model_start
        model_results['total_time'] = model_elapsed
        model_results['success'] = all(model_results['tests'].values())

        provider_results['models_tested'] += 1
        if model_results['success']:
            provider_results['models_passed'] += 1
        else:
            provider_results['models_failed'] += 1

        provider_results['model_details'][model] = model_results

        print(f"\n{'='*80}")
        print(f"Model {model} Summary:")
        print(f"  Total time: {model_elapsed:.2f}s")
        print(f"  Tests passed: {sum(model_results['tests'].values())}/{len(model_results['tests'])}")
        print(f"  Overall: {'✅ PASSED' if model_results['success'] else '❌ FAILED'}")
        print(f"{'='*80}\n")

    test_results['provider_results'][provider_name] = provider_results
    return provider_results


def print_final_summary():
    """Print comprehensive test summary"""
    print_header("FINAL TEST SUMMARY")

    end_time = datetime.now()
    total_time = (end_time - test_results['start_time']).total_seconds()

    print(f"Test Execution Time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"Start: {test_results['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Overall Test Results:")
    print(f"  Total Tests Run:    {test_results['tests_run']}")
    print(f"  Tests Passed:       {test_results['tests_passed']} ✅")
    print(f"  Tests Failed:       {test_results['tests_failed']} ❌")
    print(f"  Tests Skipped:      {test_results['tests_skipped']} ⏭️")

    if test_results['tests_run'] > 0:
        pass_rate = test_results['tests_passed'] / test_results['tests_run'] * 100
        print(f"  Pass Rate:          {pass_rate:.1f}%")
    print()

    # Provider summaries
    for provider, results in test_results['provider_results'].items():
        print(f"{provider.upper()} Provider Summary:")
        print(f"  Models Tested:      {results['models_tested']}")
        print(f"  Models Passed:      {results['models_passed']} ✅")
        print(f"  Models Failed:      {results['models_failed']} ❌")

        for model, details in results['model_details'].items():
            status = "✅ PASSED" if details['success'] else "❌ FAILED"
            tests_passed = sum(details['tests'].values())
            total_tests = len(details['tests'])
            print(f"    {model}:")
            print(f"      Status: {status}")
            print(f"      Tests:  {tests_passed}/{total_tests}")
            print(f"      Time:   {details['total_time']:.2f}s")
        print()

    # Errors summary
    if test_results['errors']:
        print("Errors Encountered:")
        for i, error in enumerate(test_results['errors'], 1):
            print(f"  {i}. {error['test']} ({error['model']})")
            print(f"     {error['error']}")
        print()

    # Success criteria evaluation
    print("Success Criteria Evaluation:")

    criteria = []

    # At least 2 providers tested
    providers_tested = len(test_results['provider_results'])
    criteria.append(("At least 1 provider tested", providers_tested >= 1))

    # Pass rate > 80%
    if test_results['tests_run'] > 0:
        pass_rate = test_results['tests_passed'] / test_results['tests_run']
        criteria.append(("Pass rate > 80%", pass_rate > 0.8))

    # At least one model fully passed
    any_model_passed = any(
        details['success']
        for provider_results in test_results['provider_results'].values()
        for details in provider_results['model_details'].values()
    )
    criteria.append(("At least one model fully passed", any_model_passed))

    for criterion, passed in criteria:
        status = "✅" if passed else "❌"
        print(f"  {status} {criterion}")

    all_criteria_met = all(c[1] for c in criteria)
    print()
    print(f"{'='*80}")
    if all_criteria_met:
        print("🎉 ALL SUCCESS CRITERIA MET!")
    else:
        print("⚠️  Some success criteria not met")
    print(f"{'='*80}")


def main():
    """Main test execution"""
    print_header("OVAGENT COMPREHENSIVE TEST SUITE")
    print("Testing OpenAI and Gemini Providers with PBMC3k Data")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please install required packages and set API keys.")
        print("\nRequired:")
        print("  - pip install omicverse scanpy")
        print("  - export OPENAI_API_KEY='your-key'")
        print("  - export GEMINI_API_KEY='your-key'")
        sys.exit(1)

    # Load data
    adata = load_pbmc3k_data()
    if adata is None:
        print("\n❌ Could not load PBMC3k data. Exiting.")
        sys.exit(1)

    # Test OpenAI
    if TEST_OPENAI:
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            test_provider('OpenAI', OPENAI_MODELS, openai_key, adata)
        else:
            print_header("SKIPPING OPENAI TESTS - No API Key")
            test_results['tests_skipped'] += len(OPENAI_MODELS) * 7  # 7 tests per model

    # Test Gemini
    if TEST_GEMINI:
        gemini_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if gemini_key:
            test_provider('Gemini', GEMINI_MODELS, gemini_key, adata)
        else:
            print_header("SKIPPING GEMINI TESTS - No API Key")
            test_results['tests_skipped'] += len(GEMINI_MODELS) * 7  # 7 tests per model

    # Print final summary
    print_final_summary()

    # Save results to file
    results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("OVAGENT TEST RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Tests Run: {test_results['tests_run']}\n")
        f.write(f"Tests Passed: {test_results['tests_passed']}\n")
        f.write(f"Tests Failed: {test_results['tests_failed']}\n")
        f.write(f"Tests Skipped: {test_results['tests_skipped']}\n\n")

        for provider, results in test_results['provider_results'].items():
            f.write(f"\n{provider.upper()} Results:\n")
            f.write(f"  Models Tested: {results['models_tested']}\n")
            f.write(f"  Models Passed: {results['models_passed']}\n")
            for model, details in results['model_details'].items():
                f.write(f"\n  {model}:\n")
                f.write(f"    Success: {details['success']}\n")
                f.write(f"    Time: {details['total_time']:.2f}s\n")
                for test_name, passed in details['tests'].items():
                    f.write(f"      {test_name}: {'PASS' if passed else 'FAIL'}\n")

    print(f"\n📄 Results saved to: {results_file}")

    # Exit with appropriate code
    if test_results['tests_failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)

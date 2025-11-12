# OVAgent Test Execution Instructions

**Test Script**: `test_ovagent_openai_gemini.py`
**Plan Reference**: `OVAGENT_PBMC3K_TESTING_PLAN.md`
**Providers**: OpenAI and Gemini

---

## Prerequisites

### 1. Install Required Packages

```bash
# If you don't have omicverse installed yet:
pip install omicverse scanpy

# Or if you're in the omicverse repo:
pip install -e .
```

### 2. Set API Keys

You need at least one API key (OpenAI or Gemini):

```bash
# For OpenAI (required for GPT models)
export OPENAI_API_KEY="sk-your-openai-api-key-here"

# For Gemini (required for Gemini models)
export GEMINI_API_KEY="your-gemini-api-key-here"
# OR
export GOOGLE_API_KEY="your-google-api-key-here"
```

**How to get API keys:**
- **OpenAI**: https://platform.openai.com/api-keys
- **Gemini**: https://aistudio.google.com/app/apikey

---

## Quick Start - Run Tests

### Option 1: Test Both Providers (Recommended)

```bash
# Set both API keys
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"

# Run all tests
python test_ovagent_openai_gemini.py
```

### Option 2: Test Only OpenAI

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-openai-key"

# Edit the script to disable Gemini
# Change line: TEST_GEMINI = False

# Run tests
python test_ovagent_openai_gemini.py
```

### Option 3: Test Only Gemini

```bash
# Set Gemini API key
export GEMINI_API_KEY="your-gemini-key"

# Edit the script to disable OpenAI
# Change line: TEST_OPENAI = False

# Run tests
python test_ovagent_openai_gemini.py
```

---

## Exact Commands (Step-by-Step)

### Setup

```bash
# 1. Navigate to omicverse directory
cd /path/to/omicverse

# 2. Ensure omicverse is installed
pip list | grep omicverse
# If not installed: pip install -e .

# 3. Verify scanpy is installed
pip list | grep scanpy
# If not installed: pip install scanpy

# 4. Set your API keys (choose one or both)
export OPENAI_API_KEY="sk-proj-..."
export GEMINI_API_KEY="AIza..."

# 5. Verify API keys are set
echo "OpenAI: ${OPENAI_API_KEY:0:10}..."
echo "Gemini: ${GEMINI_API_KEY:0:10}..."
```

### Run Tests

```bash
# Make script executable (optional)
chmod +x test_ovagent_openai_gemini.py

# Run the tests
python3 test_ovagent_openai_gemini.py

# Or if you made it executable:
./test_ovagent_openai_gemini.py
```

### Expected Runtime

- **OpenAI GPT-4o-mini**: ~3-5 minutes per model
- **OpenAI GPT-5**: ~5-7 minutes per model
- **Gemini Flash**: ~3-5 minutes per model
- **Gemini Pro**: ~5-7 minutes per model

**Total**: ~15-30 minutes for all models

---

## What the Test Script Does

### Tests Performed (per model):

1. **Agent Initialization** - Verifies agent can be created
2. **Quality Control** - Filters cells with nUMI>500, mito<0.2
3. **Preprocessing** - Normalizes and selects 2000 HVGs
4. **Clustering** - Runs Leiden clustering
5. **UMAP Visualization** - Computes UMAP embedding
6. **Semantic Matching** - Tests natural language understanding
7. **End-to-End Pipeline** - Complete workflow validation

### Models Tested:

**OpenAI:**
- `gpt-5` - Latest GPT-5 model
- `gpt-4o-mini` - Fast and cost-effective

**Gemini:**
- `gemini/gemini-2.5-pro` - Most capable
- `gemini/gemini-2.5-flash` - Fast

---

## Output and Results

### Console Output

The script prints detailed progress:

```
================================================================================
TESTING OPENAI PROVIDER
================================================================================

--------------------------------------------------------------------------------
Testing Model: gpt-5
--------------------------------------------------------------------------------

🧪 TEST: Agent Initialization - gpt-5
   Agent initialized successfully
   Model: gpt-5
✅ PASSED - Agent initialized

🧪 TEST: Quality Control - gpt-5
   QC completed in 12.34s
   Cells: 2700 → 2638 (97.7% retained)
✅ PASSED - 2638 cells retained

...
```

### Results File

A timestamped results file is created:

```
test_results_20251112_143052.txt
```

Contains:
- Summary statistics
- Per-model results
- Test details
- Timing information

### Success Criteria

The script checks:
- ✅ At least 1 provider tested
- ✅ Pass rate > 80%
- ✅ At least one model fully passed

---

## Troubleshooting

### Issue: "No module named 'omicverse'"

```bash
# Solution: Install omicverse
pip install omicverse
# Or from repo:
pip install -e .
```

### Issue: "No module named 'scanpy'"

```bash
# Solution: Install scanpy
pip install scanpy
```

### Issue: "OPENAI_API_KEY: Not set"

```bash
# Solution: Export your API key
export OPENAI_API_KEY="your-key-here"

# Verify it's set:
echo $OPENAI_API_KEY
```

### Issue: "Could not load PBMC3k data"

```bash
# Option 1: Set local path (if you have the file)
export PBMC3K_PATH="/path/to/pbmc3k.h5ad"

# Option 2: Let scanpy download it (requires internet)
# The script will automatically download pbmc3k on first run

# Option 3: Use fallback dataset
# The script automatically tries pbmc68k_reduced if pbmc3k fails
```

### Issue: API Rate Limits

If you hit rate limits:

```bash
# Edit the script to test fewer models
# Comment out models in OPENAI_MODELS or GEMINI_MODELS lists

# For example, to only test GPT-4o-mini:
OPENAI_MODELS = [
    # 'gpt-5',  # Commented out
    'gpt-4o-mini',
]
```

### Issue: Tests Failing

Check the error messages in:
1. Console output
2. `test_results_*.txt` file

Common causes:
- Invalid API key
- Network issues
- Insufficient API credits
- Model not available

---

## Advanced Usage

### Test Specific Models Only

Edit `test_ovagent_openai_gemini.py`:

```python
# Test only one OpenAI model
OPENAI_MODELS = [
    'gpt-4o-mini',  # Comment out others
]

# Test only one Gemini model
GEMINI_MODELS = [
    'gemini/gemini-2.5-flash',  # Comment out others
]
```

### Save Detailed Logs

```bash
# Redirect all output to a log file
python test_ovagent_openai_gemini.py 2>&1 | tee test_execution.log
```

### Run in Background

```bash
# Run tests in background
nohup python test_ovagent_openai_gemini.py > test_output.log 2>&1 &

# Check progress
tail -f test_output.log

# Check if still running
ps aux | grep test_ovagent
```

### Custom PBMC3k Data

```bash
# If you have a local PBMC3k file
export PBMC3K_PATH="/path/to/your/pbmc3k.h5ad"

# Then run tests
python test_ovagent_openai_gemini.py
```

---

## Cost Estimation

Approximate API costs per full test run:

**OpenAI:**
- GPT-4o-mini: ~$0.10-0.20 per test run
- GPT-5: ~$1.00-2.00 per test run (estimated)

**Gemini:**
- Gemini 2.5 Flash: Free tier available
- Gemini 2.5 Pro: Free tier available

**Total for all models**: ~$1.50-3.00

*Note: Costs vary based on token usage. The script is designed to be efficient.*

---

## Understanding the Results

### Pass Criteria

Each test passes if:
- ✅ No exceptions thrown
- ✅ Expected outputs generated
- ✅ Data transformations verified

### What "Success" Means

A model **fully succeeds** if:
- Agent initializes correctly
- All 7 test cases pass
- End-to-end pipeline completes

### Interpreting Failures

Check which test failed:

| Test | Likely Cause |
|------|--------------|
| Initialization | API key issue, model unavailable |
| Quality Control | Data loading issue, code generation problem |
| Preprocessing | Missing dependencies, incompatible data |
| Clustering | Preprocessing didn't complete |
| UMAP | Clustering failed, missing neighbors |
| Semantic Matching | LLM understanding issue |
| End-to-End | Any of the above |

---

## After Testing

### Review Results

```bash
# View the results file
cat test_results_*.txt

# Or with formatting
less test_results_*.txt
```

### Share Results

If reporting issues or sharing results:

1. **Results file**: `test_results_YYYYMMDD_HHMMSS.txt`
2. **Full log** (if saved): `test_execution.log`
3. **Error details** from console output

### Next Steps

Based on results:

✅ **All tests passed**: OVAgent working correctly with your providers!

⚠️ **Some tests failed**:
- Check error messages
- Verify API keys are correct
- Ensure models are available
- Check for rate limits

❌ **All tests failed**:
- Verify omicverse installation: `python -c "import omicverse as ov; print(ov.__version__)"`
- Check API connectivity
- Try with a single simple test first

---

## Quick Reference Card

```bash
# === QUICK START ===

# 1. Set API keys
export OPENAI_API_KEY="your-key"
export GEMINI_API_KEY="your-key"

# 2. Run tests
python3 test_ovagent_openai_gemini.py

# 3. Check results
cat test_results_*.txt

# === THAT'S IT! ===
```

---

## Support

If you encounter issues:

1. **Check Prerequisites** section above
2. **Review Troubleshooting** section
3. **Check error messages** in output
4. **Verify API keys** are valid and have credits
5. **Test connectivity** to OpenAI/Google APIs

For OmicVerse-specific issues:
- GitHub: https://github.com/Starlitnightly/omicverse
- Documentation: Check `OVAGENT_PBMC3K_TESTING_PLAN.md`

---

**Good luck with your testing!** 🚀

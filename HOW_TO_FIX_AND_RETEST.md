# How to Fix the Prompt Length Bug and Re-run Tests

**Issue**: All tests failed with `user_prompt too long (124017 chars, max 100000)`

**Solution**: Apply quick patch to increase limit, then re-run tests

---

## Option 1: Apply Patch File (Recommended)

```bash
# Navigate to omicverse directory
cd /path/to/omicverse

# Apply the patch
patch -p1 < QUICK_FIX_PROMPT_LENGTH.patch

# Verify the change
grep "max 200000" omicverse/utils/agent_backend.py
# Should show two lines with the new limit

# Re-run tests
python3 test_ovagent_openai_gemini.py
```

---

## Option 2: Manual Edit

### Step 1: Open the file

```bash
# Edit agent_backend.py
nano omicverse/utils/agent_backend.py
# or
vim omicverse/utils/agent_backend.py
# or use your favorite editor
```

### Step 2: Make changes

**Find and change Line 331** (approximately):
```python
# OLD (line 331):
if len(user_prompt) > 100000:
    raise ValueError(
        f"user_prompt too long ({len(user_prompt)} chars, max 100000)"
    )

# NEW:
if len(user_prompt) > 200000:
    raise ValueError(
        f"user_prompt too long ({len(user_prompt)} chars, max 200000)"
    )
```

**Find and change Line 373** (approximately):
```python
# OLD (line 373):
if len(user_prompt) > 100000:
    raise ValueError(
        f"user_prompt too long ({len(user_prompt)} chars, max 100000)"
    )

# NEW:
if len(user_prompt) > 200000:
    raise ValueError(
        f"user_prompt too long ({len(user_prompt)} chars, max 200000)"
    )
```

### Step 3: Save and verify

```bash
# Verify changes
grep -n "max 200000" omicverse/utils/agent_backend.py

# Should output something like:
# 333:    f"user_prompt too long ({len(user_prompt)} chars, max 200000)"
# 375:    f"user_prompt too long ({len(user_prompt)} chars, max 200000)"
```

---

## Option 3: Use Python to Make the Change

```bash
# Run this Python script to auto-patch
python3 << 'EOF'
import re

file_path = 'omicverse/utils/agent_backend.py'

# Read file
with open(file_path, 'r') as f:
    content = f.read()

# Replace 100000 with 200000 in the specific error messages
content = content.replace('> 100000:', '> 200000:')
content = content.replace('max 100000)', 'max 200000)')

# Write back
with open(file_path, 'w') as f:
    f.write(content)

print("✅ Patched successfully!")
print("   Changed limit from 100,000 to 200,000 characters")
EOF

# Verify
grep -n "200000" omicverse/utils/agent_backend.py
```

---

## Re-run Tests After Fixing

### Full Test Suite

```bash
# Set API keys (if not already set)
export OPENAI_API_KEY="your-key"
export GEMINI_API_KEY="your-key"

# Run comprehensive tests
python3 test_ovagent_openai_gemini.py
```

**Expected Results After Fix**:
```
Overall Test Results:
  Total Tests Run:    28
  Tests Passed:       28 ✅
  Tests Failed:       0 ❌
  Pass Rate:          100.0%

🎉 ALL SUCCESS CRITERIA MET!
```

### Simple Test (Faster)

```bash
# Run simplified test (just checks if fix works)
python3 test_ovagent_simple.py
```

---

## Verify Fix is Working

### Quick Test in Python

```python
import omicverse as ov
import scanpy as sc
import os

# Initialize agent
agent = ov.Agent(
    model='gpt-4o-mini',
    api_key=os.getenv('OPENAI_API_KEY')
)

# Load data
adata = sc.datasets.pbmc3k()

# Try a simple request
result = agent.run('quality control with nUMI>500, mito<0.2', adata)

print(f"✅ Fix working! Cells: {adata.n_obs} → {result.n_obs}")
```

If this runs without the `user_prompt too long` error, the fix is working!

---

## Expected Test Timing After Fix

- **OpenAI GPT-4o-mini**: ~3-5 minutes per model
- **OpenAI GPT-5**: ~5-7 minutes per model
- **Gemini Flash**: ~3-5 minutes per model
- **Gemini Pro**: ~5-7 minutes per model

**Total for all 4 models**: ~15-30 minutes

---

## Troubleshooting

### Issue: Patch command not found

```bash
# Install patch utility
# macOS:
brew install patch

# Linux:
sudo apt-get install patch  # Debian/Ubuntu
sudo yum install patch      # CentOS/RHEL
```

### Issue: Patch fails

Use **Option 2** (Manual Edit) or **Option 3** (Python script) instead.

### Issue: Still getting "prompt too long" error

Check that you changed BOTH occurrences:

```bash
# Should show 2 results
grep -c "max 200000" omicverse/utils/agent_backend.py

# Should output: 2
```

If it shows less than 2, find and fix the remaining occurrence.

### Issue: File not found

Make sure you're in the correct directory:

```bash
# Check current directory
pwd

# Should end with: .../omicverse

# List files
ls omicverse/utils/agent_backend.py

# Should show the file
```

---

## After Successful Testing

### Share Results

If all tests pass, you can share:

1. **Results file**: `test_results_YYYYMMDD_HHMMSS.txt`
2. **Confirmation**: "Tests passed after applying prompt length fix"

### Submit Bug Fix (Optional)

If you want to contribute the fix back:

```bash
# Create a branch
git checkout -b fix/prompt-length-limit

# Commit the fix
git add omicverse/utils/agent_backend.py
git commit -m "Fix: Increase prompt length limit to 200K chars

The function registry generates ~124K character prompts,
which exceeded the previous 100K limit. This caused all
agent requests to fail.

Quick fix: Increase limit to 200K chars.
Long-term: Implement progressive function loading.

Fixes #XXX"

# Push and create PR
git push origin fix/prompt-length-limit
```

---

## Important Notes

### This is a Quick Fix

⚠️ **This quick fix increases the limit but doesn't address the root cause.**

**For long-term solution**, the omicverse team should:
1. Implement progressive function loading (like skills)
2. Reduce function registry size in prompts
3. Use selective function matching

See `BUG_REPORT_PROMPT_LENGTH.md` for detailed recommendations.

### Token Costs

Increasing the prompt size from 100K to 200K chars will increase token costs slightly:
- ~100K chars ≈ ~25K tokens
- ~124K chars ≈ ~31K tokens (current usage)

**Cost impact**: Minimal (~20% increase in prompt tokens per request)

---

## Summary

**Quick Steps**:
1. Apply patch or manually edit `agent_backend.py`
2. Change `100000` to `200000` in two places (lines ~331 and ~373)
3. Re-run: `python3 test_ovagent_openai_gemini.py`
4. Expect all tests to pass ✅

**Questions?** See `BUG_REPORT_PROMPT_LENGTH.md` for detailed analysis.

---

Good luck! 🚀

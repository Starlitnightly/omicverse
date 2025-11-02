---
name: data-export-excel
title: Excel Data Export (Universal)
description: Export analysis results, data tables, and formatted spreadsheets to Excel files using openpyxl. Works with ANY LLM provider (GPT, Gemini, Claude, etc.).
---

# Excel Data Export (Universal)

## Overview
This skill enables you to export bioinformatics data, analysis results, and formatted tables to professional Excel spreadsheets. Unlike cloud-hosted solutions, this skill uses the **openpyxl** Python library and executes **locally** in your environment, making it compatible with **ALL LLM providers** including GPT, Gemini, Claude, DeepSeek, and Qwen.

## When to Use This Skill
- Export AnnData observations (.obs) or variables (.var) to Excel
- Save DEG analysis results with formatting
- Create multi-sheet workbooks with different data types
- Generate formatted Excel reports with cell styling
- Export cluster annotations, cell type assignments, or quality control metrics

## How to Use

### Step 1: Import Required Libraries
```python
import openpyxl
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
import pandas as pd
import numpy as np
```

### Step 2: Prepare Your Data
Convert your data to pandas DataFrame format:
```python
# Example: Export AnnData observations
df = adata.obs.copy()

# Example: Export DEG results
deg_df = pd.DataFrame({
    'gene': gene_names,
    'log2FC': log2_fold_changes,
    'pvalue': pvalues,
    'qvalue': qvalues
})

# Example: Export cluster statistics
cluster_stats = adata.obs.groupby('clusters').size().reset_index(name='count')
```

### Step 3: Create Excel Workbook
```python
# Create new workbook
wb = Workbook()
ws = wb.active
ws.title = "Sheet Name"

# Write DataFrame to worksheet
for r in dataframe_to_rows(df, index=False, header=True):
    ws.append(r)
```

### Step 4: Add Formatting (Optional)
```python
# Style header row
header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
header_font = Font(bold=True, color="FFFFFF")

for cell in ws[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal='center')

# Auto-adjust column widths
for column in ws.columns:
    max_length = 0
    column_letter = column[0].column_letter
    for cell in column:
        try:
            if len(str(cell.value)) > max_length:
                max_length = len(str(cell.value))
        except:
            pass
    adjusted_width = min(max_length + 2, 50)
    ws.column_dimensions[column_letter].width = adjusted_width

# Add borders
thin_border = Border(
    left=Side(style='thin'),
    right=Side(style='thin'),
    top=Side(style='thin'),
    bottom=Side(style='thin')
)
for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
    for cell in row:
        cell.border = thin_border
```

### Step 5: Save the Workbook
```python
# Save to file
output_path = "analysis_results.xlsx"
wb.save(output_path)
print(f"✅ Excel file saved to: {output_path}")
```

## Multi-Sheet Workbooks

Create workbooks with multiple sheets for different data types:

```python
wb = Workbook()

# Sheet 1: Cell metadata
ws1 = wb.active
ws1.title = "Cell Metadata"
for r in dataframe_to_rows(adata.obs, index=True, header=True):
    ws1.append(r)

# Sheet 2: Gene metadata
ws2 = wb.create_sheet("Gene Metadata")
for r in dataframe_to_rows(adata.var, index=True, header=True):
    ws2.append(r)

# Sheet 3: DEG results
ws3 = wb.create_sheet("DEG Results")
for r in dataframe_to_rows(deg_df, index=False, header=True):
    ws3.append(r)

wb.save("multi_sheet_analysis.xlsx")
```

## Best Practices

1. **Column Headers**: Always include column headers in the first row
2. **Data Types**: Convert numpy arrays to lists before writing
3. **Large Datasets**: For datasets >100K rows, consider CSV export instead
4. **File Paths**: Use absolute paths or ensure output directory exists
5. **Formatting**: Apply formatting sparingly to reduce file size
6. **Index**: Decide whether to include DataFrame index (set `index=True/False` in `dataframe_to_rows`)

## Common Use Cases

### Export Quality Control Metrics
```python
qc_metrics = adata.obs[['n_genes', 'n_counts', 'percent_mito', 'clusters']].copy()

wb = Workbook()
ws = wb.active
ws.title = "QC Metrics"

for r in dataframe_to_rows(qc_metrics, index=False, header=True):
    ws.append(r)

# Highlight cells with high mitochondrial content
for row in range(2, ws.max_row + 1):
    if ws.cell(row, 3).value > 0.2:  # percent_mito > 20%
        ws.cell(row, 3).fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")

wb.save("qc_metrics.xlsx")
```

### Export Marker Genes by Cluster
```python
# Assuming you have marker genes for each cluster
marker_dict = {
    'Cluster_0': ['CD3D', 'CD3E', 'CD8A'],
    'Cluster_1': ['CD79A', 'MS4A1', 'CD19'],
    'Cluster_2': ['LYZ', 'S100A9', 'CD14']
}

wb = Workbook()

for cluster_name, genes in marker_dict.items():
    ws = wb.create_sheet(cluster_name)
    ws.append(['Marker Gene'])
    for gene in genes:
        ws.append([gene])

# Remove default sheet
if 'Sheet' in wb.sheetnames:
    wb.remove(wb['Sheet'])

wb.save("marker_genes.xlsx")
```

### Export DEG Analysis with Conditional Formatting
```python
wb = Workbook()
ws = wb.active
ws.title = "DEG Analysis"

# Write DEG results
for r in dataframe_to_rows(deg_df, index=False, header=True):
    ws.append(r)

# Color code by fold change
for row in range(2, ws.max_row + 1):
    log2fc = ws.cell(row, 2).value  # Assuming log2FC in column 2
    if log2fc > 1:  # Upregulated
        ws.cell(row, 2).fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    elif log2fc < -1:  # Downregulated
        ws.cell(row, 2).fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")

wb.save("deg_results_formatted.xlsx")
```

## Troubleshooting

### Issue: "openpyxl not found"
**Solution**: Install the library:
```python
import subprocess
subprocess.check_call(['pip', 'install', 'openpyxl'])
```

### Issue: "Invalid data type for cell"
**Solution**: Convert numpy/pandas types to native Python types:
```python
# Convert numpy types
df = df.astype(object).where(pd.notnull(df), None)

# Or convert specific columns
df['column_name'] = df['column_name'].astype(str)
```

### Issue: "Memory error with large datasets"
**Solution**: Export in chunks or use CSV format instead:
```python
# Fallback to CSV for large data
df.to_csv('large_dataset.csv', index=False)
print("Dataset too large for Excel, saved as CSV instead")
```

## Technical Notes

- **Library**: Uses `openpyxl` (pure Python, no external dependencies)
- **Execution**: Runs locally in the agent's sandbox
- **Compatibility**: Works with ALL LLM providers (GPT, Gemini, Claude, DeepSeek, Qwen, etc.)
- **File Limits**: Excel has a 1,048,576 row limit (use CSV for larger datasets)
- **Performance**: Writing ~10K rows takes 1-2 seconds; 100K rows takes 10-20 seconds

## References
- openpyxl documentation: https://openpyxl.readthedocs.io/
- pandas DataFrame export: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_excel.html

---
name: data-export-pdf
title: PDF Report Generation (Universal)
description: Create professional PDF reports with text, tables, and embedded images using reportlab. Works with ANY LLM provider (GPT, Gemini, Claude, etc.).
---

# PDF Report Generation (Universal)

## Overview
This skill enables you to create professional PDF reports containing analysis summaries, formatted tables, and embedded visualizations. Unlike cloud-hosted solutions, this skill uses the **reportlab** Python library and executes **locally** in your environment, making it compatible with **ALL LLM providers** including GPT, Gemini, Claude, DeepSeek, and Qwen.

## When to Use This Skill
- Generate analysis reports with text and tables
- Create summary PDFs with embedded plots
- Export formatted documentation
- Produce publication-ready supplementary materials
- Combine multiple analysis results into a single document

## How to Use

### Step 1: Import Required Libraries
```python
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
import matplotlib.pyplot as plt
```

### Step 2: Create Basic PDF Document
```python
# Create PDF file
pdf_filename = "analysis_report.pdf"
doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
story = []  # Container for PDF elements

# Get default styles
styles = getSampleStyleSheet()
title_style = styles['Title']
heading_style = styles['Heading1']
normal_style = styles['Normal']

# Add title
story.append(Paragraph("Analysis Report", title_style))
story.append(Spacer(1, 0.2*inch))

# Add date
date_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
story.append(Paragraph(date_text, normal_style))
story.append(Spacer(1, 0.3*inch))

# Build PDF
doc.build(story)
print(f"✅ PDF saved to: {pdf_filename}")
```

### Step 3: Add Text Content
```python
story = []

# Title
story.append(Paragraph("Single-Cell RNA-seq Analysis Report", title_style))
story.append(Spacer(1, 0.2*inch))

# Section heading
story.append(Paragraph("1. Overview", heading_style))
story.append(Spacer(1, 0.1*inch))

# Paragraph text
overview_text = """
This report summarizes the single-cell RNA-seq analysis performed on the dataset.
The analysis includes quality control, normalization, dimensionality reduction,
clustering, and cell type annotation.
"""
story.append(Paragraph(overview_text, normal_style))
story.append(Spacer(1, 0.2*inch))
```

### Step 4: Add Tables
```python
# Prepare table data
table_data = [
    ['Metric', 'Value'],  # Header
    ['Total Cells', '5,000'],
    ['Total Genes', '20,000'],
    ['Mean Genes/Cell', '2,500'],
    ['Median UMIs/Cell', '10,000']
]

# Create table
table = Table(table_data, colWidths=[2.5*inch, 2*inch])

# Style table
table.setStyle(TableStyle([
    # Header styling
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 12),

    # Body styling
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
    ('FONTSIZE', (0, 1), (-1, -1), 10),
]))

story.append(table)
story.append(Spacer(1, 0.3*inch))
```

### Step 5: Embed Images/Plots
```python
# Save matplotlib figure first
fig, ax = plt.subplots(figsize=(6, 4))
# ... create your plot ...
plot_filename = "temp_plot.png"
fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.close(fig)

# Add image to PDF
story.append(Paragraph("2. UMAP Visualization", heading_style))
story.append(Spacer(1, 0.1*inch))
img = Image(plot_filename, width=4*inch, height=3*inch)
story.append(img)
story.append(Spacer(1, 0.2*inch))
```

## Complete Example: Analysis Report

```python
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

def create_analysis_report(adata, output_path="analysis_report.pdf"):
    """Create comprehensive PDF analysis report"""

    # Initialize PDF
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Title
    story.append(Paragraph("Single-Cell RNA-seq Analysis Report", styles['Title']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    # Overview
    story.append(Paragraph("1. Dataset Overview", styles['Heading1']))
    story.append(Spacer(1, 0.1*inch))

    overview_data = [
        ['Metric', 'Value'],
        ['Total Cells', f'{adata.n_obs:,}'],
        ['Total Genes', f'{adata.n_vars:,}'],
        ['Observations', ', '.join(adata.obs.columns[:5].tolist())],
    ]

    table = Table(overview_data, colWidths=[2.5*inch, 3.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3*inch))

    # Cluster distribution
    if 'clusters' in adata.obs:
        story.append(Paragraph("2. Cluster Distribution", styles['Heading1']))
        story.append(Spacer(1, 0.1*inch))

        cluster_counts = adata.obs['clusters'].value_counts().sort_index()
        cluster_data = [['Cluster', 'Cell Count', 'Percentage']]
        total_cells = adata.n_obs

        for cluster, count in cluster_counts.items():
            percentage = (count / total_cells) * 100
            cluster_data.append([str(cluster), str(count), f'{percentage:.1f}%'])

        table = Table(cluster_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3*inch))

    # Visualization (if UMAP exists)
    if 'X_umap' in adata.obsm:
        story.append(Paragraph("3. UMAP Visualization", styles['Heading1']))
        story.append(Spacer(1, 0.1*inch))

        # Create UMAP plot
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(
            adata.obsm['X_umap'][:, 0],
            adata.obsm['X_umap'][:, 1],
            c=adata.obs['clusters'].astype('category').cat.codes if 'clusters' in adata.obs else 'blue',
            s=5, alpha=0.5
        )
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_title('UMAP Projection')

        plot_path = 'temp_umap.png'
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        img = Image(plot_path, width=5*inch, height=4*inch)
        story.append(img)

    # Build PDF
    doc.build(story)
    print(f"✅ PDF report saved to: {output_path}")

    return output_path

# Usage
create_analysis_report(adata, "my_analysis_report.pdf")
```

## Best Practices

1. **Page Size**: Use `letter` (US) or `A4` (international) for standard documents
2. **Margins**: SimpleDocTemplate has default margins (1 inch); adjust with `leftMargin`, `rightMargin`, etc.
3. **Images**: Save matplotlib figures at 150-300 DPI for good quality
4. **Tables**: Keep column counts reasonable (4-6 columns max for readability)
5. **File Cleanup**: Delete temporary image files after PDF creation
6. **Memory**: For large documents, build in sections to manage memory

## Advanced Features

### Custom Page Header/Footer
```python
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def add_header_footer(canvas_obj, doc):
    canvas_obj.saveState()
    # Header
    canvas_obj.setFont('Helvetica', 9)
    canvas_obj.drawString(inch, letter[1] - 0.5*inch, "Analysis Report")
    # Footer
    canvas_obj.drawString(inch, 0.5*inch, f"Page {doc.page}")
    canvas_obj.restoreState()

doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
```

### Multi-Column Layout
```python
from reportlab.platypus import Frame, PageTemplate

frame1 = Frame(doc.leftMargin, doc.bottomMargin, doc.width/2-6, doc.height, id='col1')
frame2 = Frame(doc.leftMargin+doc.width/2+6, doc.bottomMargin, doc.width/2-6, doc.height, id='col2')

doc.addPageTemplates([PageTemplate(id='TwoCol', frames=[frame1, frame2])])
```

### Color-Coded Tables
```python
# Highlight significant results
for i, row in enumerate(deg_results):
    if row['qvalue'] < 0.05:
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, i+1), (-1, i+1), colors.yellow)
        ]))
```

## Common Use Cases

### QC Report
```python
qc_metrics = {
    'Total Cells': adata.n_obs,
    'Median Genes/Cell': int(adata.obs['n_genes'].median()),
    'Median UMIs/Cell': int(adata.obs['n_counts'].median()),
    'Mean Mito %': f"{adata.obs['percent_mito'].mean():.2f}%"
}

table_data = [['Metric', 'Value']] + [[k, str(v)] for k, v in qc_metrics.items()]
# ... create table as shown above
```

### DEG Summary Table
```python
# Top 10 upregulated genes
top_genes = deg_df.nlargest(10, 'log2FC')[['gene', 'log2FC', 'qvalue']]
table_data = [['Gene', 'log2FC', 'Q-value']]
for _, row in top_genes.iterrows():
    table_data.append([row['gene'], f"{row['log2FC']:.2f}", f"{row['qvalue']:.2e}"])
```

## Troubleshooting

### Issue: "reportlab not found"
**Solution**:
```python
import subprocess
subprocess.check_call(['pip', 'install', 'reportlab'])
```

### Issue: "Image not found"
**Solution**: Ensure image path is correct and file exists before adding to PDF:
```python
import os
if os.path.exists(plot_filename):
    img = Image(plot_filename, width=4*inch, height=3*inch)
    story.append(img)
```

### Issue: "Table exceeds page width"
**Solution**: Reduce column widths or font size:
```python
table = Table(data, colWidths=[1.5*inch, 1.5*inch, 2*inch])
table.setStyle(TableStyle([('FONTSIZE', (0, 0), (-1, -1), 8)]))
```

## Technical Notes

- **Library**: Uses `reportlab` (pure Python, widely supported)
- **Execution**: Runs locally in the agent's sandbox
- **Compatibility**: Works with ALL LLM providers (GPT, Gemini, Claude, DeepSeek, Qwen, etc.)
- **File Size**: Text-heavy PDFs are small (<1MB); image-heavy PDFs can be 5-20MB
- **Performance**: Typical report generation takes 1-3 seconds

## References
- reportlab documentation: https://www.reportlab.com/docs/reportlab-userguide.pdf
- reportlab platypus guide: https://www.reportlab.com/software/opensource/rl-toolkit/guide/

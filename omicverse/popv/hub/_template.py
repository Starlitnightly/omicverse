template = """\
---
{card_data}
---

Popular Vote (popV) model for automated cell type annotation of single-cell RNA-seq data. We provide here pretrained models
for plug-in use in your own analysis.
Follow our [tutorial](https://github.com/YosefLab/popV/blob/main/tabula_sapiens_tutorial.ipynb) to learn how to use the model for cell type annotation.

# Model description

{description}

**Link to CELLxGENE**:
Link to the [data]({cellxgene_url}) in the CELLxGENE browser for interactive exploration of the data and download of the source data.

**Training Code URL**:
{training_code_url}

# Metrics

We provide here accuracies for each of the experts and the ensemble model. The validation set accuracies are
computed on a 10% random subset of the data that was not used for training.

{validation_accuracies}

The train accuracies are computed on the training data.

{train_accuracies}

</details>
\n
# References

{references}
"""

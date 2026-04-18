#!/usr/bin/env Rscript
# Reproduce MetaboAnalystR's canonical Statistical_Analysis workflow on the
# human_cachexia.csv dataset and dump every intermediate result to TSVs that
# the Python side can compare against for parity testing.
#
# Usage: Rscript metabol_r_reference_driver.R <cachexia_csv> <outdir>
#
# The pipeline mirrors MetaboAnalystR's "Introduction" vignette:
#   ReadPeakList -> PerformDataInspect -> PreparePrenormData ->
#   Normalization(ref='median', norm='QN', trans='LogNorm', scale='ParetoNorm')
#   -> Ttests.Anal -> PLSR.Anal
#
# We stick to base R + MetaboAnalystR so no rpy2/bindings are needed.

suppressPackageStartupMessages({
  library(MetaboAnalystR)
})

args <- commandArgs(trailingOnly = TRUE)
csv_path <- args[[1]]
outdir   <- args[[2]]
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# 1) Create mSet, load CSV (rowu = rows are samples, disc = discrete factor)
mSet <- InitDataObjects("conc", "stat", FALSE)
mSet <- Read.TextData(mSet, csv_path, "rowu", "disc")
mSet <- SanityCheckData(mSet)
mSet <- ReplaceMin(mSet)
mSet <- PreparePrenormData(mSet)

# 2) MetaboAnalyst canonical: PQN ("MedianNorm" == row median here? Use QN for PQN)
mSet <- Normalization(mSet,
                      rowNorm  = "QuantileNorm",     # PQN
                      transNorm = "LogNorm",          # log2
                      scaleNorm = "ParetoNorm",       # pareto scaling
                      ref = NULL, ratio = FALSE,
                      ratioNum = 20)

# Write the normalized matrix — this is the input to differential/multivariate
norm_mat <- as.data.frame(mSet$dataSet$norm)
write.table(norm_mat, file.path(outdir, "norm_matrix.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)

# 3) Univariate t-test
mSet <- Ttests.Anal(mSet, nonpar = FALSE, threshp = 1.0,
                    paired = FALSE, equal.var = FALSE, pvalType = "fdr")
tt <- mSet$analSet$tt
t_table <- data.frame(metabolite = names(tt$p.value),
                      pvalue = tt$p.value,
                      padj = tt$p.adj,
                      t_stat = tt$t.stat)
write.table(t_table, file.path(outdir, "ttest.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# 4) PLS-DA
mSet <- PLSR.Anal(mSet, reg = TRUE)
mSet <- PLSDA.CV(mSet, "T", 5, "Q2")
scores <- as.data.frame(mSet$analSet$plsr$scores[, 1:2])
colnames(scores) <- c("pls_t1", "pls_t2")
scores$sample <- rownames(scores)
write.table(scores, file.path(outdir, "plsda_scores.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# VIP
vip <- as.data.frame(mSet$analSet$plsda$vip.mat)
vip$metabolite <- rownames(vip)
write.table(vip, file.path(outdir, "plsda_vip.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# Quality metrics
cv <- mSet$analSet$plsr$Q2
write.table(data.frame(metric = names(cv), value = cv),
            file.path(outdir, "plsda_q2.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

cat(sprintf("[metabol-R] wrote 4 TSVs to %s\n", outdir))

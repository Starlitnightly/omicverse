library(MAST)
library(parallel)
if(!require(data.table)) {
  install.packages(data.table, dependencies = TRUE)
}
library(data.table)

args <- commandArgs(trailingOnly = TRUE)
out_dir <- args[1]
split_num <- args[2]

# set up parallel backend
options(mc.cores = detectCores() - 1)

## Load data
exp_data <- read.csv(paste0(out_dir, '/exp_normalized.csv'), header = TRUE, row.names = 1)
exp_data <- t(exp_data)
pseudotime_lineages <- read.csv(paste0(out_dir, '/pseudotime_lineages.csv'), header = TRUE, row.names = 1)

## MAST
# MAST assumes that log-transformed approximately scale-normalized data is provided (such as log2(TPM+1))
split_num <- 4
for(p in colnames(pseudotime_lineages))
{
  lin <- pseudotime_lineages[, p]
  names(lin) <- row.names(pseudotime_lineages)
  lin <- lin[!is.na(lin)]
  lin <- sort(lin, decreasing=F)

  # split K groups based on pseudo-time
  len_per_group <- round(length(lin)/split_num)
  cell_groups <- c()
  for(i in 1:(split_num-1)) cell_groups <- c(cell_groups, rep(i,len_per_group))
  cell_groups <- c(cell_groups, rep(split_num, length(lin)-length(cell_groups)))
  names(cell_groups) <- names(lin)

  # expression data: gene * cell
  expData_p <- exp_data[, names(lin)]
  fData <- data.frame(names=rownames(expData_p), primerid=rownames(expData_p))
  rownames(fData) <- rownames(expData_p)
  cData <- data.frame(cond=cell_groups, wellKey=colnames(expData_p))
  rownames(cData) <- colnames(expData_p)

  # These are log(TPM/FPKM+1) log2(assay(dat, "TPM") + 1)
  sca <- FromMatrix(as.matrix(expData_p), cData, fData)
  colData(sca)$cdr <- scale(colSums(assay(sca)>0)) # cell detection rate
  cond <- factor(colData(sca)$cond)
  cond <- relevel(cond, 1) # set the reference level of the factor to be group 1 cells
  colData(sca)$condition <- cond

  # DE genes between the average expression of the start and end points of a lineage
  zlmdata <- zlm(~ condition + cdr, sca)
  avgLogFC <- 0
  for(i in seq(2,split_num))
  {
    end_group <- paste0('condition', i)
    summaryCond <- summary(zlmdata, doLRT=end_group) # test group i ('condition i')

    #
    summaryDt <- summaryCond$datatable
    fcHurdle <- merge(summaryDt[contrast==end_group & component=='H',.(primerid, `Pr(>Chisq)`)], #hurdle P values
                      summaryDt[contrast==end_group & component=='logFC', .(primerid, coef, ci.hi, ci.lo)], by='primerid') #logFC coefficients
    fcHurdle[,fdr:=p.adjust(`Pr(>Chisq)`, 'fdr')]
    fcHurdle <- merge(fcHurdle, as.data.table(mcols(sca)), by='primerid')
    setnames(fcHurdle,'coef','logFC')
    fcHurdle <- fcHurdle[,-7]

    fcHurdle[fdr>0.01, 'logFC'] <- 0 # set logFC to 0 if fdr>0.01
    avgLogFC <- avgLogFC + abs(fcHurdle$logFC)
  }
  avgLogFC <- avgLogFC/(split_num-1)
  avgLogFC <- as.data.frame(avgLogFC, row.names=fcHurdle$primerid)
  avgLogFC$primerid <- row.names(avgLogFC)
  logFCAll <- as.data.frame(logFC(zlmdata)$logFC)
  logFCAll$primerid <- row.names(logFCAll)
  logFCAll <- merge(logFCAll, avgLogFC, by='primerid')
  # Delete the line containing NA
  NA_row <- unique(which(is.na(logFCAll), arr.ind=T)[, 1])
  if(length(NA_row)>0) logFCAll <- logFCAll[-NA_row, ]
  # Delete the line containing 0
  logFCAll <- logFCAll[logFCAll$avgLogFC!=0, ]

  setorder(logFCAll, -avgLogFC)
  setnames(logFCAll, 'avgLogFC', 'logFC')

  # write to file
  fwrite(logFCAll, paste0(out_dir, '/DEgenes_MAST_sp', split_num, '_', p, '.csv'))
}

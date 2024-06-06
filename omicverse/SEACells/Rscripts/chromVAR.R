library(chromVAR)
library(SummarizedExperiment)
library(Matrix)

library(BSgenome.Hsapiens.UCSC.hg38)

library(motifmatchr)
library(chromVARmotifs)

# parse arguments
argReader = commandArgs(trailingOnly=TRUE)
base_dir <- argReader[1]

print(paste0("Loading files (peaks.bed, sampling_depth.txt, counts.txt, peak_names.txt and cell_names.txt) from base directory ",base_dir))

peaks <- getPeaks(paste0(base_dir,"peaks.bed"), sort_peaks = TRUE)

sampling_depth <- as.matrix(read.delim(paste0(base_dir, 'sampling_depth.txt'), header=F))
sampling_depth <- as.numeric(sampling_depth)

print("Loading counts...")

my_counts_matrix <- as.matrix(read.delim(paste0(base_dir, 'counts.txt'), header=F))

row_names <- as.matrix(read.delim(paste0(base_dir, 'peak_names.txt'), header=F))
rownames(my_counts_matrix) <- row_names

col_names <- as.matrix(read.delim(paste0(base_dir, 'cell_names.txt'), header=F))
colnames(my_counts_matrix) <- col_names

print("Creating SummarizedExperiment...")

fragment_counts <- SummarizedExperiment(assays =  list(counts = my_counts_matrix),
                                        rowRanges = peaks,
                                        colData = sampling_depth)

colnames(colData(fragment_counts)) <- c('depth')

fragment_counts

rm(my_counts_matrix)
rm(peaks)
rm(sampling_depth)
rm(row_names)


fragment_counts <- addGCBias(fragment_counts,
                             genome = BSgenome.Hsapiens.UCSC.hg38)

print("Matching motifs...")

data('human_pwms_v2')
motifs = human_pwms_v2
motif_ix <- matchMotifs(motifs, fragment_counts, genome = BSgenome.Hsapiens.UCSC.hg38)

rm(motifs)

print("Computing deviations...")

dev <- computeDeviations(object = fragment_counts, annotations = motif_ix)
variability <- computeVariability(dev)

write.csv(deviations(dev), paste0(base_dir,"deviations.csv"))
write.csv(deviationScores(dev) ,paste0(base_dir,"deviationScores.csv"))
write.csv(variability ,paste0(base_dir,"variability.csv"))

print("Finished writing files.")

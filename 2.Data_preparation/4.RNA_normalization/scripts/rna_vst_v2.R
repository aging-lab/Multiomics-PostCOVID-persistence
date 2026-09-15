#!/usr/bin/env Rscript
# DESeq2 VST normalization (v2, replaces log2CPM for Path B normative modeling)
#
# Input : rna_combat_corrected_v2.tsv   — ComBat-seq v2 corrected counts
#         sample_info_v2.tsv
# Output: rna_vst_v2.tsv.gz             — VST matrix (genes × samples)
#         rna_gene_filter_vst_v2.tsv    — gene filter log
#         figures/pca_vst_v2.png

.libPaths(c("~/.R/library", .libPaths()))
suppressPackageStartupMessages({
  library(DESeq2)
  library(ggplot2)
})

# Resolve BASE from script path (R 4.1 compatible)
args       <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("--file=", args, value = TRUE)
if (length(script_arg) > 0) {
  BASE <- normalizePath(file.path(dirname(sub("--file=", "", script_arg)), ".."))
} else {
  BASE <- normalizePath(file.path(getwd(), ".."))
}

IN_DIR  <- file.path(dirname(BASE), "2.RNA_batch_correction", "data")
DATA    <- file.path(BASE, "data")
FIGS    <- file.path(BASE, "figures")
dir.create(DATA, showWarnings = FALSE, recursive = TRUE)
dir.create(FIGS, showWarnings = FALSE, recursive = TRUE)

CPM_CUTOFF <- 1
MIN_FRAC   <- 0.50

# ── 1. Load ───────────────────────────────────────────────────────────────────
message("Loading ComBat-seq v2 corrected counts...")
counts <- read.table(file.path(IN_DIR, "rna_combat_corrected_v2.tsv"),
                     sep = "\t", header = TRUE, row.names = 1,
                     check.names = FALSE)
counts <- round(as.matrix(counts))   # DESeq2 requires integer matrix
storage.mode(counts) <- "integer"

sample_info <- read.table(file.path(IN_DIR, "sample_info_v2.tsv"),
                           sep = "\t", header = TRUE, row.names = "SampleID")
sample_info <- sample_info[colnames(counts), , drop = FALSE]

message(sprintf("  Genes: %d,  Samples: %d", nrow(counts), ncol(counts)))
message("  Groups: ", paste(names(table(sample_info$Group)),
                             table(sample_info$Group), sep = "=", collapse = ", "))

# ── 2. CPM filter (same criterion as log2CPM step) ───────────────────────────
message("\nApplying CPM filter (CPM > 1 in >= 50% of smallest group)...")
lib_sizes <- colSums(counts)
cpm       <- sweep(counts, 2, lib_sizes, "/") * 1e6

group_sizes <- table(sample_info$Group)
smallest_n  <- min(group_sizes)
min_samples <- ceiling(MIN_FRAC * smallest_n)
message(sprintf("  Smallest group: n=%d -> min %d samples", smallest_n, min_samples))

pass <- rowSums(cpm > CPM_CUTOFF) >= min_samples
message(sprintf("  Genes passing: %d / %d", sum(pass), nrow(counts)))

gene_log <- data.frame(
  gene        = rownames(counts),
  mean_count  = rowMeans(counts),
  mean_cpm    = rowMeans(cpm),
  pass_filter = pass
)
write.table(gene_log, file.path(DATA, "rna_gene_filter_vst_v2.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

counts_filt <- counts[pass, ]

# ── 3. DESeq2 VST ─────────────────────────────────────────────────────────────
message("\nRunning DESeq2 VST (blind = FALSE, design = ~Group)...")
col_data <- data.frame(
  Group = factor(sample_info$Group),
  row.names = rownames(sample_info)
)

dds <- DESeqDataSetFromMatrix(
  countData = counts_filt,
  colData   = col_data,
  design    = ~ Group
)
dds <- estimateSizeFactors(dds)

# VST: blind=FALSE uses within-group dispersion → preserves biological variation
vsd <- vst(dds, blind = FALSE)
vst_mat <- assay(vsd)     # genes × samples, log2-scale stabilized
message(sprintf("  Done. VST matrix: %d genes x %d samples", nrow(vst_mat), ncol(vst_mat)))

# ── 4. Save ───────────────────────────────────────────────────────────────────
out_path <- file.path(DATA, "rna_vst_v2.tsv.gz")
message(sprintf("\nSaving -> %s", out_path))
gz <- gzfile(out_path, "w")
write.table(vst_mat, gz, sep = "\t", quote = FALSE, col.names = NA)
close(gz)

# ── 5. PCA ────────────────────────────────────────────────────────────────────
message("Running PCA (top 5,000 variable genes)...")
rv      <- apply(vst_mat, 1, var)
top5k   <- order(rv, decreasing = TRUE)[seq_len(min(5000, length(rv)))]
pca_res <- prcomp(t(vst_mat[top5k, ]), center = TRUE, scale. = FALSE)
var_exp <- summary(pca_res)$importance[2, 1:5] * 100

pca_df <- as.data.frame(pca_res$x[, 1:3])
pca_df$Group <- sample_info[rownames(pca_df), "Group"]
message(sprintf("  PC1: %.1f%%  PC2: %.1f%%  PC3: %.1f%%",
                var_exp[1], var_exp[2], var_exp[3]))

pal <- c(AC = "#E63946", RC = "#F4A261", HC = "#457B9D")

p <- ggplot(pca_df, aes(PC1, PC2, color = Group)) +
  geom_point(size = 1.2, alpha = 0.7) +
  scale_color_manual(values = pal) +
  labs(
    title    = "RNA-seq VST v2 — PCA",
    subtitle = sprintf("DESeq2 VST (blind=FALSE) | top 5,000 variable genes\nPC1: %.1f%%  PC2: %.1f%%",
                       var_exp[1], var_exp[2]),
    x = sprintf("PC1 (%.1f%%)", var_exp[1]),
    y = sprintf("PC2 (%.1f%%)", var_exp[2])
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "right",
        plot.subtitle = element_text(size = 9, color = "grey40"))

ggsave(file.path(FIGS, "pca_vst_v2.png"), p, width = 7, height = 5, dpi = 150)
message("Saved -> figures/pca_vst_v2.png")

# ── 6. Summary ────────────────────────────────────────────────────────────────
message("\nSummary (VST v2):")
message(sprintf("  Genes after filter : %d", nrow(vst_mat)))
message(sprintf("  Samples            : %d", ncol(vst_mat)))
message(sprintf("  PC1: %.1f%%  PC2: %.1f%%", var_exp[1], var_exp[2]))
message("  VST range: [", round(min(vst_mat), 2), ",", round(max(vst_mat), 2), "]")
message("Done.")

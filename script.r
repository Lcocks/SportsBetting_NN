#!/usr/bin/env Rscript

# Load libraries
library(nflreadr)
library(tidyverse)

# ----- Parse command line arguments -----
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Usage: Rscript script.R <stat_type> <season>\nExample: Rscript script.R receiving 2025")
}

stat_type <- args[1]
season <- as.numeric(args[2])

# Validate stat_type
valid_types <- c("passing", "rushing", "receiving")
if (!(stat_type %in% valid_types)) {
  stop(paste("stat_type must be one of:", paste(valid_types, collapse = ", ")))
}

# ----- Load Next Gen Stats data -----
data <- load_nextgen_stats(stat_type = stat_type, seasons = season)

# ----- Save to CSV -----
output_file <- paste0(stat_type, "_data_", season, ".csv")
write.csv(data, output_file, row.names = FALSE)

cat("Saved:", output_file, "\n")
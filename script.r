# Load libraries
library(nflreadr)
library(tidyverse)

# Load Next Gen Stats data
# Available stat types: "passing", "rushing", "receiving"
receiving_data <- load_nextgen_stats(stat_type = "receiving", seasons = 2025)

write.csv(receiving_data, "receiving_data_2025.csv", row.names = FALSE)
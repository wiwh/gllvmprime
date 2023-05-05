library(haven)
library(tidyverse)

# Read stata data

PATH_TO_DATA <- "data/"
PATH_TO_TRACKER <- paste0(PATH_TO_DATA, "TRK2020TR_R.csv")

file <- c("H04D_R.csv")

tbl <- read_csv(paste0(PATH_TO_DATA, file))

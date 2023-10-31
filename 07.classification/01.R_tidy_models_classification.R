library(tidyverse)
library(tidymodels)
library(here)

source(here("03.scrub","01.R_tidy_verse_data_engineering.R"))



# Collect classification data ---------------------------------------------

data <- read_csv(here("01.data", "test.csv"))
data = scrub(data, train=FALSE)[[1]]

# Classification ----------------------------------------------------------
id = data$id
data <- data %>% select(!c('id'))
model <- readRDS(here("02.resources", "01.R", "modelXGBoost.rds"))
predicted <- predict(model, data)


# Result codification -----------------------------------------------------

result <- ifelse(predicted == 2, "died", ifelse(predicted == 1, "euthanized", 
                                              "lived"))

result_table <- data.frame(
  id = id,
  result
)

colnames(result_table) <- c("id", "outcome")


# Save result -------------------------------------------------------------


write.csv(result_table, file = here("08.results", "01.R_tidy_models_output.csv"), 
          row.names = FALSE, quote = FALSE)




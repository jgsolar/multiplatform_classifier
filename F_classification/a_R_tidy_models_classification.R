library(tidyverse)
library(tidymodels)
library(here)

source(here("C_scrub","a_R_tidy_verse_data_engineering.R"))



# Collect classification data ---------------------------------------------

data <- read_csv(here("A_data", "test.csv"))
data = scrub(data, train=FALSE)[[1]]

# Classification ----------------------------------------------------------
id = data$id
data <- data %>% select(!c('id'))
model <- readRDS(here("B_resources", "a_R", "modelXGBoost_saved.rds"))
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


write.csv(result_table, file = here("G_results", "a_R_tidy_models_output.csv"), 
          row.names = FALSE, quote = FALSE)




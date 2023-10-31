library(tidyverse)
library(tidymodels)

source("03.functions/01.R/cleaning.R")



# Collect classification data ---------------------------------------------

data <- read_csv("01.data/original/test.csv")
data <- cleaning(data)

# Classification ----------------------------------------------------------
id = data$id
data <- data %>% select(!c('id'))
model <- readRDS("02.resources/01.R/modelXGBoost.rds")
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

write.csv(result_table, file = "01.data/results/01.R/output.csv", 
          row.names = FALSE, quote = FALSE)




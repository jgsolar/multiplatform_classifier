library(tidyverse)
library(tidymodels)
library(impute)
source("03.functions/01.R/lesionPrep.R")

cleaning <- function(data){

  data$hospital_number <- as.character(data$hospital_number)
  ohe_cols <- c("hospital_number", "mucous_membrane", "pain", "abdomen", 
                "abdomo_appearance", "lesionSite", "lesionType", "lesionSubType",
                "lesionCode", "surgery", "surgical_lesion", "age", "peristalsis", 
                "abdominal_distention", "nasogastric_tube", "rectal_exam_feces",
                "nasogastric_reflux", "capillary_refill_time", "peripheral_pulse",
                "temp_of_extremities")
  non_ohe_cols <- setdiff(names(data), ohe_cols)
  non_ohe_cols <- non_ohe_cols[non_ohe_cols != 'outcome']
  
  data <- data %>% select(-c("cp_data"))
  non_ohe_cols <- non_ohe_cols[!non_ohe_cols %in% c("cp_data")]
  
  # cleaning data -----------------------------------------------------------
  data <- data %>%
    mutate(abdominal_distention = gsub("None", "none", abdominal_distention),
           nasogastric_tube = gsub("None", "none", nasogastric_tube),
           nasogastric_reflux = gsub("None", "none", nasogastric_reflux),
           temp_of_extremities = gsub("None", "normal", temp_of_extremities),
           peripheral_pulse = gsub("None", "normal", peripheral_pulse),
           mucous_membrane = gsub("None", "normal_pink", mucous_membrane),
           capillary_refill_time = gsub("None", "less_3_sec", 
                                        capillary_refill_time),
           capillary_refill_time = str_replace(capillary_refill_time, "^3$", 
                                               "less_3_sec"),
           pain = gsub("None", "alert", pain),
           pain = gsub("moderate", "mild_pain", pain),
           pain = gsub("slight", "alert", pain),
           peristalsis = gsub("distend_small", "normal", peristalsis),
           peristalsis = gsub("None", "normal", peristalsis),
           nasogastric_reflux = gsub("slight", "none", nasogastric_reflux),
           rectal_exam_feces = gsub("None", "normal", rectal_exam_feces),
           rectal_exam_feces = gsub("serosanguious", "normal", rectal_exam_feces),
           abdomen = gsub("None", "normal", abdomen),
           abdomo_appearance = gsub("None", "clear", abdomo_appearance)
           )
  
  # Treat lesion columns
  result <- lesionPrep(data, non_ohe_cols)
  data <- result[[1]]
  non_ohe_cols <- result[[2]]
  ohe_cols <- c(ohe_cols, "lesionSite", "lesionType", "lesionSubType", "lesionCode")
  
  # Codify attributes
  data[non_ohe_cols] <- lapply(data[non_ohe_cols], function(x) as.numeric(x))
  
  # Coding hospital_number
  hospital_number_codifier <- readRDS("02.resources/01.R/hospital_number_codifier_v0.1.rds")
  data$hospital_number <- ifelse(data$hospital_number %in% hospital_number_codifier[[1]], 
                                 data$hospital_number, as.character(hospital_number_codifier[[1]][1]))
  data$hospital_number <- data$hospital_number %>% factor(., ordered = FALSE)
  
  # Impute absent values
  data_temp <- data[, non_ohe_cols, drop = FALSE]
  data_temp <- as.matrix(data_temp)
  data_temp <- impute.knn(data_temp, k=10)$data  # This library doesn't allow memory from training imputation
  data_temp <- as.data.frame(data_temp)
  data[non_ohe_cols] <- data_temp
  rm(data_temp)
  
  return(data)
}
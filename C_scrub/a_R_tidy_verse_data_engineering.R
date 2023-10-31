library(tidyverse)
library(impute)
select <- dplyr::select  # select is masked by some libraries
library(here)
source(here("B_resources","a_R","lesion_encoder.R"))

scrub <- function(data, train=FALSE){
  # get metadata
  data <- data %>% 
    mutate_at(c("hospital_number"), as.character)
  cat_cols <- data %>% select(where(is.character)) %>% names() 
  num_cols <- setdiff(names(data), cat_cols)
  num_cols <- num_cols[num_cols != 'id']
  cat_cols <- cat_cols[cat_cols != 'outcome']
  
  # Remove cp_data
  data <- data %>% select(-c("cp_data"))
  cat_cols <- cat_cols[!cat_cols %in% c("cp_data")]
  
  # Adjust categorical features encoding
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
  
  # Adjust lesion encoding
  result <- lesion_encode(data, num_cols)
  data <- result[[1]]
  num_cols <- result[[2]]
  cat_cols <- c(cat_cols, "lesionSite", "lesionType", "lesionSubType", "lesionCode")
  
  # Coding hospital_number
  if (train==TRUE) {
    hospital_number_codifier <- as.data.frame(table(data$hospital_number))
    colnames(hospital_number_codifier) <- c("hospital_number", "freq")
    hospital_number_codifier <- hospital_number_codifier %>% arrange(desc(freq))
  } else {
    hospital_number_codifier <- readRDS("B_resources/a_R/hospital_number_codifier_v0.1.rds")
    data$hospital_number <- ifelse(data$hospital_number %in% hospital_number_codifier[[1]], 
                                   data$hospital_number, as.character(hospital_number_codifier[[1]][1]))
  }
  
  # Coding categorical features
  data <- data %>% 
    mutate_at(cat_cols, as.factor)
  
  # Impute numeric absent values
  # KNNImputation made in the pipeline was very costly
  # Categorical imputation will be done on the pipeline
    # Impute library doesn't provide memory of training set when imputing test set
    # So memory won't be used in R
  data_temp <- data[, num_cols, drop = FALSE]
  data_temp <- as.matrix(data_temp)
  data_temp <- impute.knn(data_temp, k=10)$data  
  data_temp <- as.data.frame(data_temp)
  data[num_cols] <- data_temp
  rm(data_temp)
  
  # In the training set, remove repeated observations
  if (train==TRUE) {
    data <- data %>% distinct()
  }

  metadata <- list(
    'cat_cols' = cat_cols,
    'num_cols' = num_cols
  )
  
  # Save hospital_number_codifier for using during test set cleaning
  if (train==TRUE)
    write_rds(hospital_number_codifier, here("B_resources", "a_R", "hospital_number_codifier_v0.1.rds"))
  
  # Code outcome variable for model tuning
  if (train==TRUE){
    data <- data %>% mutate(outcome =
                              gsub("died", "2", outcome))
    data <- data %>% mutate(outcome =
                              gsub("euthanized", "1", outcome))
    data <- data %>% mutate(outcome =
                              gsub("lived", "0", outcome))
    data <- data %>% relocate(outcome, .after = last_col())
    data$outcome <- data$outcome %>% factor(., levels = c("2", "1", "0"))
  }
    
    
  return(list(data, metadata))
}
select <- dplyr::select  # select is masked by some libraries

lesionPrep <- function(data, non_ohe_cols) {
  
  data$lesion_1 <- data$lesion_1 %>% as.character()
  data$lesion_2 <- data$lesion_2 %>% as.character()
  data$lesion_3 <- data$lesion_3 %>% as.character()
  
  data$numLesions <- rowSums(data[c("lesion_1", "lesion_2", "lesion_3")] != "0")
  
  data$lesionSite <-  -1
  data$lesionType <- -1
  data$lesionSubType <- -1
  data$lesionCode <- -1
  
  non_ohe_cols <-  c(non_ohe_cols, 'numLesions')
  
  for (i in 1:nrow(data)){
    for (j in c("lesion_1", "lesion_2", "lesion_3")){
      lesion_val = data[i, j]
      if (!is.na(lesion_val) && !is.na(lesion_val) ){
        if (nchar(lesion_val) == 5) {
          if (substr(lesion_val, 1, 2) == "11"){
            data$lesionSite[i] <- 11
            data$lesionType[i] <- substr(lesion_val, 3, 3) %>% as.numeric()
            data$lesionSubType[i] <- substr(lesion_val, 4, 4) %>% as.numeric()
            data$lesionCode[i] <- substr(lesion_val, 5, 5) %>% as.numeric()
          }
          else {
            if (data$lesionSite[i] < substr(lesion_val, 1, 1) %>% as.numeric()){
              data$lesionSite[i] <- substr(lesion_val, 1, 1) %>% as.numeric()
              data$lesionType[i] <- substr(lesion_val, 2, 2) %>% as.numeric()
              data$lesionSubType[i] <- substr(lesion_val, 3, 3) %>% as.numeric()
              data$lesionCode[i] <- substr(lesion_val, 4, 5) %>% as.numeric()
            }
          }
        }
        else {
          if (data$lesionSite[i] < substr(lesion_val, 1, 1) %>% as.numeric()) {
            lesion_val <- str_pad(lesion_val, width = 4, side = "right", pad = "0")
            data$lesionSite[i] <- substr(lesion_val, 1, 1) %>% as.numeric()
            data$lesionType[i] <- substr(lesion_val, 2, 2) %>% as.numeric()
            data$lesionSubType[i] <- substr(lesion_val, 3, 3) %>% as.numeric()
            data$lesionCode[i] <- substr(lesion_val, 4, 4) %>% as.numeric()
          }
        }
      }
    }
  }
  
  data <- data %>% select(-c("lesion_1", "lesion_2", "lesion_3"))
  non_ohe_cols <- non_ohe_cols[!non_ohe_cols %in% c("lesion_1", "lesion_2", "lesion_3")]
  
  lesionSiteDict <- c("11" = "all intestinal sites",
                      "1" = "gastric",
                      "2" = "sm intestine",
                      "3" = "lg colon",
                      "4" = "lg colon and cecum",
                      "5" = "cecum",
                      "6" = "transverse colon",
                      "7" = "retum/descending colon",
                      "8" = "uterus",
                      "9" = "bladder",
                      "0" = "none",
                      "-1" = "none")
  
  lesionTypeDict <- c("1" = "simple",
                      "2" = "strangulation",
                      "3" = "inflammation",
                      "4" = "other",
                      "5" = "other",
                      "6" = "other",
                      "7" = "other",
                      "8" = "other",
                      "9" = "other",
                      "0" = "other",
                      "-1" = "other") 
  
  lesionSubTypeDict <- c("1" = "mechanical",
                         "2" = "paralytic",
                         "3" = "n/a",
                         "4" = "n/a",
                         "5" = "n/a",
                         "6" = "n/a",
                         "7" = "n/a",
                         "8" = "n/a",
                         "9" = "n/a",
                         "0" = "n/a",
                         "-1" = "n/a")   
  
  lesionCodeDict <- c("10" = "displacement",
                      "1" = "obturation",
                      "2" = "intrinsic",
                      "3" = "extrinsic",
                      "4" = "adynamic",
                      "5" = "volvulus/torsion",
                      "6" = "intussuption",
                      "7" = "thromboembolic",
                      "8" = "hernia",
                      "9" = "lipoma/slenic incarceration",
                      "0" = "n/a",
                      "-1" = "n/a")
  
  data <- data %>% mutate(lesionSite = 
                            str_replace_all(lesionSite, lesionSiteDict))
  
  data <- data %>% mutate(lesionType = 
                            str_replace_all(lesionType, lesionTypeDict))
  
  data <- data %>% mutate(lesionSubType = 
                            str_replace_all(lesionSubType, lesionSubTypeDict))
  
  data <- data %>% mutate(lesionCode = 
                            str_replace_all(lesionCode, lesionCodeDict))

  return(list(data, non_ohe_cols))
}
  

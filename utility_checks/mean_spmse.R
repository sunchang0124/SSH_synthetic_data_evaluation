library(synthpop)
library(tidyverse)

original_data <- SD2011[c(1:6,8:21)]

synthetic_data <- syn(original_data)[["syn"]]

mean_spmse <- function(original_data,synthetic_data){
  
  utility <- utility.tables(original_data,synthetic_data,not.synthesised = NULL, cont.na = NULL,print.flag=FALSE)
  
  mean <- mean(utility$tabs[,2])
  
  return(mean)
}

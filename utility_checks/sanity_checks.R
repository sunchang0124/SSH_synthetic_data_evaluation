library(tidyverse)

sanity_checks <- function(original_data,synthetic_data){
  
  salsch_salnum_combis <- full_join(synthetic_data %>% 
    group_by(SALSCH,SALNUM) %>% 
    summarise(n_synthetic=n()),
  original_data %>% 
    group_by(SALSCH,SALNUM) %>% 
    summarise(n_original=n()))
  
  n_new_salsch_salnum_combis <- nrow(salsch_salnum_combis 
                                     %>% filter(is.na(n_original)))
  perc_new_salsch_salnum_combis <- sum(salsch_salnum_combis 
                                       %>% filter(is.na(n_original)) 
                                       %>% pull(n_synthetic)) / nrow(synthetic_data)
  
  perc_end_date_before_start_date <- (synthetic_data %>% filter(DATEIND<DATBEG) %>% nrow())/nrow(synthetic_data)
  
  perc_OMVDIO_in_range <- synthetic_data %>% 
    filter(OMVDIO>0) %>%
    mutate(in_range = between(gebjaarmaand,195710,196710)) %>% 
    pull(in_range) %>% 
    mean()
  
  return(list(n_new_salsch_salnum_combis,perc_new_salsch_salnum_combis,perc_end_date_before_start_date,perc_OMVDIO_in_range))
}

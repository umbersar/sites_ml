setwd("C:/Users/horat/Desktop/CSIROIntership/soilCode")

#create pivot table 
library(reshape)
library(data.table)
library(dplyr)

#load the soil data
soil <- read.csv(file = "hr_lr_labm.csv",fileEncoding="UTF-8-BOM")
#reduce the amount to 5000
rSoil <- nrow(soil)

#extract three column labm_code labr_value h_texture
featureSoilTable <- dcast(soil, agency_code + proj_code + h_texture +  s_id + o_id + h_no + samp_no + labr_no ~ labm_code,value.var = "labr_value")

#drop some of useless columns
featureSoilTable <- select (featureSoilTable,-c(agency_code,proj_code,s_id,o_id,h_no))

write.csv(featureSoilTable,"featureTable.csv", row.names=FALSE)


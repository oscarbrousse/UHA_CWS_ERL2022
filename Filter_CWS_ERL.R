library(CrowdQC); library(data.table);

dates_to_conc <- list()
for(yr in 2015:2020) {
  for(mon in 1:12) {
    tmp <- paste0(yr,"-",sprintf("%02d", mon))
    dates_to_conc[[length(dates_to_conc) + 1]] <- tmp
  }
}
  
city = 'London'

### Directory where the Netatmo data has been downloaded
directory = paste0("/",city,"/")

### Comment out and use this loop if raw monthly files are to be filtered
#for (dates in dates_to_conc){
#  file_raw = paste0("Netatmo_Filter_Ready_",city,"_",dates,".csv")
#  file_filt = paste0("Netatmo_",city,"_",dates,"_filt.csv")
#  df_raw <- data.table::fread(file = paste0(directory,file_raw), sep = '|')
#  df_qc <- CrowdQC::qcCWS(df_raw)
  
#  write.csv(df_qc, file = paste0(directory,file_filt), sep = '|')
#}

### Use the script below for aggregated dataframes longer than one month
file_raw = paste0("Netatmo_Filter_Ready_",city,"_2015-01-2020-12.csv")
df_raw <- data.table::fread(file = paste0(directory,file_raw), sep = '|')
df_qc <- CrowdQC::qcCWS(df_raw)

### Save monthly quality-checked data out of the aggregated dataframe
for (dates in dates_to_conc){
  tmp_df_qc <- df_qc[df_qc$time %like% dates]
  file_filt_tmp = paste0("Netatmo_",city,"_",dates,"_filt.csv")
  write.csv(tmp_df_qc, file = paste0(directory,file_filt_tmp))
}

### Save the quality-checked aggregated dataframe
file_filt = paste0("Netatmo_",city,"_2015-01-2020-12_filt.csv")
write.csv(df_qc, file = paste0(directory,file_filt), sep = '|')
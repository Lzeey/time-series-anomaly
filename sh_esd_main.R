library(AnomalyDetection)
library(plotly)
this.dir <- dirname(parent.frame(2)$ofile) # frame(3) also works.
setwd(this.dir)

DATA_FILE = "./power_data.txt"
test_data = read.csv(DATA_FILE, header=FALSE)
test_data$V1 <- as.numeric(test_data$V1)

data(raw_data)
#data(raw_data)
res_test = AnomalyDetectionTs(raw_data, max_anoms=0.02, direction='both', plot=TRUE)
res_power = AnomalyDetectionVec(test_data, max_anoms=0.01, direction='both', plot=TRUE, period=74)
res_power$plot

# STGAT_MAD
# Detail about WTD dataset
## Abstract
WTD dataset is collected from two abnormal-record wind turbines in a large-scale wind farm. Due to the requirements of business confidentiality, the data are processed through data masking, such as renaming parameters, hiding timestamps and standardizing data. The abnormality in dataset is the cracking of main bearing, and the main reason is frequent wind speed varieties. When the abnormalities of abrasion and wearing happen in the bearing, the temperature changes because the bearing suffers from imbalanced mechanical friction in the rotating process. Therefore, we study from the collected 10 parameters such as environmental temperature, wind speed, hub speed, hub temperature, generator-side active power.

## The number of novel events
The data are collected lasting for 1 to two years, and time interval of data collection is 10 minutes. According to on-the-spot operation and maintenance (O&M) records from wind farm operators, we divide the dataset into training set and testing set. The quantity and abnormal percentage of these two datasets are shown in the following table. 
Dataset|Training size|Testing size(Normal/Abnormal)|Dim.|Abnormal proportion
WT03|24133|28657(14122/15516)|10|54.1%
WT23|64966|21863(14535/6347)|10|43.7%

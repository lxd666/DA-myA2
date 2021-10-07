# Domain-Adaptation-LRCDR


Due to the large amount of data, we had to upload the data separately. Please run the following program before running any program to organize the data into the required format.

cd('./data/')
load('Data_AAL116_1_3sites.mat')
data1 = Data;
load('Data_AAL116_4_6sites.mat')
data2 = Data;
load('Data_AAL116_7_17sites.mat')
data3 = Data;

Data = [data1;data2;data3];
save('Data_17sites_AAL116.mat','Data')

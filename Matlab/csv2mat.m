function [] = gsr2mat(name) 
%name = nombre carpeta usuario
path = pwd; %/Users/antonialarranaga/Desktop/Memoria - scripts
fileGSRPPG = strcat(path, '/sujetos/', name, '/', name, '_GSR_PPG.csv');
csvread(fileGSRPPG)

end


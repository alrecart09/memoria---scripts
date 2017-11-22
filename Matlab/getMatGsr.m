%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
path = '/Users/antonialarranaga/Desktop/Memoria - scripts'; %/Users/antonialarranaga/Desktop/Memoria - scripts
path_sujetos = strcat(path, '/sujetos/');
participantes = dir(path_sujetos); %nombre i en lista(1, i).name

for j=58:length(participantes)
    sujeto = participantes(j).name;
    fprintf(strcat(sujeto, '\n'))
    
    gsr2Ledalab(sujeto)
    ledaLabBatch(sujeto)
    
     %     
%     if ~exist(path_folder, 'dir');
%         mkdir(path_folder)
%     end
% 
%     for i=1:length(archivos)-4
%         gsr2Ledalab(sujeto, t, i-1);
%     end
%     ledaLabBatch(sujeto, t);
end


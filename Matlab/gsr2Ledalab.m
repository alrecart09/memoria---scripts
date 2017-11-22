function [data] = gsr2Ledalab(name)
    %UNTITLED4 Summary of this function goes here
    %   Detailed explanation goes here
    path = '/Users/antonialarranaga/Desktop/Memoria - scripts'; %/Users/antonialarranaga/Desktop/Memoria - scripts

    gsr = load(strcat(path, '/sujetos/', name, '/', name, '_GSR.mat'));
    %name = nombre carpeta usuario

    timestamps = gsr.time;
    gsr_ = gsr.conductance;
    events = struct();
    events.time = 0;
    events.nid = 0;
    events.name = '0';
    events.userdata = [];

    data = struct('conductance', gsr_, 'time', timestamps/1000, 'timeoff', 0, 'event', events); %timestamps en segundos
    %gsr = [timestamps gsr_];

    path_folder = strcat(path, '/sujetos/', name, '/GSR/');
    if ~exist(path_folder, 'dir');
        mkdir(path_folder)
    end

    filemat_gsr = strcat(path, '/sujetos/', name, '/GSR/', name, '_GSRLedalab.mat');
    %filemat_ppg = strcat(path, '/sujetos/', name, '/', name, '_PPG.mat');

    save(filemat_gsr, 'data')
    %save(filemat_ppg, 'ppg')

    %LedaLab cda analysis - downsampling a 10 hz - smooth adaptativo
end



path = pwd;
folder = strcat(path, '/sujetos/');
nombres = dir(folder);
nombres = nombres(4:length(nombres));

for i=8:length(nombres)
   fprintf('%s \n', nombres(i).name)
   getMat(nombres(i).name);
end
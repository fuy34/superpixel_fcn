function [stat] = loadcsv(path, line)
if nargin == 1
    line = 1063;
end
  M = csvread(path,1,4,[1 4 line 4]);
  br = mean(M);
  M = csvread(path,1,5,[1 5 line 5]);
  bp = mean(M);
  M = csvread(path,1,8,[1 8 line 8]);
  asa = mean(M);
  M = csvread(path,1,11,[1 11 line 11]);
  co = mean(M);
  M = csvread(path,1,17,[1 17 line 17]);
  n_sp = mean(M);
%   if n_sp > 9000
%       n_sp = n_sp-7800;
%   end
  stat = [n_sp, asa, br, bp, co ];
end
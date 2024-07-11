% Task1
[Yn,On,Yt,Ot] = ann2053598(2053598, 100, 0.01, 1, 128, 0, 0, 1);


% Task2
% TSE
[Yn,On,Yt,Ot] = ann2053598(2053598, 100, 0.01, 1, 128, 0, 0, 1);
% XE
[Yn,On,Yt,Ot] = ann2053598(2053598, 100, 0.01, 1, 128, 0, 0, 2);


% Task3
% TSE
% heuristic unscaled
[Yn,On,Yt,Ot] = ann2053598(2053598, 100, 0.01, 1, 128, 0, 0, 1);
% calculus based
[Yn,On,Yt,Ot] = ann2053598(2053598, 100, 0.01, 2, 128, 0, 0, 1);

% XE
% heuristic unscaled
[Yn,On,Yt,Ot] = ann2053598(2053598, 100, 0.01, 1, 128, 0, 0, 2);
% calculus based
[Yn,On,Yt,Ot] = ann2053598(2053598, 100, 0.01, 2, 128, 0, 0, 2);

% Task4
% three hidden layers
[Yn,On,Yt,Ot] = ann2053598(2053598, 100, 0.01, 1, 128, 64, 32, 1);
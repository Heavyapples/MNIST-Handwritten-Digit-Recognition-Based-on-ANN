close all
% run it backwards... But only for TSE (final layer sigmoid)
xmin=0.0001; xmax=1-xmin;
for c = 1:10
  y = xmin*ones(10,1); y(c) = xmax;
  x = W3*( log( y ./ (1-y) ) - b3);
  % scale back to [xmin,xmax]
  x=x-min(x); x=xmin+(xmax-xmin)*x/max(x);
  x = W2*( log( x ./ (1-x) ) - b2);
  % scale back to [xmin,xmax]
  x=x-min(x); x=xmin+(xmax-xmin)*x/max(x); 
  x = xmin*(x<0.5)+ xmax*(x>=0.5);
  B = reshape(x,[28,28]);
  subplot(3,4,c);
  imh = imshow(B','InitialMagnification','fit');
end

% multi-layer feed forward neural network with sequential mode of training

clc, clear all, format compact, close all

m = input('Enter the number of hidden neurons: ');
P = input('Enter the number of training patterns(MAX.25): ');
TT = input('Enter the number of testing patterns(MAX.11): ');

% from the research paper we found that number of input neurons are 
% from the research paper we found that number of output neurons are 
% synaptic weights should have a value between -1 to 1
% learning rate 'neu' should have a value between 0 to 1
% momentum term 'alpha' should have a value between 0 to 1
% the input file name is 'INPUT'
% the output file name is 'OUT'

file = fopen('INPUT.txt');
a = textscan(file,'%f %f %f %f');
b = cell2mat(a)  % matrix containing all patterns i.e. 6 inputs & 4 outputs
fclose(file);

% we will consider out of 114 patterns, 85 will be used for training and...
% rest will be used for testing of trained ANN

l = 3;  % depends on the number of input neurons
n = 1;  % depends on the number of output neurons
I(l+1,P) = zeros;   % declaring input matrix

for p = 1:P
I(1,p) = 1;
% below values needs to be read from input file
I(2,p) = b(p,1);
I(3,p) = b(p,2);
I(4,p) = b(p,3);
%I(5,p) = b(p,4);
%I(6,p) = b(p,5);
%I(7,p) = b(p,6);

end

% Normalisation of inputs
x_normI(l+1,P) = zeros ;
xmaxI(l,1)=zeros;
xminI(l,1)=zeros;
for i = 1:l
    for p = 1:P
    xmaxI(i) = max(I(i+1,:));
    xminI(i) = min(I(i+1,:));
    x_normI(i+1,p) = 0.1 + 0.8*(I(i+1,p) - xminI(i))/(xmaxI(i)-xminI(i));
    x_normI(1,p) = 1 ;
    end
   
end


for i = 1:(l+1)
    for j = 1:m
    V(l+1,m) = zeros ;
    % formula to generate random numbers between -1 and 1
    % r = a + (b-a)*rand
    V(i,j) = -1 + ((1+1)*rand);
    end
end
for j = 1:(m+1)
    for k = 1:n
    W(m+1,n) = zeros ;
    W(j,k) = -1 + ((1+1)*rand);
    %W(1,k) = -1 + (1+1)*rand;
    end
end


% Target value matrix
TARGET(n,P) = zeros ;
for p = 1:P
    TARGET(1,p) = b(p,4);
    %TARGET(2,p) = b(p,8);
    %TARGET(3,p) = b(p,9);
    %TARGET(4,p) = b(p,10);
end
 
% Normalisation of Target value matrix
x_normT(n,P) = zeros ;
xmaxT(n)=zeros;
xminT(n)=zeros;
for k = 1:n
    for p = 1:P
    xmaxT(k) = max(TARGET(k,:));
    xminT(k) = min(TARGET(k,:));
    x_normT(k,p) = 0.1 + 0.8*((TARGET(k,p) - xminT(k))/(xmaxT(k)-xminT(k)));
    end
end
iter_num = 0 ;
MSE=10;
d_w(m+1,n) = zeros ;  d_v(l+1,m) = zeros ;
V1 = V ;
W1 = W ;


% To write in a file the MSE values
FILE2 = fopen('C:\Users\HP\Desktop\674 CODE\MSE.txt','w');
fprintf(FILE2, ' No. of iterations Vs MSE   \n\n');
fprintf(FILE2, "  ITERATIONS      MSE  \n");
z = 1 ;

while MSE>0.0040
%while iter_num<100000


hid_inp(m,P) = zeros ;
hid_inp = V1'*x_normI ;
hid_out(m+1,P) = zeros;
for j = 1:m
    for p = 1:P
%using log sigmoid transfer function
hid_out(j+1,p) = 1/(1+exp(-hid_inp(j,p))) ;
hid_out(1,p) = 1;
    end
end


out_inp(n,P) = zeros ;
out_inp =   W1'*hid_out ;
out_out(n,P) = zeros;
for p = 1:P
    for k = 1:n
    %using log sigmoid transfer function
    out_out(k,p) = 1/(1+exp(-out_inp(k,p))) ;   % matrix of outputs of output neurons
    end
end

% Normalisation of output neurons
%x_normO(n,P) = zeros ;
%for k = 1:n
   % for p = 1:P
   % xmaxO = max(out_out(k,:));
    %xminO = min(out_out(k,:));
    %x_normO(k,p) = 0.1 + 0.8*((out_out(k,p) - xminO)/(xmaxO-xminO));
    %end
%end



% Calculating Mean squared Error(MSE)

E=zeros(P,1);
for p=1:P
    for k=1:n
        E(p,1)=E(p,1)+(1/2)*((x_normT(k,p)-out_out(k,p))^2);
        % every row element is having sum of squared errors of all output..
        % neurons for a single pattern
    end
    %E(p,1)=E(p,1)/n ;
end
% MSE initialization
MSE = 0 ;
%for k=1:n
   for p=1:P
      MSE = MSE + E(p,1) ;
   end
%end
   MSE = MSE/P 

% d_w = 0; d_v = 0;

 oldd_w = d_w;
 oldd_v = d_v;

 % Updating W & V
 nn = 0.3 ;     % nn - learning rate
 d_w(m+1,n) = zeros ;
 for k = 1:n
       for j = 1:m+1
           for p = 1:P
               d_w(j,k) = d_w(j,k) + (x_normT(k,p)-out_out(k,p))*(out_out(k,p))*(1-out_out(k,p))*hid_out(j,p) ; 
           end
       end
 end
d_w = d_w*nn/P;

 d_v(l+1,m) = zeros ;
 for i = 1:l+1
       for j = 1:m
           for p = 1:P
               for  k = 1:n
               d_v(i,j) = d_v(i,j) + (x_normT(k,p)-out_out(k,p))*(out_out(k,p))*(1-out_out(k,p))*hid_out(j+1,p)*(1-hid_out(j+1,p))*x_normI(i,p)*W1(j+1,k) ; 
               end
           end
       end
 end
 d_v = d_v*(nn/(n*P)) ;
 %n_v(l+1,m) = zeros ;
 %for i = 1:l+1
  %   for j = 1:m
  %       n_v(i,j) = V(i,j) + d_v(i,j) ; 
  %   end
 %end

 % Optimum learning rate
   alpha=0.5;  % alpha - momentum term
 V1(l+1,m) = zeros ; 
 W1(m+1,n) = zeros ;
 W1 = W1 + d_w + (alpha*oldd_w);
 V1 = V1 + d_v + (alpha*oldd_v);

 iter_num = iter_num + 1 ;

fprintf(FILE2,"   %d \t \t %f  \n",iter_num, MSE);


%iter_num1(iter_num) = zeros ;
%MSE1(iter_num) = zeros ;
iter_num1(z) = iter_num ;
MSE1(z) = MSE ;
z = z+1 ;

V1
W1

end

% end of while loop  

plot (iter_num1, MSE1,'--')
fclose(FILE2);

disp(iter_num) ;

% Testing of ANN
test(4,TT) = zeros ;
for t = 1:TT
test(1,t) = 1 ;
test(2,t) = b(25+t,1) ;
test(3,t) = b(25+t,2) ;
test(4,t) = b(25+t,3) ;
%test(5,t) = b(25+t,4) ; 
%test(6,t) = b(25+t,5) ; 
%test(7,t) = b(25+t,6) ;
end

% Normalisation of inputs
x_normtI(l+1,TT) = zeros ;
for i = 1:l
    for t = 1:TT
    x_normtI(i+1,t) = 0.1 + 0.8*((test(i+1,t) - xminI(i))/(xmaxI(i)-xminI(i)));
    x_normtI(1,t) = 1 ;
    end
end


% Target value matrix
tT(n,TT) = zeros ;
for t = 1:TT
    tT(1,t) = b(25+t,4);
    %tT(2,t) = b(85+t,8);
    %tT(3,t) = b(85+t,9);
    %tT(4,t) = b(85+t,10);
end
 
% Normalisation of Target value matrix
%x_normtT(n,TT) = zeros ;
%for k = 1:n
 %   for t = 1:TT
  %  xmaxtT = max(tT(k,:));
   % xmintT = min(tT(k,:));
   % x_normtT(k,t) = 0.1 + 0.8*((tT(k,t) - xmintT)/(xmaxtT-xmintT));
   % end
%end

hid_inpt(m,TT) =  zeros ;
hid_inpt = V1'*x_normtI ;
hid_outt(m+1,TT) = zeros;
for j = 1:m
    for t = 1:TT
%using log sigmoid transfer function
hid_outt(j+1,t) = 1/(1+exp(-hid_inpt(j,t))) ;
hid_outt(1,t) = 1;
    end
end

out_inpt(n,TT) = zeros;
out_inpt =   W1'*hid_outt ;
out_outt(n,TT) = zeros;
for t = 1:TT
    for k = 1:n
    %using log sigmoid transfer function
    out_outt(k,t) = 1/(1+exp(-out_inpt(k,t))) ;   % matrix of outputs of output neurons
    end
end
%deNormalising the output of output neurons
de_normtO(n,TT)=zeros;
for t = 1:TT
    for k=1:n
        de_normtO(k,t)=(((out_outt(k,t)-0.1)/0.8)*(xmaxT(k)-xminT(k))) + xminT(k);
    end
end
% Normalisation of output neurons
%x_normtO(n,TT) = zeros ;
%for k = 1:n
    %for t = 1:TT
    %xmaxtO = max(out_outt(k,:));
    %xmintO = min(out_outt(k,:));
    %x_normtO(k,t) = 0.1 + 0.8*((out_outt(k,t) - xmintO)/(xmaxtO-xmintO));
    %end
%end

error = ((tT - de_normtO)./(tT))*100 ;  % all elements have percentage error of trained ANN
avg_error = 0 ;
for t = 1:TT
    for k = 1:n
% avg_error gives us the total error of the ANN
avg_error = avg_error + abs(error(k,t)) ; 
    end
end
avg_error = avg_error/(TT+n) ;

% Calculating Mean squared Error(MSE)

Et=zeros(TT,1);
for t=1:TT
    for k=1:n
        Et(t,1)=Et(t,1)+(1/2)*((tT(k,t)-de_normtO(k,t))^2);
        % every row element is having sum of squared errors of all output..
        % neurons for a single pattern
    end
    %Et(t,1)=Et(t,1)/n ;
end
% MSE initialization
MSEt = 0 ;
%for k=1:n
   for t=1:TT
      MSEt = MSEt + Et(t,1) ;
   end

% de_normO(n,TT)=zeros;
for p = 1:P
    for k=1:n
        de_normO(k,p)=(((out_out(k,p)-0.1)/0.8)*(xmaxT(k)-xminT(k))) + xminT(k);
    end
end
   MSEt = MSEt/TT  ;



FILE = fopen('C:\Users\HP\Desktop\674 CODE\OUTPUT.txt','w');
%for i = 1:l
fprintf(FILE, 'V matrix - matrix containing synaptic connection weights between input & hidden layer \n\n');
fprintf(FILE,' %6f\t%6f\t%6f\t%6f\t%6f\n',V1);
fclose(FILE);

FILE = fopen('C:\Users\HP\Desktop\674 CODE\OUTPUT.txt','a');
fprintf(FILE, '\n\nW matrix - matrix containing synaptic connection weights between hidden layer & output layer \n\n');
fprintf(FILE,' %6f\t%6f\t%6f\t%6f\t%6f\t%6f\n',W1);
fclose(FILE);

FILE = fopen('C:\Users\HP\Desktop\674 CODE\OUTPUT.txt','a');
fprintf(FILE, '\n\nMean Square Error for training set is %s \n', MSE);
fprintf(FILE, 'Mean Square Error for testing set is %s \n\n', MSEt);
fclose(FILE);


FILE = fopen('C:\Users\HP\Desktop\674 CODE\OUTPUT.txt','a');
fprintf(FILE, 'The output of the network is \n\n');
fprintf(FILE,' %6f \n',de_normO);
fclose(FILE);

norm_error = x_normT - out_out ;
de_norm_error(n,P) = zeros ; 
for k = 1:n
    for p = 1:P
    xmaxE = max(norm_error(k,:));
    xminE = min(norm_error(k,:));
    de_norm_error(k,p) = (((norm_error(k,p)-0.1)/0.8)*(xmaxE-xminE)) + xminE;
    end
end

FILE = fopen('C:\Users\HP\Desktop\674 CODE\OUTPUT.txt','a');
fprintf(FILE, '\n\nThe absolute error is \n\n');
fprintf(FILE,' %6f \n', abs(de_norm_error));
fclose(FILE);

cc = V'*x_normI

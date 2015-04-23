clc;
clear all;

%Mfcc calculation
s = {'classical','jazz','country','pop','rock','metal'};  % Loading genres to a string array
m=1;
finmfcc = zeros(600,13);
k=0;
ep = 1000; % epochs value
for p=1:6
                % loops for calulating fft for 600 songs
for i=0:9
for j=0:9
  % myFile = ['F:\Trilce\project3\sound\genres\blues\blues.000' 'int2str(i)' 'int2str(j)' '.wav'];
   % myFile = sprintf('F:\Trilce\project3\sound\genres\blues\blues.000%d%d.wav',i,j);
   % [a(:,k),fs(k)]=audioread(myFile);
  

    % Define variables
    Tw = 25;                % analysis frame duration (ms)
    Ts = 7.24;                % analysis frame shift (ms)
    alpha = 0.97;           % preemphasis coefficient
    M = 20;                 % number of filterbank channels 
    C = 12;                 % number of cepstral coefficients
    L = 22;                 % cepstral sine lifter parameter
    LF = 300;               % lower frequency limit (Hz)
    HF = 3700;              % upper frequency limit (Hz)
    


    % Reading each song
    [ a, fs ] = audioread( strcat('F:\Trilce\project3\sound\genres\',s{p},'\',s{p},'.000',int2str(i),int2str(j),'.wav') );


    % Feature extraction (feature vectors as columns) for each song
    [ MFCCs, FBEs, frames ] = ...
                    mfcc( a, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );


                [feat,frm] = size(MFCCs);  % features and frames sizes
                trmfc = transpose(MFCCs);   
                k=k+1;
                c = round(frm/10)+1;
                d = round((frm)*(9/10));
                genmfc = trmfc(c:d,:);      % calculating genre specific features and frames
                finmfcc(k,:) = mean(genmfc); % array 600*13 for storing mfcc values
                
                
end
end
end



%finmfcc = transpose(finmfcc);
% Normalizing mfcc values
[M,I] = max(finmfcc);
for i=1:600
   
    for j=1:13
        
        %[mv(j,1) mv(j,2)] = max(fbl(:,j))
        %fbn(i,j) = fbl(i,j)/sum(fbl(:,j)); 
        nfinmfcc(i,j) = finmfcc(i,j)/M(j);
    end
end   

% song genre label values for each song
m=1;
for i=1:6
    
   for j=1:100
    
   lb(m)=i; 
       
   m=m+1;
   end 
   
end   



% 10-fold cross validation calculation
tes = zeros(60,10);
trn = zeros(540,10);
for a=0:9
   p=0;q=0;
   for b=1:600
   if(rem(b,10)==a)
       p=p+1;
       tes(p,a+1)=b;        % array for test indices from 10- fold
   else
       q=q+1;
       trn(q,a+1)=b;        % array for train indices from 10- fold
   end
   
   end
    
end

% calculating mfcc logistic regression and accuracy
confmat = zeros(6,6); % array to store confusion matrix
for t=1:10

    
% Initial random weights for logistic regression

for i=1:6
    for j=1:14

    wgt(i,j) = 0;       %Initializing weights to 0
    %wgt(i,j) = 1;
        
    end
end

   for i=1:540
         
          xcal(i,1) = 1;
           
       
      for j=2:14
        
          xcal(i,j)= nfinmfcc(trn(i,t),j-1);  %----loading mfcc train data
          
      end
      
   end   
    
    

% transpose of X comp values of log reg

txcal = transpose(xcal);

% Multiplication of weights and X values

mulcal = wgt*txcal;
 

expmul = exp(mulcal);

for i=1:540
   
    expmul(6,i)=1;
    
end   

% calculating each logistic regression value in each fold

for i=1:6
   
    for j=1:540
       
        probmul(i,j) = expmul(i,j)/sum(expmul(:,j));    %calculating logistic regression probability values
        
    end    
    
end    


% Calculating delta values
del = zeros(6,540);
for i=1:6
    
    for j=1:540
    
        if(lb(trn(j,t))==i) %----
        del(i,j) = 1;
        end
    end    
end    

% Optimizing weights using Gradient descent

lamda = 0.001;
netao = 0.01;

for x=1:ep
    
   neta = netao/(1+(x/ep));
   wgt = wgt + (neta*(((del - probmul)*xcal) - (lamda*wgt) ));
mulcal = wgt*txcal;
 

expmul = exp(mulcal);

for y=1:540
   
    expmul(6,y)=1;
    
end   

% calculating logistic regression value for new weights

for i=1:6
   
    for j=1:540
       
        probmul(i,j) = expmul(i,j)/sum(expmul(:,j));
        
    end    
    
end    

    
end   


% Loading mfcc test data

   for i=1:60
         
          tesxcal(i,1) = 1;
           
       
      for j=2:14
        
          tesxcal(i,j)= nfinmfcc(tes(i,t),j-1);      %----
          
      end
      
   end   

% calculating test data probabilites of logistic regression

tesprob = exp((wgt) * (transpose(tesxcal)));

for i=1:60
   tesprob(6,i)=1; 
end 


for i=1:6
   
    for j=1:60
       
        probntes(i,j) = tesprob(i,j)/sum(tesprob(:,j));
        
    end    
    
end 

% Calculating maximum value genre for each song in test data for
% classification
for i=1:60
[val(i,1) val(i,2)] = max(probntes(:,i));
end
%trmaxtes = transpose(maxtes);

% Calculating percentage accuracy
h=0;

for i=1:60
   
    if (lb(tes(i,t))==val(i,2))     %----
        h=h+1;
    end
end    
perc(t) = (h/60)*100;

%confusion matrix calculation


%C = confusionmat(lb(tes(:,1)),val(:,2));
C = confusionmat(lb(tes(:,t)),val(:,2));
confmat = confmat + C;

end
% Displaying confusing matrix
S = ' The consolidated confusion matrix for mfcc components ';
disp(S);
disp(confmat);

% Calculating final accuracy percentage
per =0;
for i=1:10

per = per + perc(i);
end

faccu = per/10;
S = ' The final accuracy for mfcc components ';
disp(S);
disp(faccu);

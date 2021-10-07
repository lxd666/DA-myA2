clear;clc;
addpath('./functions/')
addpath('../data/tools/')
addpath('../data/tools/liblinear-2.43/windows/')
load('../data/Data_17sites_CC200.mat')
load('../data/Best_Params_LRCDR_CC200.mat')
data = Data;
Num_domains = length(data);
options.feaSelNum = 400;
options.ReducedDim = 100;
options.T = 30;
options.k = 5;

display('Starting......')
for j = 1:Num_domains
    i = [1: j-1,j+1 : Num_domains];
    Xs = [];Xt = [];Ys = [];Yt = [];
    
    % target domain
    tgt = data{j,1};
    tgt_name = tgt(1 : find(tgt=='_')-1);
    Xt = data{j,2}; Yt = data{j,3};  Xt(isnan(Xt)==1)=0;
    Xt = Xt ./ repmat(sum(Xt,2),1,size(Xt,2));
    Xt = double(zscore(Xt,1));
    
    % source domain
    for i_src = i
        fts = data{i_src,2};
        fts(isnan(fts)==1)=0;
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
        Xs = [Xs;double(zscore(fts,1))];
        Ys = [Ys;data{i_src,3}];
    end
    
    % feature selection
    fea_index = FeaSel_Xs(Xs,Ys,options.feaSelNum);
    Xs = Xs(:,fea_index);
    Xt = Xt(:,fea_index);
    
    options.alpha = params(j,1);
    options.beta = params(j,2);
    options.gamma = params(j,3);
    options.lambda = params(j,4);
    options.int = params(j,5);
    
    [Acc,Res_performance] = LRCDR(Xs',Xt',Ys,Yt,options,params,j);
    Result(j,:)  = [Acc(end), Res_performance.SEN,  Res_performance.SPE,Res_performance.PPV,Res_performance.NPV,Res_performance.F1];
end
save('Result_LRCDR_CC200.mat','Result')
display('Ending......')



function fea_index = FeaSel_Xs(Xs,Ys,feaSelNum)
cols_summary = zeros(1,size(Xs,2));
for dim_i = 1:size(Xs,2)
    cols_summary(dim_i) = abs(corr(Ys,Xs(:,dim_i)));
end
sort_cols = sort(cols_summary,'descend');
gate = sort_cols(feaSelNum);
fea_index = (cols_summary>=gate);
end

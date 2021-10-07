clear;clc;
addpath('./functions/')
addpath('../data/tools/')
addpath('../data/tools/libsvm-3.17/windows/')
load('../data/Data_17sites_AAL116.mat')
load('../data/Best_Params_MEDA.mat')
data = Data;
Num_domains = length(data);
feaSelNum = 400;
options.T = 10;
options.d = 20;
options.rho = 1.0;
options.p = 10;
options.kernel_type = 'linear';
options.n_neighbors = 1;

display('Starting......')
for j = 1 : Num_domains
    i = [1: j-1,j+1 : Num_domains];
    Xs = [];Xt = [];Ys = [];Yt = [];
    
    tgt = data{j,1};
    tgt_name = tgt(1 : find(tgt=='_')-1);
    Xt = data{j,2}; Yt = data{j,3};  Xt(isnan(Xt)==1)=0;
    Xt = Xt ./ repmat(sum(Xt,2),1,size(Xt,2));
    Xt = double(zscore(Xt,1));
    
    for i_src = i
        fts = data{i_src,2};
        fts(isnan(fts)==1)=0;
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
        Xs = [Xs;double(zscore(fts,1))];
        Ys = [Ys;data{i_src,3}];
    end
    
    fea_index = FeaSel_Xs(Xs,Ys,feaSelNum);
    Xs = Xs(:,fea_index);
    Xt = Xt(:,fea_index);
    
    options.lambda = params(j,1);
    options.eta = params(j,2);
    
    [Acc,Res_performance] = MEDA(Xs,Ys,Xt,Yt,options);
    Result(j,:)  = [Acc, Res_performance.SEN,Res_performance.SPE,Res_performance.PPV,Res_performance.NPV,Res_performance.F1];    
end
save('Result_MEDA.mat','Result')
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

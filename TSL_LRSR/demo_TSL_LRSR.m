clear;clc;
addpath('../data/tools/')
addpath('../data/tools/libsvm-new/')
load('../data/Data_17sites_AAL116.mat')
load('../data/Best_Params_TSL_LRSR.mat')
data = Data;
Num_domains = length(data);
feaSelNum = 400;

display('Starting......')
for j = 1:Num_domains 
    i = [1: j-1,j+1 : Num_domains];
    Xs = [];Xt = [];Ys = [];Yt = [];
    
    tgt = data{j,1};
    tgt_name = tgt(1 : find(tgt=='_')-1);
    Xt = data{j,2}; Yt = data{j,3};  Xt(isnan(Xt)==1)=0;
    Xt = Xt ./ repmat(sqrt(sum(Xt.^2)),[size(Xt,1) 1]);

    for i_src = i
        fts = data{i_src,2};
        fts(isnan(fts)==1)=0;
        fts = fts./repmat(sqrt(sum(fts.^2)),[size(fts,1) 1]);
        Xs = [Xs;double(zscore(fts,1))];
        Ys = [Ys;data{i_src,3}];
    end

    fea_index = FeaSel_Xs(Xs,Ys,feaSelNum);
    Xs = Xs(:,fea_index);
    Xt = Xt(:,fea_index);
    
    Xs = Xs';Xt = Xt';
    
    alpha = params(j,1);
    beta = params(j,2);
    gamma = params(j,3);
    P = TSL_LRSR(Xs,Xt,Ys,alpha,beta,gamma);
    X_train = P'*Xs;
    Y_test  = P'*Xt;
    
    X_train = X_train./repmat(sqrt(sum(X_train.^2)),[size(X_train,1) 1]);
    Y_test  = Y_test ./repmat(sqrt(sum(Y_test.^2)),[size(Y_test,1) 1]);
    tmd = ['-s 0 -t 2 -g ' num2str(1e-3) ' -c ' num2str(1000)];
    model = svmtrain(Ys, X_train', tmd);
    [Yt0, acc] = svmpredict(Yt, Y_test', model);
    acc = acc(1);
    
    clab = [1,2];
    Res_performance = performance(Yt,Yt0,[],clab);
    
    Result(j,:) = [acc,Res_performance.SEN,Res_performance.SPE,Res_performance.PPV,Res_performance.NPV,Res_performance.F1];
end
save('Result_TSL_LRSR.mat','Result')
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

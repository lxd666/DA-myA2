function res = performance(Yt_true_test,Yt_pred_test,score,clab)
%input:
%   Yt_true_test:n*1，真实标签
%   Yt_pred_test:n*1，预测标签
%   score:n*1，是正类的概率
%   clab:[正类，负类]
%   Acc_test:nfold*1，测试集每个fold准确率
%   Acc_train:nfold*1，训练集每个fold准确率
%output：
%   res:结构体
%       ACC：训练集准确率
%       VAR：标准差
%       SEN：灵敏度，(正)查全率R: recall
%       SPE：特异度，(负)查全率
%       PPV：阳性检测，诊断为病人中，真正有病的概率   (正)查准率 P: precision
%       NPV：阴性检测，诊断为正常中，真正正常的概率   (负)查准率
%       F1：F1值= 2*PR/(P+R)
%       AUC：ROC曲线下面积
%       ACC_tr：测试集平均准确率
%       Score：n*1，是正类的概率
%       Label_Predict：预测标签
%       Label_True：真实标签


[~,nt,~,~] = performance_metric([Yt_true_test,Yt_pred_test],2,clab);
TP = nt(1,1);
FN = nt(1,2);
FP = nt(2,1);
TN = nt(2,2);
   
res.ACC = (TP+TN)/length(Yt_true_test)*100;
res.SEN = TP/(TP+FN)*100;%灵敏度  (正)查全率R: recall
res.SPE = TN/(FP+TN)*100;%特异度  (负)查全率
res.PPV = TP/(TP+FP)*100;%阳性检测，诊断为病人中，真正有病的概率   (正)查准率P: precision
res.NPV = TN/(TN+FN)*100;%阴性检测，诊断为正常中，真正正常的概率   (负)查准率
res.F1 = 2*res.PPV*res.SPE/(res.PPV+res.SPE); %2*PR/(P+R)


res.Score = score;
res.Label_Predict = Yt_pred_test;
res.Label_True = Yt_true_test;
end
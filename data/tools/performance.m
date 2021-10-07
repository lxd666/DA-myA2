function res = performance(Yt_true_test,Yt_pred_test,score,clab)
%input:
%   Yt_true_test:n*1����ʵ��ǩ
%   Yt_pred_test:n*1��Ԥ���ǩ
%   score:n*1��������ĸ���
%   clab:[���࣬����]
%   Acc_test:nfold*1�����Լ�ÿ��fold׼ȷ��
%   Acc_train:nfold*1��ѵ����ÿ��fold׼ȷ��
%output��
%   res:�ṹ��
%       ACC��ѵ����׼ȷ��
%       VAR����׼��
%       SEN�������ȣ�(��)��ȫ��R: recall
%       SPE������ȣ�(��)��ȫ��
%       PPV�����Լ�⣬���Ϊ�����У������в��ĸ���   (��)��׼�� P: precision
%       NPV�����Լ�⣬���Ϊ�����У����������ĸ���   (��)��׼��
%       F1��F1ֵ= 2*PR/(P+R)
%       AUC��ROC���������
%       ACC_tr�����Լ�ƽ��׼ȷ��
%       Score��n*1��������ĸ���
%       Label_Predict��Ԥ���ǩ
%       Label_True����ʵ��ǩ


[~,nt,~,~] = performance_metric([Yt_true_test,Yt_pred_test],2,clab);
TP = nt(1,1);
FN = nt(1,2);
FP = nt(2,1);
TN = nt(2,2);
   
res.ACC = (TP+TN)/length(Yt_true_test)*100;
res.SEN = TP/(TP+FN)*100;%������  (��)��ȫ��R: recall
res.SPE = TN/(FP+TN)*100;%�����  (��)��ȫ��
res.PPV = TP/(TP+FP)*100;%���Լ�⣬���Ϊ�����У������в��ĸ���   (��)��׼��P: precision
res.NPV = TN/(TN+FN)*100;%���Լ�⣬���Ϊ�����У����������ĸ���   (��)��׼��
res.F1 = 2*res.PPV*res.SPE/(res.PPV+res.SPE); %2*PR/(P+R)


res.Score = score;
res.Label_Predict = Yt_pred_test;
res.Label_True = Yt_true_test;
end
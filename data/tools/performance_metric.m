function [recall_precision,n1,recall,precision]=performance_metric(te_true_predict_lable,class_num,clab)
        
test_true_predict_lable=te_true_predict_lable;
for k=1:class_num
    index1=find(test_true_predict_lable(:,1)==clab(k)); 
    n0(k,1)=length(index1);
    for i=1:class_num
       index2=find(test_true_predict_lable(index1,2)==clab(i));
       n1(k,i)=length(index2);    
                             %n1(1,1)����ʵ��ǩΪ1��Ԥ���ǩҲΪ1 �ĸ���  TP
                             %n1(1,2)����ʵ��ǩΪ1��Ԥ���ǩΪ-1 �ĸ���   FN
                             %n1(2,1)����ʵ��ǩΪ-1��Ԥ���ǩΪ 1 �ĸ���  FP
                             %n1(2,2)����ʵ��ǩΪ-1��Ԥ���ǩΪ -1 �ĸ��� TN                        
    end
    recall(1,k)=n1(k,k)/n0(k,1)*100; %recall�ǲ�ȫ��
end
for k=1:class_num
    precision(1,k)=n1(k,k)/sum(n1(:,k))*100;
end
recall_precision=[recall;precision];
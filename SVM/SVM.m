function [acc,Res_performance] = SVM(Xs,Ys,Xt,Yt)
Xs = double(Xs);
Xt = double(Xt);
time_start = clock();
[acc,y_pred] = LinAccuracy(Xs,Xt,Ys,Yt);
time_end = clock();
time_pass = etime(time_end,time_start);


clab=[1,2];
Res_performance = performance(Yt,y_pred,[],clab);
end

function [acc,predicted_label] = LinAccuracy(trainset,testset,trainlbl,testlbl)
model = trainSVM_Model(trainset,trainlbl);
[predicted_label, accuracy, decision_values] = svmpredict(testlbl, testset, model);
acc = accuracy(1,1);
end

function svmmodel = trainSVM_Model(trainset,trainlbl)
C = [0.001 0.01 0.1 1.0 10 100 1000];
for i = 1 :size(C,2)
    model(i) = svmtrain(double(trainlbl), sparse(double((trainset))),sprintf('-c %d -q -v 2',C(i) ));
end
[val indx]=max(model);
CVal = C(indx);
svmmodel = svmtrain(double(trainlbl), sparse(double((trainset))),sprintf('-c %d -q',CVal));
end
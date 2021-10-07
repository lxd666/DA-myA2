% https://blog.csdn.net/qq_35166974/article/details/89889262

function [best_c,best_g,best_acc] = SvmSearchParas(data,label,c_max,c_min,c_step,g_max,g_min,g_step,v)
%--------------------------------------------------------------------------
%The function looks for the SVM's most important parameters c and g
%The Author:�ȵȵǵ�-Ande
%The Email:18356768364@163.com
%The Blog:qq_35166974
%%
%Initialization parameter
if nargin < 9
    v = 10;
end
if nargin < 8
    v = 10;
    g_step = 1;
end
if nargin < 7
    v = 10;
    g_step = 1;
    c_step = 1;
end
if nargin < 6
    v = 10;
    g_step = 1;
    c_step = 1;
    g_min = -5;
end
if nargin < 5
    v = 10;
    g_step = 1;
    c_step = 1;
    g_min = -5;
    g_max = 5;
end
if nargin < 4
    v = 10;
    g_step = 1;
    c_step = 1;
    g_min = -5;
    g_max = 5;
    c_min = -5;
end
if nargin < 3
    v = 10;
    g_step = 1;
    c_step = 1;
    g_min = -5;
    g_max = 5;
    c_min = -5;
    c_max = 5;
end
if nargin < 2
    warning('You did not enter enough parameters!');
end
%%
%Parameter optimization
[mesh1,mesh2] = meshgrid(c_min:c_step:c_max,g_min:g_step:g_max);
[raw,col] = size(mesh1);
acc = zeros(raw,col);
for i=1:raw
    for j=1:col
        cg_paras = ['-v ',num2str(v),'-c ',num2str(2.^mesh1(i,j)),' ','-g ',num2str(2.^mesh2(i,j))];
        acc(i,j) = libsvmtrain(double(label),double(data),cg_paras);
    end
end
best_acc = max(max(acc));
[label_i,label_j] = find(acc==best_acc);
best_c = 2.^mesh1(label_i,label_j);
best_g = 2.^mesh2(label_i,label_j);
figure
mesh(mesh1,mesh2,acc);
xlabel('log2c');
ylabel('log2g');
zlabel('Accuracy')
% % % ��������������������������������
% % % ��Ȩ����������ΪCSDN�������ȵȵǵ�-Ande����ԭ�����£���ѭCC 4.0 BY-SA��ȨЭ�飬ת���븽��ԭ�ĳ������Ӽ���������
% % % ԭ�����ӣ�https://blog.csdn.net/qq_35166974/article/details/89889262
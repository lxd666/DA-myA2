function [ACC,Res_performance] = LRCDR(Xs,Xt,Ys,Yt,options,params,j)
alpha = options.alpha;
beta = options.beta;
gamma = options.gamma;
lambda = options.lambda;
int = options.int;
k = options.k;
[m,ns] = size(Xs);
[~,nt] = size(Xt);
X = [Xs,Xt];
X = X*diag((1./sqrt(sum(X.^2))));
C = length(unique(Ys));
n = ns + nt;

%% Low Rank
rho = params(j,6);  mu = params(j,7);
max_mu = 10^6;  convergence = 10^-4;    d = options.ReducedDim;
Qs = [eye(ns);zeros(nt,ns)];    Qt = [zeros(ns,nt);eye(nt)];
[P1,~] = PCA1(X(:,1:ns)', options);
Z = zeros(nt,ns);   J = zeros(nt,ns);   E = zeros(d,ns);    F = zeros(d,C);
Y1 = zeros(d,ns);   Y2 = zeros(nt,ns);  leq1 = [];          leq2 = [];

%% Class-Discriminative Representation
Ms = zeros(ns,C);
for c = reshape(unique(Ys),1,C)
    Ms(find(Ys==c),c) = 1/length(find(Ys==c));
end
M = [Ms;zeros(nt,C)];

Ds_same = computeLs(Ys,C);
Ds_diff = compute_Ds_diff(Ys,C);

% initialize Lt
distXt = L2_distance_1(X(:,ns+1:end),X(:,ns+1:end));
[distX1, idx] = sort(distXt,2);
S = zeros(nt,nt);
rr = zeros(nt,1);
for i = 1:nt
    di = distX1(i,2:k+2);
    rr(i) = 0.5*(k*di(k+1)-sum(di(1:k)));
    id = idx(i,2:k+2);
    S(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end
r = mean(rr);
S = (S+S')/2;
Dt_neighbor = diag(sum(S,2))-S;
Dt_neighbor = Dt_neighbor/norm(Dt_neighbor,'fro');

% compute D
D = blkdiag(Ds_same - lambda*Ds_diff,Dt_neighbor);
D = D/norm(D,'fro');

% initialize Gt
svmmodel = train(double(Ys), sparse(double(X(:,1:ns)')),'-s 1 -B 1.0 -q');
[Yt0_CMMS,~,~] = predict(Yt, sparse(X(:,ns+1:end)'), svmmodel,'-q');
Gt = full(sparse(1:nt,Yt0_CMMS,1));
if size(Gt,2) < C
    Gt = [Gt,zeros(nt,C-size(Gt,2))];
end

for iter = 1:options.T
    %update P
    A5 = X*Qs - X*Qt*Z;
    A6 = E - Y1/mu;
    FAI = alpha*Qt*Qt'+ M*M'+gamma*D;
    if (iter == 1)
        P = P1;
    else
        P = inv(beta*eye(m) + X*FAI*X'+ mu/2 *A5*A5')*(alpha*X*Qt*Gt*F' + X*M*F'+mu/2*A5*A6');
    end
    
    %update J
    ta = 1/mu;
    temp = Z + Y2/mu;
    temp(isnan(temp)) = 0; temp(temp==inf) = 0;
    [U,S,V] = svd( temp,'econ' );
    sigma = diag(S);
    svp = length(find(sigma>ta));
    if svp>=1
        sigma = sigma(1:svp)-ta;
    else
        svp=1;
        sigma = 0;
    end
    J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    
    %update E
    the2 = int/mu;
    temp_E = P'*X*Qs-P'*X*Qt*Z+Y1/mu;
    E = max(0,temp_E-the2)+min(0,temp_E+the2);
    
    %update Z
    A1 = P'*X*Qt;
    A2 = P'*X*Qs - E + Y1/mu;
    A3 = J - Y2/mu;
    Z = inv(  A1'*A1 + eye(nt)  )*(A1'*A2 + A3);
    
    %update F
    A4 = P'*X*M;
    F = (alpha*A1*Gt + A4)*inv(alpha*Gt'*Gt + eye(C));
    
    %update G
    PTX = P'*X;
    PTXt = PTX(:,ns+1:end);
    for j = 1:nt
        Yt0(j) = searchBestIndicator(PTXt(:,j),F,C);
    end
    Gt = full(sparse(1:nt,Yt0,1));
    if size(Gt,2) < C
        Gt = [Gt,zeros(nt,C-size(Gt,2))];
    end
    
    %update Dt_neighbor
    Dt_neighbor = computeLt(PTXt,k,r);
    
    %update D
    D = blkdiag(Ds_same - lambda*Ds_diff,Dt_neighbor);
    D = D/norm(D,'fro');
    
    %Update the multilpers
    Y1_temp = PTX*Qs - PTX*Qt*Z - E;
    Y1 = Y1 + mu*Y1_temp;
    Y2_temp = Z - J;
    Y2 = Y2 + mu*Y2_temp;
    
    % updating mu
    mu = min(rho*mu,max_mu);
    
    % checking convergence
    leq1 = norm(Y1_temp,Inf);
    leq2 = norm(Y2_temp,Inf);
    if iter > 2
        if leq1<convergence && leq2<convergence
            break
        end
    end
    
    %compute Acc
    Xs_SVM = P'*X(:,1:ns);
    Xt_SVM = P'*X(:,ns+1:end);
    Xs_SVM = Xs_SVM./repmat(sqrt(sum(Xs_SVM.^2)),[size(Xs_SVM,1) 1]);
    Xt_SVM = Xt_SVM ./repmat(sqrt(sum(Xt_SVM.^2)),[size(Xt_SVM,1) 1]);
    svmmodel = train(double(Ys),sparse(Xs_SVM') ,'-s 1 -B 1.0 -q');
    [Yt0_SVM,~,~] = predict(Yt, sparse(Xt_SVM'), svmmodel,'-q ');
    ACC(iter,1) = length(find(Yt == Yt0_SVM))/nt * 100;
end

% indicators
clab=[1,2]; % labels: ASD:1,TC:2
Res_performance = performance(Yt,Yt0_SVM,[],clab);
end
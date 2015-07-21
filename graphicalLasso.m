function [Theta W] = graphicalLasso(S, rho, maxIt, tol)
% Solve the graphical Lasso
% minimize_{Theta > 0} tr(S*Theta) - logdet(Theta) + rho * ||Theta||_1
% Input:
% S -- sample covariance matrix
% rho --  regularization parameter
% maxIt -- maximum number of iterations
% tol -- convergence tolerance level
%
% Output:
% Theta -- inverse covariance matrix estimate(precision matrix)
% W -- regularized covariance matrix estimate, W = Theta^-1

p = size(S, 1);

if nargin < 4, tol = 1e-6; end
if nargin < 3, maxIt = 1e2; end

% Initialization
W = S + rho(1) * eye(p); 
W_old = W;
i = 0;

% Graphical Lasso loop
while i < maxIt
    i = i+1; 
    for j = p:-1:1
        jminus = setdiff(1:p, j); % each dim except j
        [V D] = eig(W(jminus, jminus));
        d = diag(D);
        X = V * diag(sqrt(d)) * V'; % W^(1/2)
        Y = V * diag(1./sqrt(d)) * V' * S(jminus, j); % W^(-1/2)*s
		beta_0 = W(jminus, j);
        beta = LassoInference(X, Y, W(jminus, jminus), beta_0, rho, maxIt);
        W(jminus, j) = W(jminus, jminus) * beta;
        W(j, jminus) = W(jminus, j)'; % symmatric update
    end
	fprintf('L1 norm difference: %g\n', norm(W-W_old));
    if norm(W-W_old, 1) < tol
        break;
    end
    W_old = W;
end

if i == maxIt
    fprintf('Maximum number of iteration reached, glasso may not converge. \n');
end
Theta = W^-1;
end


function beta = LassoInference(X, Y, W, beta_0, rho, maxIt)
    gOptions.maxIter = maxIt;
    gOptions.verbose = 0;
    gOptions.corrections = 10;
    beta = L1General2_PSSgb(@(beta)objL1MV(X, Y, W, beta), beta_0, rho, gOptions);
end

function [f, g] = objL1MV(X, Y, W, beta)
    f = 0.5*(norm(X*beta - Y))^2;
    g = W*beta - Y;
end
    

function sparse_graph = learning_sparse_graph(X, rho, maxIt, tol)
% Interface for inferencing a sparse graph of data X
% Input:
% X -- input data, it should be a N*M matrix (N: data amount, M: feature dimension)
% rho --  regularization parameter
% maxIt -- maximum number of iterations
% tol -- convergence tolerance level

	if exist('zscore') == 0
		fprintf('function zscore in stats toolbox doesnt exist\n');
		return
	end
	X = zscore(X); % zero mean unit variance
    [N, M] = size(X);
    
% preparing sample covariance matrix S
    
    S = zeros(M);
    for i=1:N
        S = S + X(i, :)'*X(i, :); % Si, j = 1/N*sum(xi(n)*xj(n))
    end
    S = S / N;
    
    [Theta, W] = graphicalLasso(S, rho, maxIt, tol);
    % Theta -- precision matrix
    % W -- covariance matrix
    sparse_graph = {Theta, W};
end

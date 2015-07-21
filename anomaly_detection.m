function Ascore = anomaly_detection(reference_set, testing_set)
    maxIt = 1e2;
    tol = 1e-5;
    rho = ones(size(reference_set, 2)-1, 1)*0.1; % default rho filler
	addpath(genpath('/home/extra/b01902004/KPP/Code/KPP/L1General'));    
    GraphRef = learning_sparse_graph(reference_set, rho, maxIt, tol);
    GraphTest = learning_sparse_graph(testing_set, rho, maxIt, tol);
    
    Ascore = score_graph(GraphRef, GraphTest);
end

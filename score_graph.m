function Ascore = score_graph(ref, test)
% compute anomaly score of each variable in the graph based on KL
% divergence
% Input:
% ref -- reference graph, {1}: precision matrix {2}: covariance matrix
% test -- testing graph
    Lref = ref{1};
    Wref = ref{2};
    Ltest = test{1};
    Wtest = test{2};
    M = size(Lref, 1);
    Ascore = zeros(M, 1);
    for j=1:M
        jminus = setdiff(1:M, j);
        % pre-compute variables
        wA = Wref(jminus, j);
        wB = Wtest(jminus, j);
        lA = Lref(jminus, j);
        lB = Ltest(jminus, j);
        WA = Wref(jminus, jminus);
        WB = Wtest(jminus, jminus);
        lambdaA = Lref(j, j);
        lambdaB = Ltest(j, j);
        phiA = Wref(j, j);
        phiB = Wtest(j, j);
        
        dAB = wA'*(lB-lA) + ...
                 0.5*(lB'*WA*lB/lambdaB - lA'*WA*lA/lambdaA) + ...
                 0.5*(log(lambdaA/lambdaB) + phiA*(lambdaB-lambdaA));
        dBA = wB'*(lA-lB) + ...
                 0.5*(lA'*WB*lA/lambdaA - lB'*WB*lB/lambdaB) + ...
                 0.5*(log(lambdaB/lambdaA) + phiB*(lambdaA-lambdaB));
             
        Ascore(j) = max(dAB, dBA);
    end
    
end
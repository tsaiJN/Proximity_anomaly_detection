myStruct = load('/tmp3/andy/Proximity/exchange_rate.mat');
currency = {'AUD', 'BEF', 'CAD', 'FRF'};
term1 = myStruct.term1(:, 1:length(currency)); % n*12
term2 = myStruct.term2(:, 1:length(currency)); % n*12

Ascore = anomaly_detection(term1, term2);


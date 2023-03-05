
prior0=[
0.2 % Stanje C
0.8 % Stanje H
];

transmat0=[
0.5 0.5 % P(C|C) P(H|C)
0.4 0.6 % P(C|H) P(H|H)
];

Q=size(prior0,1);   % broj stanja modela

obsmat0=[
0.5 0.4 0.1 % P(1|C) P(2|C) P(3|C)
0.2 0.4 0.4 % P(1|H) P(2|H) P(3|H)
];

O=size(obsmat0,2);


T_1 = 15;
nex_1 = 10;
data_1 = dhmm_sample(prior0, transmat0, obsmat0, nex_1, T_1);

llm1 = zeros(nex_1, 1);
for i=1:nex_1
    llm1(i) = dhmm_logprob(data_1(i,:), prior0, transmat0, obsmat0);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_2 = 1500;
nex_2 = 10;
data_1500 = dhmm_sample(prior0, transmat0, obsmat0, nex_2, T_2);

llm2 = zeros(nex_2, 1);
for i=1:nex_2
    llm2(i) = dhmm_logprob(data_1500(i,:), prior0, transmat0, obsmat0);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T_3 = 15;
nex_3 = 1000;
data_1000 = dhmm_sample(prior0, transmat0, obsmat0, nex_3, T_3);

llm3 = zeros(nex_3, 1);
for i=1:nex_3
    llm3(i) = dhmm_logprob(data_1000(i,:), prior0, transmat0, obsmat0);
end

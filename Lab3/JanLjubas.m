% ===============================================================
% Oznacavanje stanja HMM modela

% Imamo tri pristrane kocke od kojih uvijek bacamo jednu odabranu.
% Stanja modela su indeksi koristene pristrane kocke.

% Vektor inicijalne vjerojatnosti stanja (za t=1) odredjen bacanjem nepristrane kocke:
prior0=[
1 % Prva kocka (ako je palo '1')
2 % Druga kocka (ako je palo '2' ili '3')
3 % Treca kocka (ako je palo '4', '5' ili '6')
]/6;

% Broj stanja HMM modela
Q=size(prior0,1);

% ---------------------------------------------------------------

% Matrica vjerojatnosti promjena stanja

% a11 a12 a13
% a21 a22 a23
% a31 a32 a33

% Za eksperiment sa stohastickom izmjenom stanja, parametar
% M se koristi za definiranje vjerojatnosi prijelaza u
% novo stanje u matrici prijelaza A, pri cemu se stanja nuzno
% mijenjaju ciklicki radi forsirane strukture tranzicijske matrice.

M= 5; % Ovdje definirate M iz vaseg personaliziranog zadatka.

% Formiraj matricu vjerojatnosti prijelaza stanja
% (uz ciklicku strukturu izmjene stanja, jer su
% prijelazi 1->3, 2->1 i 3->2 zabranjeni)
transmat0=[
M-1 1 0 % P(1|1) P(2|1) P(3|1)
0 M-1 1 % P(1|2) P(2|2) P(3|2)
1 0 M-1 % P(1|3) P(2|3) P(3|3)
]/M;


obsmat0=[
20 5 4 3 6 2
1 5 20 5 4 5
4 1 4 2 20 9
] / 40;
% izvjesnost ~ udio u 40 mjerenja

O1 = [ 5 5 5 5 6 5 6 1 1 1 3 5 3 4 3 4 5 5 6 4 1 1 1 5 6 6 3 4 3 5 3 3 4 3 3 6 6 5 1 1 5];
O2 = [ 1 1 4 2 3 4 2 1 2 2 4 6 2 4 2 3 4 2 2 1 1 2 2 2 2 2 4 1 1 2 5 5 6 6 5 5 2 1 2 3 5];

logprob1 = dhmm_logprob(O1, prior0, transmat0, obsmat0);
logprob2 = dhmm_logprob(O2, prior0, transmat0, obsmat0);

obslik1 = multinomial_prob(O1, obsmat0);
obslik2 = multinomial_prob(O2, obsmat0);

[alpha, beta, gamma, ll] = fwdback(prior0, transmat0, obslik1, 'scaled', 0);

vpath1 = viterbi_path(prior0, transmat0, obslik1);
[ll1, p1] = dhmm_logprob_path(prior0, transmat0, obslik1, vpath1);

vpath2 = viterbi_path(prior0, transmat0, obslik2);
[ll2, p2] = dhmm_logprob_path(prior0, transmat0, obslik2, vpath2);

% 5. a) [ll1-logprob1 ll2-logprob2]

obslik_skraceni = multinomial_prob(O1(1:4), obsmat0);
vpath_skraceni = viterbi_path(prior0, transmat0, obslik_skraceni);
[ll_skraceni, p_skraceni] = dhmm_logprob_path(prior0, transmat0, obslik_skraceni, vpath_skraceni);

% 6. b) 
% dijeljenje -> razlika u exp
udio = exp(ll_skraceni - dhmm_logprob(O1(1:4), prior0, transmat0, obsmat0));


% 7. generiranje
T_1 = 156;
nex_1 = 14;
rng("default");
data_1 = dhmm_sample(prior0, transmat0, obsmat0, nex_1, T_1);

% 8. odredivanje dugotrajne statistike
hm = hist(data_1(1,:)', [1 2 3 4 5 6]);
% hm(:,1)

% uzastopno mnozenje matrice
a0 = transmat0;
for i=1:156
    a0 = a0*transmat0;
end

tablica_8c = mean(hist(data_1', [1 2 3 4 5 6])')/T_1;
% a0(1,:) * obsmat0
najveca_razlika = max(abs(a0(1,:) * obsmat0 - tablica_8c));


% 9. zadatak
svi_uzorci = zeros(nex_1, 1);
for i = 1:nex_1
 svi_uzorci(i, :) = dhmm_logprob(data_1(i, :), prior0, transmat0, obsmat0);
end
najveci = max(svi_uzorci);
najmanji = min(svi_uzorci);
sredina = mean(svi_uzorci);


% 10. zadatak

    % 1. HMM model -> slucajni parametri
rng('default');
prior1 = normalise(rand(3,1));
transmat1 = mk_stochastic(rand(3,3));
obsmat1 = mk_stochastic(rand(3,6));

[LL_1, prior2, transmat2, obsmat2] = dhmm_em(data_1, prior1, transmat1, obsmat1, 'thresh', 1e-6 ,'max_iter', 200);

    % 2. HMM model -> zadani parametri

[LL_2, prior3, transmat3, obsmat3] = dhmm_em(data_1, prior0, transmat0, obsmat0, 'thresh', 1e-6, 'max_iter', 200);


% 11. zadatak -> evaluacija treniranih modela

LL_zadani = dhmm_logprob(data_1, prior0, transmat0, obsmat0);

LL_los = dhmm_logprob(data_1, prior1, transmat1, obsmat1);

LL_prvi = dhmm_logprob(data_1, prior2, transmat2, obsmat2);

LL_drugi = dhmm_logprob(data_1, prior3, transmat3, obsmat3);

 

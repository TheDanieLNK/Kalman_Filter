% This function generates a random variable equal to k.w.p.
% P(k)/sum(P).
% Here, P = [P(1), P(2), ..., P(K)] where the P[K] are nonnegative.
%
% Re-used from the course material: Probability in EE and CS by Jean
% Walrand

function T = discrete(P)
    Pnorm = [0 P]/sum(P);
    Pcum = cumsum(Pnorm);
    R=rand(1);
    [~,T] = histc(R, Pcum);


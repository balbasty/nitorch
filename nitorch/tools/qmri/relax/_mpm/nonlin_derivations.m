clear
syms TE TR sina cosa 'positive'   % Fixed parameters (a \in [0 90])
syms lR1 lR2 lA ld 'real'         % Optimised parameters (log)
syms X 'real'                     % Observation

% Precompute useful values
A  = exp(lA);
R1 = exp(lR1);
R2 = exp(lR2);
d  = 1 / (1 + exp(-ld));        % delta = MT ratio
md = 1 - d;                     % 1 - delta
e1 = exp(-TR*R1);               % Exponential decay w.r.t. R1
e2 = exp(-TE*R2);               % Exponential decay w.r.t. R2

% Signal fit and negative log-likelihood
S = A * sina * md * (1 - e1) / (1 - md * cosa * e1) * e2;    % Signal
R = S - X;                                                   % Residual
L = 0.5*R^2;                                                 % Objective

% Compute gradients automatically
g = [diff(L, lA); ...
     diff(L, lR2); ...
     diff(L, lR1); ...
     diff(L, ld)];
H = [diff(L, lA,  lA) diff(L, lA,  lR2) diff(L, lA,  lR1) diff(L, lA,  ld); ...
     diff(L, lR2, lA) diff(L, lR2, lR2) diff(L, lR2, lR1) diff(L, lR2, ld); ...
     diff(L, lR1, lA) diff(L, lR1, lR2) diff(L, lR1, lR1) diff(L, lR1, ld); ...
     diff(L, ld,  lA) diff(L, ld,  lR2) diff(L, ld,  lR1) diff(L, ld,  ld)];
H0 = subs(H, X, S); % Hessian at optimum -> positive-definite

% Check that our gradients are correct
gg = [S; ...
      -TE * R2 * S; ...
      -TR * R1 * e1 * S * (md .* cosa - 1) ./ ((1 - e1) .* (1 - md .* cosa .* e1));...
      - (1 - md) * S / (1 - md * cosa * e1)];
HH0 = gg * gg.';
gg = gg * R;

simplify(g-gg, 100)
simplify(H0-HH0, 100)

%% Check true hessian of the signal term
%
% full hessian = signal_grad * signal_grad' + signal_hess * residual
% hessian at optimum (residual == 0) = signal_grad * signal_grad'

gs = [diff(S, lA); ...
      diff(S, lR2); ...
      diff(S, lR1); ...
      diff(S, ld)];
Hs = [diff(S, lA,  lA) diff(S, lA,  lR2) diff(S, lA,  lR1) diff(S, lA,  ld); ...
      diff(S, lR2, lA) diff(S, lR2, lR2) diff(S, lR2, lR1) diff(S, lR2, ld); ...
      diff(S, lR1, lA) diff(S, lR1, lR2) diff(S, lR1, lR1) diff(S, lR1, ld); ...
      diff(S, ld,  lA) diff(S, ld,  lR2) diff(S, ld,  lR1) diff(S, ld,  ld)];


beta = md .* cosa;
k_r1 = (beta - 1) * e1 / ((1 - e1) * (1 - beta * e1));      % < 0
l_r1 = (1 + beta * e1) / (1 - beta * e1);                   % > 0
k_mt = 1 / (md*(1 - cosa * md * e1));                       % > 0
l_mt = cosa * md * e1;                                      % > 0 if a in [0, pi/4] 
gsig = - d * md;                                            % gradient of 1 - sig
hsig = d * md * (d - 1/2) * 2;                              % hessian of 1 - sig: take abs(d-1/2) to ensure pos. def.

h_mt = md * (1 - cosa * e1) / (1 - beta * e1);
% ggs(4) * 2 * (k_mt * l_mt * gsig + md - 1/2)

ggs = [S; ...
       -TE * R2 * S; ...
       -TR * R1 * S * k_r1;...
       S * k_mt * gsig];
HHs = [ggs(1); ...
       ggs(2) * (1 - TE * R2); ...
       ggs(3) * (1 - TR * R1 * l_r1); ...
       ggs(4) * (2 * h_mt - 1)];
   
   
simplify(gs-ggs, 100)
simplify(Hs(1,1)-HHs(1), 100)
simplify(Hs(2,2)-HHs(2), 100)
simplify(Hs(3,3)-HHs(3), 100)
simplify(Hs(4,4)-HHs(4), 100)

%% Sub-problem: R1 term (without log encoding)
% x = -TR * R1
% a = cos(fa) * (1 - delta)
clear
syms a x 'real'

ex = exp(x);
f = (1 - ex) / (1 - a*ex);
g = diff(f, x);
h = diff(g, x);

k = (a - 1) * ex / ((1 - ex) * (1 - a * ex));
l = 1 + ex / (1 - ex) + a * ex / (1 - a * ex);
gg = f * k;
hh = f * k * (k + l);

simplify(g-gg, 100)
simplify(h-hh, 100)

% derivation of k + l
% numerator:
% num = (a-1)ex + (1-ex)(1-b*ex) + ex(1-a*ex) + a*ex*(1-ex)
%     = - ex + 1 + a*ex - a*ex^2
%     = a*ex*(1 - ex) + 1 - ex
%     = (1 - ex)*(1 + a*ex)
% den = (1 - ex)*(1 - a * ex)
% num/den = (1 + a * ex) / (1 - a * ex)
ll = (1 + a * ex) / (1 - a * ex);
hh2 = f * k * ll;
simplify(h-hh2, 100)


%% Sub-problem: R1 term (with log encoding)
% y = log(R1)
% a = cos(fa) * (1 - delta)
% b = -TR
clear
syms a b y 'real'

x = b * exp(y);
ex = exp(x);
f = (1 - ex) / (1 - a*ex);
g = diff(f, y);
h = diff(g, y);

k = (a - 1) * ex / ((1 - ex) * (1 - a * ex));
l = (1 + a * ex) / (1 - a * ex);
gg = x * f * k;
hh = x * f * k * (1 + x * l);

simplify(g-gg, 100)
simplify(h-hh, 100)

%% Sub-problem: MT term (no log encoding)
% x = 1 - delta
% e1 = exp(-TR * R1)
% a = cos(fa)
clear
syms x e1 a 'real'

f = x * (1 - e1) / (1 - a * x * e1);
g = diff(f, x);
h = diff(g, x);

k = a * e1 / (1 - a * x * e1);
l = 1/x;
gg = f * (k + l);
hh = f * (k + l) * (2 * k);

simplify(g-gg, 100)
simplify(h-hh, 100)


%% Sub-problem: MT term (with log encoding)
% y = logit(delta)
% e1 = exp(-TR * R1)
% a = cos(fa)
clear
syms y e1 a 'real'

x = 1 - 1 / (1 + exp(-y));
f = x * (1 - e1) / (1 - a * x * e1);
g = diff(f, y);
h = diff(g, y);

k = 1/(x*(1 - a * x * e1));
l = a * e1 / (1 - a * x * e1);
gs = x * (x - 1);               % gradient of 1 - sig
hs = x * (x - 1) * (2*x - 1);   % hessian of 1 - sig: take -abs(2*x-1) to ensure pos. def.
gg = f * k * gs;
hh = f * k * gs * 2 * (l * gs + x - 1/2);

simplify(g-gg, 100)
simplify(h-hh, 100)

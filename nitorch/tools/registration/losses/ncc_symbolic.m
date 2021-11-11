N = 3;
x = sym('x', [N 1], 'real');
y = sym('y', [N 1], 'real');

xnorm = (x - mean(x)) / std(x, 1);
ynorm = (y - mean(y)) / std(y, 1);
ncc = (xnorm'*ynorm)/N;

%% --- components


% d sigma^2 / dx = 2 * x * (1 - mean(x)) / n
g = diff(var(x, 1), x(1));
gg = 2 * x * (1 - mean(x)) / N;
simplify(g - gg(1), 100)


% d2 sigma^2 / dxdx = 2 * (I - ) / n
g = diff(var(x, 1), x(1));
gg = 2 * x * (1 - mean(x)) / N;
simplify(g - gg(1), 100)


% d sigma / dx = xnorm / n
g = diff(std(x, 1), x(1));
gg = xnorm / N;
simplify(g - gg(1), 100)

% d xnorm / dx = ( (n*I - 11') - xnorm*xnorm' ) / (n * sigma)
gg = ( (N * eye(N) - 1)- xnorm * xnorm' ) / (N * std(x, 1));
g = diff(xnorm(1), x(1));
simplify(g - gg(1, 1), 100)
g = diff(xnorm(2), x(1));
simplify(g - gg(2, 1), 100)
g = diff(xnorm(1), x(2));
simplify(g - gg(1, 2), 100)

% full gradient
g = sym(zeros(N, 1));
for i=1:length(x)
    g(i) = diff(ncc, x(i));
end

gg = (ynorm - xnorm * ncc) / (N * std(x, 1));

simplify(g - gg, 100)


% full hessian
h = sym(zeros(N));
for i=1:length(x)
for j=1:length(x)
    h(i, j) = diff(gg(i), x(j));  % does not work with g for some reason
end
end

hh = (3 * ncc) * (xnorm*xnorm') - (N*eye(N) - 1) * ncc - (ynorm*xnorm' + xnorm*ynorm');
hh = hh / (N.^2 * std(x, 1).^2);
simplify(h - hh, 100)



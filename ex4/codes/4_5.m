format long
F = 253.86549552;
c = 1.16674;
m = 239.245;
alpha = F / m;
beta = c / m;

xspan = [0 91.44];
[x, V] = ode45(@(x, V) alpha - beta * sqrt(2.0 * V), xspan, 0);
Answer1 = sqrt(2.0 * V(end))

syms v(x);
eqn = diff(v, x) * v == alpha - beta * v;
cond = v(0) == 0;
vSol(x) = dsolve(eqn, cond);
Answer2 = double(vSol(91.44))

syms v;
eqn = alpha * 91.44 + alpha / beta * v + (alpha^2.0) / (beta^2.0) * log(1 - beta / alpha * v) == 0;
Answer3 = double(solve(eqn, v))
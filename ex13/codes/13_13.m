format long
clear all
t = 0 : 12;
y = [43.65, 109.86, 187.21, 312.67, 496.58, 707.65, 960.25, 1238.75, 1560.00, 1824.29, 2199.00, 2438.89, 2737.71];

% Logistic模型拟合
[b, bint, r, rint, s] = regress(log(3000 ./ y - 1)', [ones(13,1), t']);
b
s
a = exp(b(1)) % 参数a的估计值
k = -b(2) % 参数k的估计值

nlintool(t, y, @Logistic, [3000, a, k]);
[beta, R, J, CovB, MSE, ErrorModelInfo] = nlinfit(t, y, @Logistic, [3000, a, k]);
beta % 参数估计值
sqrt(MSE) % 均方根误差

% Gompertz模型拟合
nlintool(t, y, @Gompertz, [3000, a, k]);
[beta, R, J, CovB, MSE, ErrorModelInfo] = nlinfit(t, y, @Gompertz, [3000, 30, 0.4]);
beta % 参数估计值
sqrt(MSE) % 均方根误差

function y = Logistic(b, t)
    y = b(1) ./ (1 + b(2) * exp(-b(3) .* t));
end

function y = Gompertz(b, t)
    y = b(1) .* exp(-b(2) .* exp(-b(3) .* t));
end
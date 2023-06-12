format long
clear all

% 情况一：搅拌程度 x1 视为普通变量
x1 = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3];
x2 = [6, 7, 8, 9, 10, 6, 7, 8, 9, 10, 6, 7, 8, 9, 10];
y = [28.1, 32.3, 34.8, 38.2, 43.5, 65.3, 67.7, 69.4, 72.2, 76.9, 82.2, 85.3, 88.1, 90.7, 93.6];

% 使用 regress 函数进行线性回归分析，构建回归模型
[b, bint, r, rint, s] = regress(y', [ones(15,1), x1', x2']);
b
bint
s

% 使用 rcoplot 函数进行残差分析，观察模型的拟合情况和异常值
figure(1), rcoplot(r, rint);

% 情况二：搅拌程度 x1 视为三个水平的无定量关系变量
z = [0, 0; 0, 0; 0, 0; 0, 0; 0, 0; 0, 1; 0, 1; 0, 1; 0, 1; 0, 1; 1, 0; 1, 0; 1, 0; 1, 0; 1, 0];

% 使用 stepwise 函数进行逐步回归分析，确定哪些因变量应被采用
stepwise([z, x2'], y);

% 根据确定的因变量，使用 regress 函数构建回归模型
[b, bint, r, rint, s] = regress(y', [ones(15,1), z, x2']);
b
bint
s

% 使用 rcoplot 函数进行残差分析，观察模型的拟合情况和异常值
figure(2), rcoplot(r, rint);

% 引入交互项的情况
rstool([x1', x2'], y, 'interaction');

% 使用 stepwise 函数进行逐步回归分析，确定哪些因变量应被采用
stepwise([z, x2', (z(:, 1) .* x2'), (z(:, 2) .* x2')], y);

% 根据确定的因变量，使用 regress 函数构建回归模型
[b, bint, r, rint, s] = regress(y', [ones(15,1), z, x2', (z(:, 1) .* x2'), (z(:, 2) .* x2')]);
b
bint
s

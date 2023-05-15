% x, y, w, z1, z2
lb = [0; 0; 0; 0; 0];
ub = [500; 500; 500 + 500; 500; 500];

A = [0, 0, 0, +1, +1;
    -1, -1, 1, 0, 0;
    0, 0, +1, +1, 0;
    +1, +1, -1, 0, +1;];
b1 = [500;0;100;200];
b2 = [500;0;600;200];

% Define the constraints
nonlcon = @(x)deal(mix(x));

% Set options for optimization
options = optimoptions('fmincon','Display','off');

% Initialize variables to store the solutions
ans1 = 0;
ansx1 = [];
ans2 = 0;
ansx2 = [];
ans3 = 0;
ansx3 = [];
ans4 = 0;
ansx4 = [];

% Loop through the optimization problems and store the best solutions
for iter_cnt = 1:50
    x0 = rand(1, 5) * 100;
    [x, fval, exitflag] = fmincon(@(vec)profit1(vec), x0, A, b1, [], [], lb, ub, nonlcon, options);
    if exitflag > 0 && -fval > ans1
        ans1 = -fval;
        ansx1 = x;
    end
    
    x0 = rand(1, 5) * 100;
    [x, fval, exitflag] = fmincon(@(vec)profit1(vec), x0, A, b2, [], [], lb, ub, nonlcon, options);
    if exitflag > 0 && -fval > ans2
        ans2 = -fval;
        ansx2 = x;
    end
    
    x0 = rand(1, 5) * 100;
    [x, fval, exitflag] = fmincon(@(vec)profit2(vec), x0, A, b1, [], [], lb, ub, nonlcon, options);
    if exitflag > 0 && -fval > ans3
        ans3 = -fval;
        ansx3 = x;
    end
    
    x0 = rand(1, 5) * 100;
    [x, fval, exitflag] = fmincon(@(vec)profit2(vec), x0, A, b2, [], [], lb, ub, nonlcon, options);
    if exitflag > 0 && -fval > ans4
        ans4 = -fval;
        ansx4 = x;
    end
    disp(ans1);
    disp(ansx1);
end

% Print the solutions
disp(ans1);
disp(ansx1);
disp(ans2);
disp(ansx2);
disp(ans3);
disp(ansx3);
disp(ans4);
disp(ansx4);

function [c, ceq] = mix(vec)
    x = vec(1);
    y = vec(2);
    w = vec(3);
    z_1 = vec(4);
    z_2 = vec(5);
    c(1) = (0.03 * x + 0.01 * y) / (x + y) * w + 0.02 * z_1 - 2.5 / 100 * (w + z_1);
    c(2) = (0.03 * x + 0.01 * y) / (x + y) * (x + y - w) + 0.02 * z_2 - 1.5 / 100 * ((x + y - w) + z_2);
    ceq = [];
end

function F = profit1(vec)
    x = vec(1);
    y = vec(2);
    w = vec(3);
    z_1 = vec(4);
    z_2 = vec(5);
    F = 9 * (w + z_1) + 15 * ((x + y - w) + z_2) - 6 * x - 16 * y - 10 * (z_1 + z_2);
    F = -F;
end

function F = profit2(vec)
    x = vec(1);
    y = vec(2);
    w = vec(3);
    z_1 = vec(4);
    z_2 = vec(5);
    F = 9 * (w + z_1) + 15 * ((x + y - w) + z_2) - 6 * x - 13 * y - 10 * (z_1 + z_2);
    F = -F;
end

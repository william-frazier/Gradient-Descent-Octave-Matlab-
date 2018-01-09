%William Frazier

%brings in the data
data = load('x15a.txt');
data = data(:, 2:end);
[m,n] = size(data);

%puts data into a more usable form
x = data(:, 1:n-1);
y = data(:, n);
X = addones(x);

%normalizes the features for use in gradient descent
[x2 mu sigma] = featureNormalize(x);
X = addones(x2);

%I increased the number of iterations and decreased alpha to see how much more 
%accurate the equation would get
num_iters = 500;
alpha = 0.05;
theta = zeros(n,1);
%j_allZeros = computeCost(X, y, theta)

axis([0 num_iters 0 j])
 
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

% Plot the convergence graph
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
theta
j = computeCost(X, y, theta)
xtest = [7.5  4200  9999  0.7];
xtest2 = xtest - mu;
xtest2 = xtest2 ./ sigma;
ytest = addones(xtest2) * theta


%normal equation stuff
x = addones(x);
theta2 = pinv(x'*x)*x'*y
j2 = computeCost(x, y, theta2)
ytest2 = addones(xtest) * theta2
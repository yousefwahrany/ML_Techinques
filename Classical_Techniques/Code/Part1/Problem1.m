close all
clear 
clc

disp("%--------------------------Before Removing Outlier-------------------------%");

% Given data
x = 1987:1996; % Years since 1987
y = [13300, 12400, 10900, 10100, 10150, 10000, 800, 9000, 8750, 8100];

% Scatterplot of the data
figure;
scatter(x, y);
xlabel('Years since 1987');
ylabel('Number of Insured Persons');
title('Scatter Plot of Insured Persons');
grid on;

% Linear fitting
p1 = polyfit(x, y, 1);
y1 = polyval(p1, x);
R2_1 = 1 - sum((y - y1).^2) / sum((y - mean(y)).^2);
figure;
scatter(x, y);
grid on;
hold on;
plot(x, y1, 'LineWidth', 2);
xlabel('Years since 1987');
ylabel('Number of Insured Persons');
title('Linear Fitting Model for the Insured Persons');

% Display the Linear Fitting equation
fprintf('The linear fitting equation is: y = %.2fx + %.2f\n', p1(1), p1(2));

% Display the R^2 Value
fprintf('For linear fitting: R^2 = %.4f\n', R2_1);

% Quadratic fitting
p2 = polyfit(x, y, 2);
y2 = polyval(p2, x);
R2_2 = 1 - sum((y - y2).^2) / sum((y - mean(y)).^2);
figure;
scatter(x, y);
grid on;
hold on;
plot(x, y2, 'LineWidth', 2);
xlabel('Years since 1987');
ylabel('Number of Insured Persons');
title('Quadratic Fitting Model for the Insured Persons');

% Display the Quadratic Fitting equation
fprintf('The Quadratic fitting equation is: y = %.2fx^2 + %.2fx + %.2f\n', p2(1), p2(2), p2(3));

% Display the R^2 Value
fprintf('For Quadratic fitting: R^2 = %.4f\n', R2_2);

% Cubic fitting
p3 = polyfit(x, y, 3);
y3 = polyval(p3, x);
R2_3 = 1 - sum((y - y3).^2) / sum((y - mean(y)).^2);
figure;
scatter(x, y);
grid on;
hold on;
plot(x, y3, 'LineWidth', 2);
xlabel('Years since 1987');
ylabel('Number of Insured Persons');
title('Cubic Fitting Model for the Insured Persons');

% Display the Cubic Fitting equation
fprintf('The Cubic fitting equation is: y = %.2fx^3 + %.2fx^2 + %.2fx + %.2f\n', p3(1), p3(2), p3(3), p3(4));

% Display the R^2 Value
fprintf('For Cubic fitting: R^2 = %.4f\n', R2_3);

% Plot all fittings
figure;
scatter(x, y);
xlabel('Years since 1987');
ylabel('Number of Insured Persons');
title('All Fitting Models for the Insured Persons');
grid on;
hold on;
plot(x, y1, 'b-', 'LineWidth', 2);
plot(x, y2, 'g--', 'LineWidth', 2);
plot(x, y3, 'm-.', 'LineWidth', 2);
legend('Data', 'Linear Fit', 'Quadratic Fit', 'Cubic Fit');

% Display R^2 values
fprintf('R^2 values:\n');
fprintf('Linear: %f\n', R2_1);
fprintf('Quadratic: %f\n', R2_2);
fprintf('Cubic: %f\n', R2_3);

% Graph the Best-Fit Function
figure;
scatter(x, y, 'ro', 'filled'); % Scatter plot
hold on;
best_p = p2;
y_best_fit = polyval(best_p, x); % Compute best-fit y values
fprintf('Best fit: Quadratic Model\n');
plot(x, y_best_fit, 'b-', 'LineWidth', 2); % Plot best-fit function
xlabel('Years since 1987');
ylabel('Number of Insured Persons');
title('Best-Fit Function with Scatterplot Before Cleaning');
grid on;
legend('Data', 'Best-Fit Function');
hold off;

disp("%--------------------------After Removing Outlier-------------------------%");

% Given data
x_no_outlier = [1987, 1988, 1989, 1990, 1991, 1992, 1994, 1995, 1996]; % Years since 1987 (1993 Removed)
y_no_outlier = [13300, 12400, 10900, 10100, 10150, 10000, 9000, 8750, 8100]; % (800 Removed)

% Scatterplot of the data
figure;
scatter(x_no_outlier, y_no_outlier);
xlabel('Years since 1987');
ylabel('Number of Insured Persons');
title('Scatter Plot of Insured Persons (After Removing Outlier)');
grid on;
ylim([0 15000]);

% Linear fitting
p1_no_outlier = polyfit(x_no_outlier, y_no_outlier, 1);
y1_no_outlier = polyval(p1_no_outlier, x_no_outlier);
R2_1_no_outlier = 1 - sum((y_no_outlier - y1_no_outlier).^2) / sum((y_no_outlier - mean(y_no_outlier)).^2);
figure;
scatter(x_no_outlier, y_no_outlier);
grid on;
hold on;
plot(x_no_outlier, y1_no_outlier, 'LineWidth', 2);
xlabel('Years since 1987');
ylabel('Number of Insured Persons');
title('Linear Fitting Model for the Insured Persons - (After Removing Outlier)');
ylim([0 15000]);

% Display the Linear Fitting equation
fprintf('The linear fitting equation is: y = %.2fx + %.2f\n', p1_no_outlier(1), p1_no_outlier(2));

% Display the R^2 Value
fprintf('For linear fitting: R^2 = %.4f\n', R2_1_no_outlier);

% Quadratic fitting
p2_no_outlier = polyfit(x_no_outlier, y_no_outlier, 2);
y2_no_outlier = polyval(p2_no_outlier, x_no_outlier);
R2_2_no_outlier = 1 - sum((y_no_outlier - y2_no_outlier).^2) / sum((y_no_outlier - mean(y_no_outlier)).^2);
figure;
scatter(x_no_outlier, y_no_outlier);
grid on;
hold on;
plot(x_no_outlier, y2_no_outlier, 'LineWidth', 2);
xlabel('Years since 1987');
ylabel('Number of Insured Persons');
title('Quadratic Fitting Model for the Insured Persons - (After Removing Outlier)');
ylim([0 15000]);

% Display the Quadratic Fitting equation
fprintf('The Quadratic fitting equation is: y = %.2fx^2 + %.2fx + %.2f\n', p2_no_outlier(1), p2_no_outlier(2), p2_no_outlier(3));

% Display the R^2 Value
fprintf('For Quadratic fitting: R^2 = %.4f\n', R2_2_no_outlier);

% Cubic fitting
p3_no_outlier = polyfit(x_no_outlier, y_no_outlier, 3);
y3_no_outlier = polyval(p3_no_outlier, x_no_outlier);
R2_3_no_outlier = 1 - sum((y_no_outlier - y3_no_outlier).^2) / sum((y_no_outlier - mean(y_no_outlier)).^2);
figure;
scatter(x_no_outlier, y_no_outlier);
grid on;
hold on;
plot(x_no_outlier, y3_no_outlier, 'LineWidth', 2);
xlabel('Years since 1987');
ylabel('Number of Insured Persons');
title('Cubic Fitting Model for the Insured Persons - (After Removing Outlier)');
ylim([0 15000]);

% Display the Cubic Fitting equation
fprintf('The Cubic fitting equation is: y = %.2fx^3 + %.2fx^2 + %.2fx + %.2f\n', p3_no_outlier(1), p3_no_outlier(2), p3_no_outlier(3), p3_no_outlier(4));

% Display the R^2 Value
fprintf('For Cubic fitting: R^2 = %.4f\n', R2_3_no_outlier);

% Plot all fittings
figure;
scatter(x_no_outlier, y_no_outlier);
xlabel('Years since 1987');
ylabel('Number of Insured Persons');
title('Scatter Plot of Insured Persons (After Removing Outlier)');
grid on;
hold on;
plot(x_no_outlier, y1_no_outlier, 'b-', 'LineWidth', 2);
plot(x_no_outlier, y2_no_outlier, 'g--', 'LineWidth', 2);
plot(x_no_outlier, y3_no_outlier, 'm-.', 'LineWidth', 2);
legend('Data', 'Linear Fit', 'Quadratic Fit', 'Cubic Fit');
ylim([0 15000]);

fprintf('After Removing Outlier \n');
% Display R^2 values
fprintf('R^2 values:\n');
fprintf('Linear: %f\n', R2_1_no_outlier);
fprintf('Quadratic: %f\n', R2_2_no_outlier);
fprintf('Cubic: %f\n', R2_3_no_outlier);

% Graph the Best-Fit Function
figure;
scatter(x_no_outlier, y_no_outlier, 'ro', 'filled'); % Scatter plot
hold on;
best_p_no_outlier = p2_no_outlier;
y_best_fit_no_outlier = polyval(best_p_no_outlier, x_no_outlier); % Compute best-fit y values
fprintf('Best fit: Quadratic Model\n');
plot(x_no_outlier, y_best_fit_no_outlier, 'b-', 'LineWidth', 2); % Plot best-fit function
xlabel('Years since 1987');
ylabel('Number of Insured Persons');
title('Best-Fit Function with Scatterplot After Cleaning');
grid on;
legend('Data', 'Best-Fit Function');
hold off;

% Predict for 1997
x_pred = 1997; % 1997
y_pred = polyval(best_p_no_outlier, x_pred);
fprintf('Predicted insured persons in 1997: %.2f\n', y_pred);


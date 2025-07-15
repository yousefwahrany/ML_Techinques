clc; 
clear;
close all;

disp("%--------------------------Before Removing Outlier-------------------------%");

% Given data
Oil        = [270, 362, 162, 45, 91, 233, 372, 305, 234, 122, 25, 210, 450, 325, 52]';
Temp       = [40,  27,  40,  73, 65, 65,  10,  9,   24,  65,  66, 41,  22,  40,  60]';
Insulation = [4,   4,   10,  6,  7,  40,  6,   10,  10,  4,  10,  6,   4,   4,   10]';

% Combine independent variables into matrix X
X = [Temp, Insulation];

% Scatterplot of the data
figure;
scatter3(Temp, Insulation, Oil, 'ro', 'filled'); % 3D scatter plot  
xlabel('Temperature (°F)');
ylabel('Insulation (inches)');
zlabel('Oil Consumption');
title('Scatter Plot of Data');
grid on;

% Linear Model: Oil = a*Temp + b*Insulation + c
[linearModel, gofLinear] = fit(X, Oil, 'poly11');

% Extract coefficients of the linear model
coeffsLinear = coeffvalues(linearModel);
a = coeffsLinear(1);
b = coeffsLinear(2);
c = coeffsLinear(3);

% Print Linear Model equation
fprintf('Coefficient for Temp      : %.2f\n', a);
fprintf('Coefficient for Insulation: %.2f\n', b);
fprintf('Constant term             : %.2f\n', c);
fprintf('Linear Fit Equation: Oil = %.2f * Temp + %.2f * Insulation + %.2f\n', a, b, c);

% Graph the Data and Linear Model Fit  
figure;
scatter3(Temp, Insulation, Oil, 'ro', 'filled'); % 3D scatter plot  
hold on;
xlabel('Temperature (°F)');
ylabel('Insulation (inches)');
zlabel('Oil Consumption');
title('Linear Regression Fit');
grid on;
plot(linearModel, X, Oil);


% Quadratic Model: Oil = a*T^2 + b*I^2 + c*T*I + d*T + e*I + f
[quadraticModel, gofQuad] = fit(X, Oil, 'poly22');

% Extract coefficients of the quadratic model
coeffsQuad = coeffvalues(quadraticModel);
a2 = coeffsQuad(1); % Coefficient for Temp^2
b2 = coeffsQuad(2); % Coefficient for Insulation^2
c2 = coeffsQuad(3); % Coefficient for Temp*Insulation
d2 = coeffsQuad(4); % Coefficient for Temp
e2 = coeffsQuad(5); % Coefficient for Insulation
f2 = coeffsQuad(6); % Constant term

% Print Quadratic Model equation
fprintf('Coefficient for Temp^2         : %.2f\n', a2);
fprintf('Coefficient for Insulation^2   : %.2f\n', b2);
fprintf('Coefficient for Temp*Insulation: %.2f\n', c2);
fprintf('Coefficient for Temp           : %.2f\n', d2);
fprintf('Coefficient for Insulation     : %.2f\n', e2);
fprintf('Constant term                  : %.2f\n', f2);
fprintf(['Quadratic Fit Equation: Oil = %.2f * Temp^2 + %.2f * Insulation^2 + %.2f * (Temp * Insulation)\n' ...
         '                              + %.2f * Temp + %.2f * Insulation + %.2f\n'], ...
         a2, b2, c2, d2, e2, f2);

% Graph the Data and the Quadratic Model Fit  
figure;
scatter3(Temp, Insulation, Oil, 'ro', 'filled'); % 3D scatter plot  
hold on;
xlabel('Temperature (°F)');
ylabel('Insulation (inches)');
zlabel('Oil Consumption');
title('Quadratic Regression Fit');
grid on;
plot(quadraticModel, X, Oil);

% Display R^2 values for both Models
fprintf('R^2 for Linear Model: %f\n', gofLinear.rsquare);
fprintf('R^2 for Quadratic Model: %f\n', gofQuad.rsquare);

disp("%--------------------------After Removing Outlier-------------------------%");

Oil_cleaned        = [270, 362, 162, 45, 91, 372, 305, 234, 122, 25, 210, 450, 325, 52]';
Temp_cleaned       = [40,  27,  40,  73, 65, 10,  9,   24,  65,  66, 41,  22,  40,  60]';
Insulation_cleaned = [4,   4,   10,  6,  7,  6,   10,  10,  4,  10,  6,   4,   4,   10]';

% Combine independent variables into matrix X
X_cleaned = [Temp_cleaned Insulation_cleaned];

% Linear Model: Oil = a*Temp + b*Insulation + c
[linearModel_cleaned, gofLinear_cleaned] = fit(X_cleaned, Oil_cleaned, 'poly11');

% Extract coefficients of the linear model
coeffsLinear_cleaned = coeffvalues(linearModel_cleaned);
a_cleaned = coeffsLinear_cleaned(1);
b_cleaned = coeffsLinear_cleaned(2);
c_cleaned = coeffsLinear_cleaned(3);

% Print Linear Model equation
fprintf('Coefficient for Temp      : %.2f\n', a_cleaned);
fprintf('Coefficient for Insulation: %.2f\n', b_cleaned);
fprintf('Constant term             : %.2f\n', c_cleaned);
fprintf('Linear Fit Equation: Oil = %.2f * Temp + %.2f * Insulation + %.2f\n', a_cleaned, b_cleaned, c_cleaned);

% Graph the Data and Linear Model Fit    
figure;
scatter3(Temp_cleaned, Insulation_cleaned, Oil_cleaned, 'ro', 'filled'); % 3D scatter plot  
hold on;
xlabel('Temperature (°F)');
ylabel('Insulation (inches)');
zlabel('Oil Consumption');
title('Linear Regression Fit After Cleaning');
grid on;
plot(linearModel_cleaned, X_cleaned, Oil_cleaned);


% Quadratic Model: Oil = a*T^2 + b*I^2 + c*T*I + d*T + e*I + f
[quadraticModel_cleaned, gofQuad_cleaned] = fit(X_cleaned, Oil_cleaned, 'poly22');

% Extract coefficients of the quadratic model
coeffsQuad_cleaned = coeffvalues(quadraticModel_cleaned);
a2_cleaned = coeffsQuad_cleaned(1); % Coefficient for Temp^2
b2_cleaned = coeffsQuad_cleaned(2); % Coefficient for Insulation^2
c2_cleaned = coeffsQuad_cleaned(3); % Coefficient for Temp*Insulation
d2_cleaned = coeffsQuad_cleaned(4); % Coefficient for Temp
e2_cleaned = coeffsQuad_cleaned(5); % Coefficient for Insulation
f2_cleaned = coeffsQuad_cleaned(6); % Constant term

% Print Quadratic Model equation
fprintf('Coefficient for Temp^2         : %.2f\n', a2_cleaned);
fprintf('Coefficient for Insulation^2   : %.2f\n', b2_cleaned);
fprintf('Coefficient for Temp*Insulation: %.2f\n', c2_cleaned);
fprintf('Coefficient for Temp           : %.2f\n', d2_cleaned);
fprintf('Coefficient for Insulation     : %.2f\n', e2_cleaned);
fprintf('Constant term                  : %.2f\n', f2_cleaned);
fprintf(['Quadratic Fit Equation: Oil = %.2f * Temp^2 + %.2f * Insulation^2 + %.2f * (Temp * Insulation)\n' ...
         '                              + %.2f * Temp + %.2f * Insulation + %.2f\n'], ...
         a2_cleaned, b2_cleaned, c2_cleaned, d2_cleaned, e2_cleaned, f2_cleaned);

% Graph the Data and the Quadratic Model Fit 
figure;
scatter3(Temp_cleaned, Insulation_cleaned, Oil_cleaned, 'ro', 'filled'); % 3D scatter plot  
hold on;
xlabel('Temperature (°F)');
ylabel('Insulation (inches)');
zlabel('Oil Consumption');
title('Quadratic Regression Fit After Cleaning');
grid on;
plot(quadraticModel_cleaned, X_cleaned, Oil_cleaned);

% Display R^2 values for Both Models
fprintf('R^2 for Linear Model: %f\n', gofLinear_cleaned.rsquare);
fprintf('R^2 for Quadratic Model: %f\n', gofQuad_cleaned.rsquare);

% Predict the needed oil for temperature is 15 Fahrenheit and the insulation is 5 attic
predicted_oil_linear_cleaned = linearModel_cleaned(15,5);
predicted_oil_quad_cleaned = quadraticModel_cleaned(15,5);
fprintf('\n\nPredicted needed oil for temp = 15 F and insulation = 5 attic:\n');
fprintf('Linear Cleaned: %f\n', predicted_oil_linear_cleaned);
fprintf('Quadratic Cleaned: %f\n\n', predicted_oil_quad_cleaned);
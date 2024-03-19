import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from cleandataset import clean_data
from math import sqrt

X = clean_data.drop(["INS", "BMI", "AGE", "Diabetic"], axis=1)
y = clean_data['AGE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=85)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = sqrt(mse)
mae = abs(y_test - y_pred).mean()
n = len(y_test)
p = X.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2) Score:", r2)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("Adjusted R-Square:", adj_r2)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.title('Actual vs. Predicted Age')
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.show()
'''
coefficients = lr_model.coef_
intercept = lr_model.intercept_
equation = f"y = {intercept:.2f} + "

for i, coef in enumerate(coefficients):
    equation += f"{coef:.2f} * x{i+1} + "

equation = equation.rstrip(' + ')

print("Linear Regression Equation:")
print(equation)
print("Model Coefficients:", lr_model.coef_)
'''
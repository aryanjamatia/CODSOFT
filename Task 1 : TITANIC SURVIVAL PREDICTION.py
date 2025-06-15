#  Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#  Load Dataset
df = pd.read_csv('advertising.csv')
print("First 5 rows:")
print(df.head())

#  Data Info & Cleaning
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

#  Feature Selection
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

#  Visualize Data Relationships
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=5, aspect=0.8, kind='reg')
plt.suptitle("Feature vs Sales Regression Plots", y=1.02)
plt.show()

#  Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train Model
model = LinearRegression()
model.fit(X_train, y_train)

#  Predict
y_pred = model.predict(X_test)

#  Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n Evaluation Metrics:")
print(" RMSE:", round(rmse, 2))
print(" R² Score:", round(r2, 2))

#  Actual vs Predicted Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()

#  Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, color='orange')
plt.title("Distribution of Residuals")
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

#  Coefficients of Features
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("\n Feature Coefficients:")
print(coefficients)

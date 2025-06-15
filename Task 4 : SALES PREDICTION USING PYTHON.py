#  Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#  Load the Dataset
df = pd.read_csv('advertising.csv')  # Make sure this file is uploaded in Colab
print(" First 5 Rows of the Dataset:")
print(df.head())

# ℹ Basic Info & Null Checks
print("\n Dataset Info:")
print(df.info())
print("\n Missing Values per Column:")
print(df.isnull().sum())

#  Feature and Label Selection
X = df[['TV', 'Radio', 'Newspaper']]  # Independent variables
y = df['Sales']                       # Dependent variable

#  Pairwise Plot of Features vs Sales
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', kind='reg', height=5)
plt.suptitle(" Feature vs Sales", y=1.02)
plt.show()

#  Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

#  Make Predictions
y_pred = model.predict(X_test)

#  Evaluate the Model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n Evaluation Metrics:")
print(" RMSE:", round(rmse, 2))
print(" R² Score:", round(r2, 2))

#  Actual vs Predicted Plot
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title(" Actual vs Predicted Sales")
plt.grid(True)
plt.show()

#  Residuals Distribution
residuals = y_test - y_pred
plt.figure(figsize=(8,5))
sns.histplot(residuals, kde=True, color='orange')
plt.title(" Distribution of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

#  Coefficients of Features
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("\n Feature Coefficients:")
print(coeff_df)

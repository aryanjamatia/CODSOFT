import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('IRIS.csv')  # Make sure the file is in the Files panel

# View dataset structure
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Features and label
X = df.drop('species', axis=1)  # Input features
y = df['species']               # Target labels

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# (Optional) Plotting pairplot
sns.pairplot(df, hue="species")
plt.suptitle("Iris Features by Species", y=1.02)
plt.show()

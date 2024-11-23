import pandas as pd #this library analysis the data 
import numpy as np # library used foe numericall operations
import matplotlib.pyplot as plt # it is using for making the graphs
import seaborn as sns # used foe statisyics
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder #calculate ,mean and sd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#all the required libraries imported firstly
data = pd.read_csv('C:/Users/himes/Desktop/dataset.csv')# reading the csv file 

data = data.loc[:, ~data.columns.str.contains('^Unnamed')]# this remove the unnamed column

# this provides info from the csv file like mean, median, standard deviation, min, max
print(data.info())
print(data.describe())

# Handle missing values if necessary
missing_values = data.isnull().sum()
print(f"Missing Values: \n{missing_values}")

# Visualizing Salary Distribution
sns.histplot(data['Salary'], kde=True, bins=30)
plt.title("Salary Distribution")
plt.show()

# Visualizing correlations
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Prepare features and target variable
X = data.drop(columns=['Salary'])  # Independent variables (features)
y = data['Salary']  # Dependent variable (target)

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing pipeline
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Created a column transformer to handle different data types
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# this will train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# this is for evaluation for model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# this visualize Actual vs Predicted salaries in graph
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Salaries')
plt.ylabel('Predicted Salaries')
plt.title('Actual vs Predicted Salaries')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.show()

# please input Years of Experience from user
n = input("Enter years of experience: ")

# taking input if it is not giveb it takes default as 5
if n == "":
    n = 5
else:
    n = int(n)

# Create a DataFrame with the proper column name
new_data = pd.DataFrame({'YearsExperience': [n]})
# this predict he salary
predicted_salary = model.predict(new_data)
print(f"Predicted Salary for {n} years of experience: {predicted_salary[0]}")

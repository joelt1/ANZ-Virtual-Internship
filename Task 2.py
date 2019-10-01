# Imports and loading dataset
print("Importing modules and loading dataset...\n")
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Reading in dataframe
df = pd.read_excel("ANZ synthesised transaction dataset.xlsx")

# Modifying data to obtain salaries for each customer
# Amount column shows annual salary
df_salaries = df[df["txn_description"] == "PAY/SALARY"].groupby("customer_id").mean()

salaries = []
for customer_id in df["customer_id"]:
    salaries.append(int(df_salaries.loc[customer_id]["amount"]))
df["annual_salary"] = salaries

df_cus = df.groupby("customer_id").mean()
print("Mean annual salary by customer: ")
print(df_cus.head(), "\n")

# PREDICTIVE ANALYTICS:
# Linear regression
print("LINEAR REGRESSION:\n")
N_train = int(len(df_cus)*0.8)
X_train = df_cus.drop("annual_salary", axis=1).iloc[:N_train]
Y_train = df_cus["annual_salary"].iloc[:N_train]
X_test = df_cus.drop("annual_salary", axis=1).iloc[N_train:]
Y_test = df_cus["annual_salary"].iloc[N_train:]

linear_reg = LinearRegression()
linear_reg.fit(X_train, Y_train)
print(f"Linear Regression Training Score: {linear_reg.score(X_train, Y_train)}\n")

print("Predictions using test data:")
print(linear_reg.predict(X_test), "\n")

print(f"Linear Regression Testing Score: {linear_reg.score(X_test, Y_test)}\n")

# Decision tree - classification and regression
# Categorical columns
df_cat = df[["txn_description", "gender", "age", "merchant_state", "movement"]]
# Changing all categories to dummies
pd.get_dummies(df_cat).head()

N_train = int(len(df)*0.8)
X_train = pd.get_dummies(df_cat).iloc[:N_train]
Y_train = df["annual_salary"].iloc[:N_train]
X_test = pd.get_dummies(df_cat).iloc[N_train:]
Y_test = df["annual_salary"].iloc[N_train:]

# Classification
print("DECISION TREE - CLASSIFIER:\n")
decision_tree_class = DecisionTreeClassifier()
decision_tree_class.fit(X_train, Y_train)
print(f"Decision Tree Classifier Training Score: {decision_tree_class.score(X_train, Y_train)}\n")

print("Predictions using test data:")
print(decision_tree_class.predict(X_test), "\n")

print(f"Decision Tree Classifier Testing Score: {decision_tree_class.score(X_test, Y_test)}\n")

# Regression
print("DECISION TREE - REGRESSOR:\n")
decision_tree_reg = DecisionTreeRegressor()
decision_tree_reg.fit(X_train, Y_train)
print(f"Decision Tree Regressor Training Score: {decision_tree_reg.score(X_train, Y_train)}\n")

print("Predictions using test data:")
print(decision_tree_reg.predict(X_test), "\n")

print(f"Decision Tree Regressor Testing Score: {decision_tree_reg.score(X_test, Y_test)}\n")

input("Press ENTER to exit: ")

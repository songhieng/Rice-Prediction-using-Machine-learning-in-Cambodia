import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import plotly.express as px

# Load your data
df = pd.read_csv('global_food_prices.csv')

# Handle missing values
numeric_columns = df.select_dtypes(include=[np.number]).columns
categorical_columns = df.select_dtypes(exclude=[np.number]).columns

df[numeric_columns] = df[numeric_columns].apply(lambda x: x.fillna(x.median()))
df[categorical_columns] = df[categorical_columns].apply(
    lambda x: x.fillna(x.mode().iloc[0]))

# Label Encoding for categorical variables
labelencoder = LabelEncoder()
categorical_columns = ['CITY', 'Product_Name', 'Unit_Name']
df[categorical_columns] = df[categorical_columns].apply(lambda series: pd.Series(
    labelencoder.fit_transform(series[series.notnull()]),
    index=series[series.notnull()].index
))

# Define features and target variable
features = ['ID', 'CITY_ID', 'Product_Name', 'Unit', 'Month', 'Year']
target = 'Price_KHR'

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.2, random_state=42)

# Fit a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
print(f'Root Mean Squared Error: {rmse}')

# LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
lin_acc = r2_score(y_test, lin_reg.predict(X_test))
print("Linear Regression")
print("Train Set Accuracy:"+str(r2_score(y_train, lin_reg.predict(X_train))*100))
print("Test Set Accuracy:"+str(r2_score(y_test, lin_reg.predict(X_test))*100))

# DecisionTreeRegressor
d_reg = DecisionTreeRegressor()
d_reg.fit(X_train, y_train)
y_pred = d_reg.predict(X_test)
d_acc = r2_score(y_test, d_reg.predict(X_test))
print("Decision Tree Regressor")
print("Train Set Accuracy:"+str(r2_score(y_train, d_reg.predict(X_train))*100))
print("Test Set Accuracy:"+str(r2_score(y_test, d_reg.predict(X_test))*100))

# RandomForestRegressor
r_reg = RandomForestRegressor()
r_reg.fit(X_train, y_train)
y_pred = r_reg.predict(X_test)
r_acc = r2_score(y_test, r_reg.predict(X_test))
print("RandomForest Regressor")
print("Train Set Accuracy:"+str(r2_score(y_train, r_reg.predict(X_train))*100))
print("Test Set Accuracy:"+str(r2_score(y_test, r_reg.predict(X_test))*100))

# KNeighborsRegressor
k_reg = KNeighborsRegressor()
k_reg.fit(X_train, y_train)
y_pred = k_reg.predict(X_test)
k_acc = r2_score(y_test, k_reg.predict(X_test))
print("KNeightbors Regressor")
print("Train Set Accuracy:"+str(r2_score(y_train, k_reg.predict(X_train))*100))
print("Test Set Accuracy:"+str(r2_score(y_test, k_reg.predict(X_test))*100))

# Support vector
s_reg = SVR()
s_reg.fit(X_train, y_train)
y_pred = s_reg.predict(X_test)
s_acc = r2_score(y_test, s_reg.predict(X_test))
print("Support Vector")
print("Train Set Accuracy:"+str(r2_score(y_train, s_reg.predict(X_train))*100))
print("Test Set Accuracy:"+str(r2_score(y_test, s_reg.predict(X_test))*100))

# Columns of the prediction
models = pd.DataFrame({
    'Model': ['Linear regression', 'Decision tree regression', 'RandomForestRegressor', 'KNeighborsRegressor', 'SVR'],
    'Score': [lin_acc, d_acc, r_acc, k_acc, s_acc]
})

models.sort_values(by='Score', ascending=False)

# Predict for next year
next_year_data = pd.DataFrame({
    'ID': [806],
    'CITY_ID': [0],  # It should be 'CITY_ID' instead of 'CITY'
    # It should be 'Product_Name' instead of 'Product_ID'
    'Product_Name': [96],
    'Unit': [5],  # It should be 'Unit' instead of 'Unit_Size' and 'Unit_Name'
    'Month': [8],
    'Year': [2018]
})

predicted_price = model.predict(next_year_data)
print(f'Predicted price for next year in USD: {predicted_price[0]}')


# Plot the scores
px.bar(models, x='Model', y='Score', color='Model').show()

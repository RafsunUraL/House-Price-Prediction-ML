from sklearn.linear_model import LinearRegression
import pandas as pd

data = pd.read_csv("house_data.csv")

X = data[['area', 'bedrooms']]
y = data['price']

model = LinearRegression()
model.fit(X, y)

print("Model trained successfully")

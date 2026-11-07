import os
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("dataset/cloud_cost.csv")
features = ["cpu", "memory", "network", "storage", "running_hours"]
X, y = data[features], data["monthly_cost"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("RMSE:", round(mean_squared_error(y_test, predictions) ** 0.5, 2))
print("R2:", round(r2_score(y_test, predictions), 3))

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/cloud_cost_model.pkl")
print("Model saved to model/cloud_cost_model.pkl")

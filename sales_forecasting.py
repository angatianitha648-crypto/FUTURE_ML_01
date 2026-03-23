import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Step 1: Create dataset
data = {
    "Month": [1,2,3,4,5,6,7,8,9,10,11,12],
    "Sales": [100,120,130,150,170,160,180,200,210,230,250,270]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# Step 2: Prepare data
X = df[['Month']]
y = df['Sales']

# Step 3: Train model
model = LinearRegression()
model.fit(X, y)

# Step 4: Predict existing (for evaluation)
y_pred = model.predict(X)

# Step 5: Calculate error
error = mean_absolute_error(y, y_pred)
print("Model Error (MAE):", error)

# Step 6: Predict future sales
future_months = [[13],[14],[15]]
future_predictions = model.predict(future_months)

print("Future Predictions:")
for i, val in enumerate(future_predictions, start=13):
    print(f"Month {i}: {val:.2f}")

# Step 7: Visualization
plt.figure(figsize=(8,5))
plt.scatter(X, y, label="Actual Sales")
plt.plot(X, y_pred, linestyle='dashed', label="Predicted Line")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales Forecasting")
plt.legend()
plt.show()

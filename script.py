import torch
import pandas as pd

# Load data
data = pd.read_csv("simple_regression_data.csv")

# Extract features and target
X = data[["Feature1", "Feature2"]].values
y = data["Target"].values

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

import torch.nn as nn

# Define a linear regression model
model = nn.Linear(2, 1)  # 2 input features, 1 output

# Mean Squared Error Loss
criterion = nn.MSELoss()
# Stochastic Gradient Descent optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # Forward pass
    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    test_input = torch.tensor([[6, 12]], dtype=torch.float32)  # Example input
    prediction = model(test_input)
    print(f"Prediction for input [6, 12]: {prediction.item():.2f}")

# Save the model
torch.save(model.state_dict(), "model.pth")
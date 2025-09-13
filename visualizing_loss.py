import sys
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

df = pd.read_csv("./data/used_cars.csv")

age = df["model_year"].max() - df["model_year"]

milage = df["milage"]
milage = milage.str.replace(",", "")
milage = milage.str.replace("mi.", "")
milage = milage.astype(int)

price = df["price"]
price = price.str.replace("$", "")
price = price.str.replace(",", "")
price = price.astype(int)

X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)
])
x_mean = X.mean(axis=0)
x_std = X.std(axis=0)
X = (X - x_mean) / x_std
# print(X)
# sys.exit()

Y = torch.tensor(price, dtype=torch.float32)\
    .reshape((-1, 1))
y_mean = Y.mean()
y_std = Y.std()
Y = (Y - y_mean) / y_std

model = nn.Linear(2, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

losses = []
for i in range(0, 500):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, Y)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    # if i % 100 == 0:
    #     print(loss.item())

plt.plot(losses)
plt.show()

X_data = torch.tensor([
    [5, 10000],
    [10, 10000],
    [5, 20000]
], dtype=torch.float32)

prediction = model((X_data - x_mean) / x_std)
print(prediction * y_std + y_mean)

    
import sys
import os
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

if not os.path.isdir("./model"):
    os.mkdir("./model")

X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)
])
x_mean = X.mean(axis=0)
x_std = X.std(axis=0)
torch.save(x_mean, "./model/x_mean.pt")    
torch.save(x_std, "./model/x_std.pt")
X = (X - x_mean) / x_std

Y = torch.tensor(price, dtype=torch.float32)\
    .reshape((-1, 1))
y_mean = Y.mean()
y_std = Y.std()
torch.save(y_mean, "./model/y_mean.pt")    
torch.save(y_std, "./model/y_std.pt")
Y = (Y - y_mean) / y_std

model = nn.Linear(2, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for i in range(0, 500):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, Y)
    loss.backward()
    optimizer.step()

    # if i % 100 == 0:
    #     print(loss.item())

torch.save(model.state_dict(), "./model/model.pt")



    
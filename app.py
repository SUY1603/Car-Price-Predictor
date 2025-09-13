import sys
import torch
from torch import nn
import gradio as gr

x_mean = torch.load("./model/x_mean.pt")
x_std = torch.load("./model/x_std.pt")
y_mean = torch.load("./model/y_mean.pt")
y_std = torch.load("./model/y_std.pt")

model = nn.Linear(2, 1)
model.load_state_dict(
    torch.load("./model/model.pt")
)
model.eval()

def predict_price(age, milage):
    X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)
    ])

    with torch.no_grad():
        prediction = model((X - x_mean) / x_std)
        return (prediction * y_std + y_mean).item()

demo = gr.Interface(
    fn=predict_price,
    inputs=[ 
        gr.Slider(minimum=0, maximum=20, label="Age"),
        gr.Slider(minimum=0, maximum=100000, label="Milage")
    ],
    outputs=gr.Text(label="Predicted Price (USD)"),
    title="Used Car Price Predictor",
    description="Predict the price of a used car based on its age and mileage.",
    flagging_mode="never",
)

demo.launch(share=True)
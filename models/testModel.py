from mlmodel import Volume
from torchsummary import summary
import torch
from siamese import Siamese

# model = Siamese()
# summary(model, [(1, 105, 105), (1, 105, 105)])

model = Volume()
model.eval()
input_data = torch.randn(1, 1, 105, 105)
out = model.forward(input_data, input_data)
print(out)

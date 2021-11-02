from tqdm import trange
from tqdm import tqdm
from sklearn.metrics import precision_score, f1_score
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, data_loader, criterion, optimizer, scheduler, num_epochs=5):
    for epoch in trange(num_epochs, desc="Epochs"):
        result = []
        for phase in ["train", "val"]:
            if phase == "train":  # put the model in training mode
                model.train()
                scheduler.step()
            else:  # put the model in validation mode
                model.eval()

            # keep track of training and validation loss
            running_loss = 0.0
            running_corrects = 0.0

            for data, data2, target in data_loader[phase]:
                # load the data and target to respective device
                data, data2, target = data.to(device), data2.to(device), target.to(device)

                with torch.set_grad_enabled(phase == "train"):
                    # feed the input
                    output = model.forward(data, data2)
                    # calculate the loss
                    loss = criterion(output, target)
                    preds = torch.sigmoid(output).data > 0.5
                    preds = preds.to(torch.float32)

                    if phase == "train":
                        # backward pass: compute gradient of the loss with respect to model parameters
                        loss.backward()
                        # update the model parameters
                        optimizer.step()
                        # zero the grad to stop it from accumulating
                        optimizer.zero_grad()

                # statistics
                running_loss += loss.item() * data.size(0)
                running_corrects += (
                    f1_score(
                        target.to("cpu").to(torch.int).numpy(),
                        preds.to("cpu").to(torch.int).numpy(),
                        average="samples",
                    )
                    * data.size(0)
                )

            epoch_loss = running_loss / len(data_loader[phase].dataset)
            epoch_acc = running_corrects / len(data_loader[phase].dataset)

            result.append(
                "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
            )
        print(result)

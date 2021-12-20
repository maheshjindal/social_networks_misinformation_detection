import torch
import torch.nn.functional as F


def train_model(model, training_dataset_loader, device, optimizer):
    """
    Trains the model

    :param model: Pytorch model instance
    :param training_dataset_loader: Pytorch training dataset loader
    :param device: Pytorch Device instance
    :param optimizer: Pytorch Optimizer instance
    :return: Model loss after training completion
    """
    model.train()
    total_loss = 0
    for data in training_dataset_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(training_dataset_loader.dataset)


@torch.no_grad()
def evaluate_model(model, dataset_loader, device):
    """
    Evaluates the model performance using a particular dataset

    :param model: Pytorch model instance
    :param dataset_loader: Datasetloader instance on which model performance needs to be evaluated
    :param device: Pytorch device instance
    :return: Pytorch model accuracy on test data
    """
    model.eval()
    total_correct = total_examples = 0
    for data in dataset_loader:
        data = data.to(device)
        predictions = model(data).argmax(dim=-1)
        total_correct += int((predictions == data.y).sum())
        total_examples += data.num_graphs
    return total_correct / total_examples

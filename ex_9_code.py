import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib.legend_handler import HandlerLine2D
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
from torchvision import datasets, models
from sklearn.metrics import confusion_matrix
import numpy as np

USE_CUDA = torch.cuda.is_available()
criterion = nn.CrossEntropyLoss()


class MyNet(nn.Module):
    def __init__(self, image_size, num_of_classes):
        super(MyNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(self.image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, num_of_classes)

        self.convolution1 = nn.Conv2d(3, 50, kernel_size=5)
        self.convolution2 = nn.Conv2d(50, 16, kernel_size=5)

        self.fc0_bn = nn.BatchNorm1d(100)
        self.fc1_bn = nn.BatchNorm1d(50)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(f.relu(self.convolution1(x)))
        x = self.pool(f.relu(self.convolution2(x)))

        x = x.view(-1, self.image_size)
        x = f.relu(self.fc0_bn(self.fc0(x)))
        x = f.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)
        return f.log_softmax(x, dim=1)


def train(epoch, model, train_loader, optimizer, batch_size):
    model.train()
    train_loss = 0
    correct_train = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        if batch_idx % 100 == 0:
            print(batch_idx * batch_size / (len(train_loader) * batch_size))

        if USE_CUDA:
            data = data.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()
        output = model(data)

        # loss = f.nll_loss(output, labels, size_average=True)
        # train_loss += loss.item()
        loss = criterion(output, labels)
        train_loss += criterion(output, labels)

        loss.backward()
        optimizer.step()
        prediction = output.data.max(1, keepdim=True)[1]
        correct_train += prediction.eq(labels.data.view_as(prediction)).cpu().sum()

    train_loss /= len(train_loader)
    print('\nTraining Epoch: {}\tAccuracy {}/{} ({:.3f}%)\tAverage Loss: {:.3f}'.format(
        epoch, correct_train, (len(train_loader) * batch_size),
        100. * float(correct_train) / (len(train_loader) * batch_size), train_loss))

    return train_loss


def validate(epoch, model, valid_loader, batch_size):
    model.eval()

    validation_loss = 0
    correct_valid = 0
    for data, label in valid_loader:
        if USE_CUDA:
            data = data.cuda()
            label = label.cuda()
        output = model(data)

        validation_loss += criterion(output, label)
        # validation_loss += f.nll_loss(output, label, size_average=False).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct_valid += pred.eq(label.data.view_as(pred)).cpu().sum()

    validation_loss /= (len(valid_loader) * batch_size)
    print('Validation Epoch: {}\tAccuracy: {}/{} ({:.3f}%)\tAverage Loss: {:.3f}'.format(
        epoch, correct_valid, (len(valid_loader) * batch_size),
        100. * float(correct_valid) / (len(valid_loader) * batch_size), validation_loss))

    return validation_loss


def test(learning_model, test_loader):
    learning_model.eval()
    test_loss = 0
    correct = 0
    predictions = list()
    y_predictions = list()
    y_tag_predications = list()
    for data, target in test_loader:
        if USE_CUDA:
            data = data.cuda()
            target = target.cuda()

        output = learning_model(data)

        # Sums up the batch loss.
        test_loss += criterion(output,target)
        # test_loss += f.nll_loss(output, target, size_average=False).item()

        # Gets index of max log-probability.
        prediction = output.data.max(1, keepdim=True)[1]
        prediction_vector = prediction.view(len(prediction))
        for x in prediction_vector:
            predictions.append(x.item())
            y_predictions.append(x.item())
        correct += prediction.eq(target.data.view_as(prediction)).cpu().sum()

        for t in target:
            y_tag_predications.append(t.item())

    conf_matrix = confusion_matrix(y_tag_predications,y_predictions)
    print(conf_matrix)


    test_loss /= len(test_loader.dataset)
    print('\nTesting Set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return predictions


def get_data_loaders(batch_size, validation_ratio, transforms_compose):
    # Loads data sets.
    train_data_set = datasets.CIFAR10('./train_data', train=True, download=True, transform=transforms_compose)
    test_data_set = datasets.CIFAR10('./test_data', train=False, download=True, transform=transforms_compose)

    # Splits train data set to the corresponding validation ratio.
    train_size = len(train_data_set)
    indices = list(range(train_size))
    split = int(validation_ratio * train_size)

    valid_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(valid_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(valid_idx)

    train_ldr = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size, sampler=train_sampler, num_workers=2)

    validation_ldr = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size, sampler=validation_sampler,
                                                 num_workers=2)
    test_ldr = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_ldr, validation_ldr, test_ldr


def draw_loss(x, train_y, valid_y):
    fig = plt.figure(0)
    fig.canvas.set_window_title('Training Loss vs. Validation Loss')

    plt.axis([0, 11, 0.25, 1.75])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    train_graph, = plt.plot(x, train_y, 'm.:', label='Training Loss')

    valid_graph, = plt.plot(x, valid_y, 'k.-', label='Validation Loss')

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.legend(handler_map={train_graph: HandlerLine2D(numpoints=3)})
    plt.show()


def write_to_file(predictions):
    with open('test.pred', 'w') as f:
        for p in predictions:
            f.write(str(p) + '\n')
    f.close()


def my_train(num_of_epochs, model, batch_size, optimizer, training_validation_ratio, transforms_compose):
    train_loader, validation_loader, test_loader = get_data_loaders(batch_size, training_validation_ratio,
                                                                    transforms_compose)
    # Trains model and validates.
    x = list()
    train_y = list()
    valid_y = list()
    for epoch in range(num_of_epochs):
        train_loss = train(epoch, model, train_loader, optimizer, batch_size)
        valid_loss = validate(epoch, model, validation_loader, batch_size)
        x.append(epoch)
        train_y.append(train_loss)
        valid_y.append(valid_loss)

    predictions = test(model, test_loader)

    write_to_file(predictions)
    draw_loss(x, train_y, valid_y)


def main():
    # Settings
    num_of_classes = 10
    image_size = 400

    # My Net
    my_net_model = MyNet(image_size, num_of_classes)
    my_net_optimizer = optim.Adam(my_net_model.parameters(), 0.001)
    if USE_CUDA:
        my_net_model = my_net_model.cuda()

    # ResNet
    resnet_model = models.resnet18(pretrained=True)
    for param in resnet_model.parameters():
        param.requires_grad = False

    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, num_of_classes)

    resnet_optimizer = optim.SGD(resnet_model.fc.parameters(), 0.1)
    if USE_CUDA:
        resnet_model = resnet_model.cuda()
        resnet_model.fc.cuda()

    # Trains MyNet
    # my_net_t = transforms.Compose([transforms.ToTensor(),
    #                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    # my_train(10, my_net_model, 256, my_net_optimizer, 0.2, my_net_t)

    # Trains ResNet
    resnet_t = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    my_train(2, resnet_model, 100, resnet_optimizer, 0.2, resnet_t)


if __name__ == '__main__':
    main()


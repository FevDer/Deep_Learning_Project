Cloth Classifier using CNN in PyTorch


This project implements a cloth classifier using Convolutional Neural Networks (CNN) in PyTorch. The model is trained and evaluated on the Fashion MNIST dataset.

Overview

This repository contains the code for a cloth classifier implemented in PyTorch. The model is designed to classify images of clothing items into 10 different categories using Convolutional Neural Networks.

Dataset

The Fashion MNIST dataset is used for training and testing the cloth classifier. It consists of 28x28 grayscale images of 10 fashion categories.

train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())


Model Architecture


The cloth classifier model is built with a Convolutional Neural Network (CNN) using PyTorch. The architecture includes convolutional layers, ReLU activation, max-pooling, and a fully connected layer.

class MultiClassImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MultiClassImageClassifier, self).__init__()
        # Define layers here
        ...

    def forward(self, x):
        # Forward pass
        ...


Training the Model


The model is trained using the training dataset with the Adam optimizer and CrossEntropyLoss.

dataloader_train = DataLoader(train_data, shuffle=True, batch_size=10)
net = MultiClassImageClassifier(num_classes)
optimizer = optim.Adam(net.parameters(), lr=0.001)
train_model(optimizer, net, num_epochs=1)


Evaluating the Model


The model is evaluated on the test set using metrics like accuracy, precision, and recall.

dataloader_test = DataLoader(test_data, shuffle=False, batch_size=10)

# Define the metrics
accuracy_metric = Accuracy(task='multiclass', num_classes=num_classes)
precision_metric = Precision(task='multiclass', num_classes=num_classes, average=None)
recall_metric = Recall(task='multiclass', num_classes=num_classes, average=None)

# Run model on the test set
net.eval()
predicted = []
for i, (features, labels) in enumerate(dataloader_test):
    output = net.forward(features.reshape(-1, 1, image_size, image_size))
    cat = torch.argmax(output, dim=-1)
    predicted.extend(cat.tolist())
    accuracy_metric(cat, labels)
    precision_metric(cat, labels)
    recall_metric(cat, labels)

# Compute the metrics
accuracy = accuracy_metric.compute().item()
precision = precision_metric.compute().tolist()
recall = recall_metric.compute().tolist()
print('Accuracy:', accuracy)
print('Precision (per class):', precision)
print('Recall (per class):', recall)


Feel free to customize this README according to your specific project details and requirements.


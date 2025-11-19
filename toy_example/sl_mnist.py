#!/usr/bin/env python3

import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm

def prepare_dataset():
    # ------Setting up device-------
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ------import data from kaggle using api------
    # setting up kaggle API
    BASEDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
    if os.path.exists(os.path.expanduser('~/.kaggle')):
        print("Kaggle API file already exists, skipping setup.")
    elif os.path.exists(os.path.join(BASEDIR, 'kaggle.json')):
        os.system('mkdir ~/.kaggle')
        os.system('cp kaggle.json ~/.kaggle')
    else:
        print('please download the kaggle api file and try again!')
        exit()

    data_dir = 'data/sign-language-mnist'
    if os.path.exists(data_dir):
        print("sign-language-mnist dataset is ready!")
    else:
        # Downloading the dataset directly from kaggle
        os.system('kaggle datasets download datamunge/sign-language-mnist')

        # Unzipping the data
        os.system('unzip sign-language-mnist.zip -d data/sign-language-mnist')

    train_set = pd.read_csv('data/sign-language-mnist/sign_mnist_train.csv')
    test_set = pd.read_csv('data/sign-language-mnist/sign_mnist_test.csv')

    train_set.head(5)

    # ------Image data------
    X = train_set.drop(['label'], axis = 1)
    y = train_set['label']

    print(X.shape, y.shape)
    global X_train, X_val, y_train, y_val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    im = np.array(X_train.iloc[2])
    im = im.reshape((28,28,1))
    plt.imshow(im)
    plt.show()

    # ------experimental pipeline------
    wandb.login()
    config = dict(
        epochs= 20,           # no of epochs
        classes=24 +1,        # classes (total 24 letters plus additional None)
        image_size = 28,      # size of the image
        kernels=[16, 32],     # kernel size for each layer in CNN, you can tweak this according to your need
        batch_size=32,
        learning_rate=0.005,
        architecture="CNN"
    )

    return config
def pre_transform():
    random_transforms = transforms.Compose([
        transforms.RandomRotation(30),  # Randomly rotate the image by up to 30 degrees
        # transforms.RandomResizedCrop(IMAGE_SIZE),  # Randomly crop and resize the image to 224x224
        # transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    ])

    # Define the fixed transformations
    fixed_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Define the overall transformation pipeline
    transform = transforms.Compose([
        transforms.RandomApply([random_transforms], p=0.5),  # Apply random transformations with a probability of 0.5
        fixed_transforms
    ])

    return transform
class SignDataSet(Dataset):
  def __init__(
      self,
      image_df, 
      label_df,
      transform,
      split = None,
  ):
    self.image_df = image_df 
    self.label_df = torch.nn.functional.one_hot(torch.tensor(np.array(label_df))).float()
    self.split = split 
    self.transform = transform

  def __len__(self):
    return len(self.label_df)
  
  def __getitem__(self, index):
    image = self.image_df.iloc[index]
    image = np.reshape(np.array(image), (28,28))

    image = Image.fromarray(image.astype(np.uint8))

    label = self.label_df[index]
    # label = torch.nn.functional.one_hot(torch.tensor(label))

    if self.split == 'train':
      image = self.transform(image)

    if self.split == 'test':
      image = self.transform(image)
    return image, label
            

def make_loader(x, y, config, mode):
    transform = pre_transform()
    data = SignDataSet(x, y, transform, mode)
    data_loader = DataLoader(data, batch_size = config['batch_size'], drop_last = True)
    return data_loader

class SignLabelModel(nn.Module):
    def __init__(self, kernels, num_classes):
        super(SignLabelModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, kernels[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(kernels[0], kernels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def model_creation(config):
    train_loader = make_loader(X_train, y_train, config, 'train')
    test_loader = make_loader(X_val, y_val, config, 'test')

    model = SignLabelModel(config.kernels, config.classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    return model, train_loader, test_loader, criterion, optimizer

def train_batch(images, labels, model, optimizer, criterion):
    images, labels = images.to(device), labels.to(device)

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Step with opimizer
    optimizer.step()

    return loss

def train_log(loss, example_ct, epoch):
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)}  exmaples: {loss:.3f}")

def train_model(model, loader, criterion, optimizer, config):
    wandb.watch(model, criterion, log="all", log_freq=10)
    total_batches = len(loader) * config.epochs
    print(total_batches)
    example_ct = 0
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):
            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct += len(images)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)
def test_model(model, test_loader):
    model.eval()

    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            labels = [torch.argmax(l).item() for l in labels]
            # print(predicted.shape, labels)
            # correct += (predicted == labels).sum().item()
            count = sum([1 for i in range(len(labels)) if labels[i] == predicted[i].item()])

        print(f"Accuracy of the model on the {total} " +
              f"test images: {correct / total:%}")
        
        wandb.log({"test_accuracy": correct / total})

    # Save the model in the exchangeable ONNX format
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")

def model_pipeline(hyperparameters):
    
    with wandb.init(project='MNIST_viz', config=hyperparameters):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer = model_creation(config)
        print(model)

        train_model(model, train_loader, criterion, optimizer, config)
        test_model(model, test_loader)

    return model

if __name__ == "__main__":
    config = prepare_dataset()
    model = model_pipeline(config)
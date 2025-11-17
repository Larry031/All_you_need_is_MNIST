#!/usr/bin/env python3

import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import Dataset, DataLoader
from torch.nn import nn

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
        os.system('kaggle datasets download sign-language-mnist')

        # Unzipping the data
        os.makedirs(data_dir, exist_ok=True)
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
def make_loader(x, y, config, mode):
    data = SignDataSet(x, y, transform, mode)
    data_loader = DataLoader(data, batch_size = config['batch_size'], drop_last = True)
    return data_loader
def model_creation(config):
    train_loader = make_loader(X_train, y_train, config, 'train')
    test_loader = make_loader(X_val, y_val, config, 'test')

    model = SignLableModel(config.kernels, config.classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    return model, train_loader, test_loader, criterion, optimizer
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
#!/usr/bin/env python3

import os
import torch

# ------Setting up device-------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------import data from kaggle using api------
# setting up kaggle API
if os.path.exists(os.path.expanduser('~/.kaggle')):
    print("Kaggle API file already exists, skipping setup.")
else:
    os.system('mkdir ~/.kaggle')
    os.system('cp kaggle.json ~/.kaggle')

data_dir = 'data/sign-language-mnist'
if os.path.exists(data_dir):
    print("sign-language-mnist dataset is ready!")
else:
    # Downloading the dataset directly from kaggle
    os.system('kaggle datasets download -d datamunge/sign-language-mnist')

    # Unzipping the data
    os.system('unzip sign-language-mnist.zip -d data/sign-language-mnist')
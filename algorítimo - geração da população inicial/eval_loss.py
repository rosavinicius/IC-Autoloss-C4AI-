import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from copy import deepcopy as dc
from lstm import LSTM
from utils import load_LSTM_dataset, save_model, train_one_epoch, validate_one_epoch
from utils import load_custom_loss
from utils import load_LSTM_dataset, save_model
from eval_loss import custom_loss, custom_loss_composed, custom_loss_composed_ponderated
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from lstm import X_test, y_test
from lstm import X_train, y_train
from lstm import X_val, y_val


num_epochs = 10
with open('../data/population_lossFramework.json', 'r') as file:
    Losses = json.load(file)
loss = list(Losses.values())
loss_functions = {}
for index, indiv in enumerate(loss):
    loss_functions[index] = str(load_custom_loss(f"{indiv.lossFunction}"))


def custom_loss(loss_function):
    def loss(y_true, y_pred):
        loss = loss_function(y_pred, y_true)
        return torch.mean(loss)
    return loss

def custom_loss_composed(loss_function):
    def loss(y_true, y_pred):
        mse_loss = torch.mean((y_true - y_pred) ** 2)
        loss = loss_function(y_pred, y_true)
        return torch.mean(loss) + mse_loss
    return loss

def custom_loss_composed_ponderated(loss_function):
    def loss(y_true, y_pred):
        mse_loss = torch.mean((y_true - y_pred) ** 2)
        loss = torch.mean(loss_function(y_pred, y_true))
        return 0.4 * mse_loss + 0.6 * loss
    return loss

if __name__ == '__main__':

    for evolution_step, loss_string_repr in loss_functions.items():
        # Loading custom loss functin
        loss = load_custom_loss(loss_string_repr)
        print(loss)

        # -------------------------------- custom_loss --------------------------------
        print("Tranining model_1")
        model_1 = LSTM(1, 4, 1)
        learning_rate = 0.001
        epochs = 200

        for epoch in range(num_epochs):
            train_one_epoch(model_1, loss)
            validate_one_epoch(model_1, loss)

        save_model(f"../outputs/{evolution_step}_trainedModel_customLoss", model_1)


        # -------------------------------- custom_loss composed --------------------------------
        print("Tranining model_2")
        model_2 = LSTM(1, 4, 1)

        learning_rate = 0.001
        epochs = 200
        optimizer = torch.optim.Adam(model_1.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            train_one_epoch(model_2, custom_loss_composed(loss))
            validate_one_epoch(model_2, custom_loss_composed(loss))

        save_model(f"../outputs/{evolution_step}_trainedModel_composed", model_2)



        #  -------------------------------- custom_loss composed ponderated --------------------------------
        print("Tranining model_3")
        model_3 = LSTM(1, 4, 1)

        learning_rate = 0.001
        epochs = 200
        optimizer = torch.optim.Adam(model_1.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            train_one_epoch(model_3, custom_loss_composed_ponderated(loss))
            validate_one_epoch(model_3, custom_loss_composed_ponderated(loss))

        save_model(f"../outputs/{evolution_step}_trainedModel_composed_ponderated", model_3)


        #  -------------------------------- output --------------------------------
        print("#" * 100)
        print(f"Loss function and MSE step {evolution_step}: ")
        print(loss_string_repr)
        print("MSE custom_loss: ")
        try:
            y_pred = model_1(torch.Tensor(X_test))
            mse = nn.MSELoss()(torch.Tensor(y_test), y_pred).item()
            print(mse)
        except ValueError:
            print("ValueError!")

        print("MSE custom_loss + composed: ")
        try:
            y_pred = model_2(torch.Tensor(X_test))
            mse = nn.MSELoss()(torch.Tensor(y_test), y_pred).item()
            print(mse)
        except ValueError:
            print("ValueError!")

        print("MSE custom_loss + composed + ponderated: ")
        try:
            y_pred = model_3(torch.Tensor(X_test))
            mse = nn.MSELoss()(torch.Tensor(y_test), y_pred).item()
            print(mse)
        except ValueError:
            print("ValueError!")

        print("#" * 100)

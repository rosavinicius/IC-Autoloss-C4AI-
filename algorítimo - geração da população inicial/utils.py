import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.optim import Optimizer
import torch
import torch.nn as nn
import torch.optim as optim # Import the optim module from PyTorch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lstm import prepare_dataframe_for_lstm
from loss import PRIMITIVE_OPERATIONS
from loss import ComputationalGraph
from loss import Node
from copy import deepcopy as dc

def load_LSTM_dataset():
    lookback = 7

    data = pd.read_csv('../data/ts_sea_level.csv')
    data = data[['Date', 'Level']]
    shifted_df =prepare_dataframe_for_lstm(data, lookback)
    shifted_df_as_np = shifted_df.to_numpy()
    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]

    X = dc(np.flip(X, axis=1))

    X_train = X_train.reshape((-1, lookback, 1))
    X_test = X_test.reshape((-1, lookback, 1))

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    return X_train, X_test, y_train, y_test



def save_model(file_name='', model=None):
    print('Saving architecture and weights in {}'.format(file_name))

    # Salvar o estado completo do modelo em um único arquivo .pth
    torch.save(model.state_dict(), file_name + '.pth')


def load_model(file_path):
    model = torch.load(file_path)
    return model


class LearningRateScheduler:
    def __init__(self, optimizer: Optimizer, init_lr=0.01, schedule=[(25, 1e-2), (50, 1e-3), (100, 1e-4)]):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.schedule = schedule
        self._update_lr(self.init_lr)

    def _update_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def step(self, epoch):
        # Ajusta a taxa de aprendizado com base no epoch atual
        lr = self.init_lr
        for i in range(len(self.schedule) - 1):
            if epoch >= self.schedule[i][0] and epoch < self.schedule[i + 1][0]:
                lr = self.schedule[i][1]

        if epoch >= self.schedule[-1][0]:
            lr = self.schedule[-1][1]

        self._update_lr(lr)
        print(f'Learning rate: {lr}')


def load_custom_loss(string_repr):
    print(string_repr)
    def parse_node(tokens):
        token = tokens.pop(0)
        if token in ['prediction', 'target', 'constant']:
            return Node(operation=token)
        elif token in PRIMITIVE_OPERATIONS:
            children = []
            while tokens[0] != ')':
                children.append(parse_node(tokens))
            tokens.pop(0)  # Remove ')'
            return Node(operation=token, children=children)
        elif token == '(':
            return parse_node(tokens)
        else:
            raise ValueError("Invalid token: " + token)

    tokens = string_repr.replace('(', ' ( ').replace(')', ' ) ').split()
    root = parse_node(tokens)
    return ComputationalGraph(root=root)


# Operações primitivas disponíveis
PRIMITIVE_OPERATIONS_PYTORCH = {
    'Add': torch.add,             # Soma
    'Mul': torch.mul,             # Multiplicação
    'Neg': torch.neg,             # Negação
    'Abs': torch.abs,             # Valor absoluto
    'Inv': torch.reciprocal,      # Inverso (1/x)
    'Log': torch.log,             # Logaritmo natural
    'Exp': torch.exp,             # Exponencial
    'Tanh': torch.tanh,           # Tangente hiperbólica
    'Square': lambda x: x ** 2,   # Quadrado (não existe direto, usamos x ** 2)
    'Sqrt': torch.sqrt            # Raiz quadrada
}

def evaluate_node(node, prediction, target):
    if node.operation == 'prediction':
        return prediction
    elif node.operation == 'target':
        return target
    elif node.operation in PRIMITIVE_OPERATIONS_PYTORCH:
        # Avalia os filhos recursivamente e aplica a operação correspondente
        evaluated_children = [evaluate_node(child, prediction, target) for child in node.children]
        # Executa a operação com os filhos avaliados
        return PRIMITIVE_OPERATIONS_PYTORCH[node.operation](*evaluated_children)
    else:
        raise ValueError(f"Operação desconhecida: {node.operation}")

def custom_loss_function(graph, prediction, target):
    # Avalia o grafo para cada amostra e calcula a média das perdas
    batch_losses = evaluate_node(graph.root, prediction, target)
    return batch_losses.mean()  # Retorna a média das perdas para o lote
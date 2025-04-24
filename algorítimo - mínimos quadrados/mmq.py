
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from lstm import LSTM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
import json
from lstm import model_LSTM
from lstm import device
from scipy.linalg import lstsq
import math 

data = pd.read_csv('../data/ts_sea_level.csv')
data = data[['Date', 'Level']]
data['Date'] = pd.to_datetime(data['Date'])

from copy import deepcopy as dc

def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)
    df.set_index('Date', inplace=True)

    for i in range(1, n_steps + 1):
        df[f'Level(t-{i})'] = df['Level'].shift(i)

    df.dropna(inplace=True)
    return df

lookback = 7
shifted_df = prepare_dataframe_for_lstm(data, lookback)
shifted_df_as_np = shifted_df.to_numpy()

# Escalar os dado
scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

X = shifted_df_as_np[:, 1:]
targets = shifted_df_as_np[:, 0]

# Dividir os dados
split_index = int(len(X) * 0.95)
X_train_val = X[:split_index]
targets_train_val = targets[:split_index]

# IOA como média dos alvos (substitua por seu cálculo real, se necessário)
IOA = np.mean(targets_train_val, axis=0) * np.ones_like(targets_train_val)

# Ajustar os dados para 3D: (batch_size, seq_length, input_size)
X_train_val = X_train_val.reshape((-1, lookback, 1))


# Definir operações suportadas para a avaliação dinâmica
operations = {
    "Add": lambda x, y: x + y,
    "Tanh": math.tanh,
    "Sqrt": math.sqrt,
    "Inv": lambda x: 1 / x,
    "Exp": math.exp,
    "Neg": lambda x: -x,
    "Abs": abs,
    "Log": math.log,
    "Square": lambda x: x ** 2,
    "Mul": lambda x, y: x * y
}

def evaluate_expression(expr, prediction, target):
    """
    Avalia uma expressão no formato prefixado.

    Args:
        expr (str): A expressão no formato prefixado.
        prediction (float): Valor da previsão.
        target (float): Valor do alvo.

    Returns:
        float: Resultado da avaliação da expressão.
    """
    tokens = expr.replace("(", "").replace(")", "").split()
    stack = []

    for token in reversed(tokens):
        if token in operations:
            stack.append(token)  # É uma operação
        elif token == "prediction":
            stack.append(prediction)  # Substituir por valor
        elif token == "target":
            stack.append(target)  # Substituir por valor
        else:
            stack.append(float(token))  # É um número

        # Se houver o suficiente para computar uma operação
        while len(stack) >= 3 and isinstance(stack[-1], (float, int)) and isinstance(stack[-2], (float, int)):
            arg2 = stack.pop()
            arg1 = stack.pop()
            op = stack.pop()
            result = operations[op](arg1, arg2) if op in ["Add", "Mul"] else operations[op](arg1)
            stack.append(result)

    return stack[0]  # Resultado final

# Função principal para resolver os mínimos quadrados
def least_squares_loss_functions_with_model(loss_functions, targets, X, IOA, model):
   
    # Avaliar todas as funções de perda
    parsed_functions = {key: lambda p, t, expr=expr: evaluate_expression(expr, p, t) for key, expr in loss_functions.items()}

    # Obter previsões do modelo para os dados de entrada
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    predictions = model(X_tensor).cpu().detach().numpy()

    # Montar a matriz L
    L = np.array([
        [parsed_functions[key](predictions[i, 0], targets[i]) for key in loss_functions]
        for i in range(len(targets))
    ])

    # Resolver o sistema de mínimos quadrados L * U = IOA
    U, _, _, _ = lstsq(L, IOA)

    return U

loss_functions = {
    "0": "(Mul (Square (Log prediction)) (Abs (Mul prediction target)))",
    "1": "(Sqrt (Abs (Abs (Neg (Add prediction target)))))",
    "2": "(Sqrt (Exp (Sqrt (Mul prediction target))))",
    "3": "(Add (Abs (Neg (Add (Inv prediction) (Square prediction)))) (Add (Inv (Add (Inv prediction) (Mul prediction target))) (Exp (Tanh (Tanh prediction)))))",
    "4": "(Log (Abs (Add prediction target)))"
}

# Resolver o problema de mínimos quadrados
U = least_squares_loss_functions_with_model(loss_functions, targets_train_val, X_train_val, IOA, model_LSTM)
print("Pesos (U):", U)
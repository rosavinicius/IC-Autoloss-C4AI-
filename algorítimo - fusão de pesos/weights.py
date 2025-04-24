import torch
from lstm import LSTM
from lstm import device
from utils import load_custom_loss
from utils import load_LSTM_dataset, save_model
from eval_loss import custom_loss, custom_loss_composed, custom_loss_composed_ponderated
from torch.utils.data import DataLoader
import torch.nn as nn
import json
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
import pandas as pd


model = LSTM(1, 4, 1)
model.to(device)

learning_rate = 0.001
num_epochs = 10
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
THRESHOLD = 1e6  # Ajuste conforme necessário
with open('../data/population_lossFramework.json', 'r') as file:
    Losses = json.load(file)
loss_array = list(Losses.values())


def train_one_epoch(model, optimizer, lossFunction):
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)

        # Aplica a função de perda
        loss = lossFunction(output, y_batch)

        # Se a loss retornar um tensor do tamanho da batch, tiramos a média
        if loss.dim() > 0:
            loss = loss.mean()

        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0
    print()
    return avg_loss_across_batches

def validate_one_epoch(model, lossFunction):
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
        if callable(lossFunction):
            # Se for uma função de perda padrão, aplica diretamente
            loss = lossFunction(output, y_batch)
        else:
            # Se for um gráfico computacional, usa a função personalizada
            loss = custom_loss_function(lossFunction, output, y_batch)

    # Se a loss retornar um tensor do tamanho da batch, tiramos a média
    if loss.dim() > 0:
        loss = loss.mean()

    running_loss += loss.item()
    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()

    return avg_loss_across_batches

def train(model_parameter, optimizer, loss_function_parameter):
  for epoch in range(num_epochs):
      loss_onde_epoch = train_one_epoch(model_parameter, optimizer, loss_function_parameter)
      validate_one_epoch(model_parameter, loss_function_parameter)


# Salvar o estado inicial dos pesos
initial_weights = model.state_dict()

# Define os pesos iniciais no modelo
model.load_state_dict(initial_weights)

# Treina o modelo usando MSE
train(model, nn.MSELoss())

# Salvar os pesos após o primeiro treino
u = {key: value.clone() for key, value in model.state_dict().items()}

# Inicializa delta como zero
delta = {key: torch.zeros_like(value) for key, value in u.items()}

# Lista para armazenar funções de perda inválidas (opcional)
invalid_losses = []

for loss_fn in loss_array:
    # Reseta os pesos iniciais
    model.load_state_dict(initial_weights)

    # Treina com a função de perda específica
    #train(model, load_custom_loss(loss_fn))
    for epoch in range(num_epochs):
      loss_one_epoch = train_one_epoch(model, load_custom_loss(loss_fn))
      if np.isnan(loss_one_epoch) or np.isinf(loss_one_epoch) or loss_one_epoch > THRESHOLD:
          print(f"Skipping loss function {loss_fn} due to invalid loss at epoch {epoch}.")
          break  # Sai do loop de epochs e passa para a próxima loss_fn

      validate_one_epoch(model, load_custom_loss(loss_fn))

    # Salvar os pesos após este treino
    v = {key: value.clone() for key, value in model.state_dict().items()}

    # Atualiza delta com a diferença entre os pesos
    for key in delta:
        delta[key] += u[key] - v[key]

    # Ajusta os pesos com a diferença (u + u - v)
    adjusted_weights = {key: u[key] + (u[key] - v[key]) for key in u}
    model.load_state_dict(adjusted_weights)

# Define os pesos finais no modelo
final_weights = {key: u[key] + delta[key] for key in u}
model.load_state_dict(final_weights)
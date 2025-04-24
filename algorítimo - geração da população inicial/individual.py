import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error as MSE

class IndividualLSTM(nn.Module):
    def __init__(self, model, X, y, input_size, hidden_size, num_stacked_layers, lossFunction):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lossFunction = lossFunction
        self.device = 'cpu'
        self.fitness = None
        self.valid_loss_flag = None
        self.model = model


        # Define the LSTM and fully connected layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

        # Prepare input and target tensors
        self.X = X
        self.y = y

        # Compile model with optimizer and custom loss function
        self.compileModel()

    def compileModel(self):
        # Define the optimizer (Adam) with a learning rate of 0.001
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = self.customLoss

    def customLoss(self, y_true, y_pred):
        # Compute custom loss using the provided `lossFunction`
        loss = self.lossFunction.evaluate(y_pred, y_true)
        return torch.mean(loss)

    def is_valid_loss(self, losses, max_loss=70):
        # Check if the loss values meet validity criteria
        if len(losses) < 2:
            self.valid_loss_flag = 0
            return

        # Check if any loss value is negative, NaN, or exceeds max_loss
        for loss in losses:
            if loss < 0 or np.isnan(loss) or loss > max_loss:
                self.valid_loss_flag = 0
                return

        # Check if the loss value remains the same for 90% of the epochs
        unique_losses = set(losses)
        if len(unique_losses) <= 0.1 * len(losses):
            self.valid_loss_flag = 0
            return

        self.valid_loss_flag = 1
        return

    def modelFit(self, epochs=10):
        # Store loss history for validation
        losses = []
        print("Input shape:", self.X.shape)
        print("Output shape:", self.y.shape)

        for epoch in range(epochs):
            self.train()
            self.optimizer.zero_grad()

            # Forward pass
            y_pred = self(self.X)
            loss = self.criterion(self.y, y_pred)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Store current loss for validation
            losses.append(loss.item())

        # Check if the loss is valid
        self.is_valid_loss(losses)

    def forward(self, x):
        # Forward pass through LSTM and fully connected layer
        h0 = torch.zeros(self.num_stacked_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def calculateFitness(self, X_test, y_test):
        # Calculate model fitness using Mean Squared Error
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(self.device)
            y_pred = self(X_test)
            self.fitness = MSE(y_test.cpu().numpy(), y_pred.cpu().numpy())
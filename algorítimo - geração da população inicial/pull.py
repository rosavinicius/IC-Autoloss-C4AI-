from individual import IndividualLSTM
import random
from utils import load_model
from lstm import X_train, y_train
from eval_loss import initialize_loss_function


def initialPull(X, y, individualsNumber, maxDepth, baseModel):
  population = []
  i = 0
  while i < individualsNumber:
      tree_depth = random.randint(4, maxDepth)
      print(i)
      indiv = IndividualLSTM(baseModel, X, y, input_size=1, hidden_size=4, num_stacked_layers=1, lossFunction = initialize_loss_function(depth=tree_depth))
      indiv.modelFit()
      loss_flag = indiv.valid_loss_flag
      if loss_flag:
          population.append(indiv)
          i = i + 1
  # population = [Individual(X, y, treeDepth) for i in range(individualsNumber)]
  return population


max_tree_depth = 6
indiv_numbers = 100

baseModel = load_model("../data/BaseLSTMModel.pth")
initialPull = initialPull(X_train, y_train, indiv_numbers, max_tree_depth, baseModel)
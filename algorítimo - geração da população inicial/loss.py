import torch
import random

PRIMITIVE_OPERATIONS = ['Add', 'Mul', 'Neg', 'Abs', 'Inv', 'Log', 'Exp', 'Tanh', 'Square', 'Sqrt']

class Node:
    def __init__(self, operation=None, children=None):
        self.operation = operation
        self.children = children if children else []

    def evaluate(self, context):
        if self.operation == 'Add':
            return self.children[0].evaluate(context) + self.children[1].evaluate(context)
        elif self.operation == 'Mul':
            return self.children[0].evaluate(context) * self.children[1].evaluate(context)
        elif self.operation == 'Neg':
            return -self.children[0].evaluate(context)
        elif self.operation == 'Abs':
            return torch.abs(self.children[0].evaluate(context))
        elif self.operation == 'Inv':
            return 1.0 / (self.children[0].evaluate(context) + 1e-12)
        elif self.operation == 'Log':
            return torch.sign(self.children[0].evaluate(context)) * torch.log(torch.abs(self.children[0].evaluate(context)) + 1e-12)
        elif self.operation == 'Exp':
            return torch.exp(self.children[0].evaluate(context))
        elif self.operation == 'Tanh':
            return torch.tanh(self.children[0].evaluate(context))
        elif self.operation == 'Square':
            return torch.square(self.children[0].evaluate(context))
        elif self.operation == 'Sqrt':
            return torch.sign(self.children[0].evaluate(context)) * torch.sqrt(torch.abs(self.children[0].evaluate(context)) + 1e-12)
        elif self.operation == 'constant':
            return 1.0
        elif self.operation in ['prediction', 'target']:
            return context[self.operation]
        else:
            raise ValueError("Unknown operation")

    def __repr__(self):
        if self.operation in ['prediction', 'target', 'constant']:
            return self.operation
        elif self.operation in ['Add', 'Mul']:
            return f"({self.operation} {self.children[0]} {self.children[1]})"
        else:
            return f"({self.operation} {self.children[0]})"

class ComputationalGraph:
    def __init__(self, root=None):
        self.root = root

    def evaluate(self, prediction, target):
        context = {'prediction': prediction, 'target': target}
        return self.root.evaluate(context)

    def __repr__(self):
        return repr(self.root)


def initialize_node(depth=5, has_prediction=False, has_target=False):
    if depth == 0:
        if not has_prediction:
            return Node(operation='prediction')
        elif not has_target:
            return Node(operation='target')
        else:
            return Node(operation=random.choice(['constant']))
    if depth == 1:
        if not has_prediction:
            return Node(operation='prediction')
        elif not has_target:
            return Node(operation='target')
    operation = random.choice(PRIMITIVE_OPERATIONS)
    if operation in ['Add', 'Mul']:
        child1 = initialize_node(depth - 1, has_prediction, has_target)
        has_prediction = has_prediction or (child1.operation == 'prediction')
        has_target = has_target or (child1.operation == 'target')
        child2 = initialize_node(depth - 1, has_prediction, has_target)
        has_prediction = has_prediction or (child2.operation == 'prediction')
        has_target = has_target or (child2.operation == 'target')
        return Node(operation=operation, children=[child1, child2])
    else:
        child = initialize_node(depth - 1, has_prediction, has_target)
        has_prediction = has_prediction or (child.operation == 'prediction')
        has_target = has_target or (child.operation == 'target')
        return Node(operation=operation, children=[child])


def initialize_loss_function(depth=5):
    root = initialize_node(depth, has_prediction=False, has_target=False)

    if not any(n.operation == 'prediction' for n in iterate_nodes(root)):
        root = Node(operation='Add', children=[Node(operation='prediction'), root])
    if not any(n.operation == 'target' for n in iterate_nodes(root)):
        root = Node(operation='Add', children=[Node(operation='target'), root])

    return ComputationalGraph(root=root)

def iterate_nodes(node):
    yield node
    for child in node.children:
        yield from iterate_nodes(child)



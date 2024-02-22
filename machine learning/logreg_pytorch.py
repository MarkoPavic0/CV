import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import data

class PTLogreg(nn.Module):
    def __init__(self, D, C):
        super(PTLogreg, self).__init__()
        self.W = nn.Parameter(torch.randn(C, D))
        self.b = nn.Parameter(torch.zeros(C))

    def forward(self, X):
        scores = torch.mm(X, self.W.t()) + self.b
        probs = torch.softmax(scores, dim=1)
        return probs

    def get_loss(self, X, Yoh_):
        N = X.shape[0]
        probs = self.forward(X)
        logprobs = torch.log(probs)
        loss = -(1/N) * torch.sum(logprobs * Yoh_)
        return loss

def train(model, X, Yoh_, param_niter, param_delta):
    optimizer = optim.SGD(model.parameters(), lr=param_delta)

    for i in range(param_niter):
        optimizer.zero_grad()
        loss = model.get_loss(X, Yoh_)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"iteration {i}: loss {loss.item()}")

def eval(model, X):
    X_tensor = torch.Tensor(X)
    probs = model(X_tensor).detach().numpy()
    return probs

if __name__ == "__main__":
    np.random.seed(100)

    X1, Y_ = data.sample_gauss_2d(3, 100)

    X = torch.Tensor(X1)
    C = np.max(Y_) + 1
    Yoh_2 = np.eye(C)[Y_.flatten()]
    Yoh_ = torch.from_numpy(Yoh_2)

    ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])

    train(ptlr, X, Yoh_, 10000, 0.05)

    probs = eval(ptlr, X)

    Y = np.argmax(probs, axis=1).reshape(-1, 1)

    accuracy, confusion, precision, recall = data.eval_perf_multi(Y, Y_)
    Yr = sorted(Y_)
    AP = data.eval_AP(Yr)
    print (accuracy, confusion, precision, recall, AP)

    decfun = lambda X: np.argmax(eval(ptlr, X), axis=1)
    bbox=(torch.min(X, dim=0)[0], torch.max(X, dim=0)[0])
    data.graph_surface(decfun, bbox, 0.5, 256, 256)

    data.graph_data(X, Y_, Y)

    plt.show()
import numpy as np
import matplotlib.pyplot as plt
import data

def logreg_train(X, Y_):
    N = X.shape[0]
    D = X.shape[1]
    C = np.max(Y_) + 1

    W = np.random.randn(C, D)
    b = np.zeros((C, 1))

    param_niter = 10000
    param_delta = 0.05

    for i in range(param_niter):
        scores = np.dot(X, W.T) + b.T
        expscores = np.exp(scores)
        
        sumexp = np.sum(expscores, axis=1, keepdims=True)

        probs = expscores / sumexp
        logprobs = np.log(probs)

        loss = -(1/N) * np.sum(logprobs[range(N), Y_.flatten()])
        
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        dL_ds = probs.copy()
        dL_ds[range(N), Y_.flatten()] -= 1 
        gs = dL_ds

        grad_W = (1/N) * np.dot(gs.T, X)
        grad_b = (1/N) * np.sum(gs, axis=0, keepdims=True).T

        W += -param_delta * grad_W
        b += -param_delta * grad_b

    return W, b


def logreg_classify(X, W, b):
    scores = np.dot(X, W.T) + b.T
    expscores = np.exp(scores)
    sumexp = np.sum(expscores, axis=1, keepdims=True)
    probs = expscores / sumexp

    return probs

def logreg_decfun(W, b):
    def classify(X):
      return np.argmax(logreg_classify(X, W, b), axis=1)
    return classify

if __name__=="__main__":
    np.random.seed(100)

    X, Y_ = data.sample_gauss_2d(3, 100)

    W, b = logreg_train(X, Y_)

    probs = logreg_classify(X, W, b)

    Y = np.argmax(probs, axis=1).reshape(-1, 1)

    accuracy, confusion, precision, recall = data.eval_perf_multi(Y, Y_)
    Yr = sorted(Y_)
    AP = data.eval_AP(Yr)
    print (accuracy, confusion, precision, recall, AP)

    decfun = logreg_decfun(W, b)

    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, 0.5, 256, 256)
    
    data.graph_data(X, Y_, Y)

    plt.show()
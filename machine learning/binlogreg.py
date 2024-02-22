import numpy as np
import matplotlib.pyplot as plt
import data

def stable_softmax(x):
    exp_x_shifted = np.exp(x - np.max(x))
    probs = exp_x_shifted / np.sum(exp_x_shifted)
    return probs

def sigma_func(x):
    return np.exp(x) / (1 + np.exp(x))

def binlogreg_train(X,Y_):
    N = X.shape[0]
    D = X.shape[1]
    w = np.random.randn(D, 1)
    b = 0

    param_niter = 1000
    param_delta = 0.01
  
    for i in range(param_niter):
        scores = np.dot(X, w) + b

        probs = sigma_func(scores)

        loss = (-1/N) * np.sum(Y_ * np.log(probs) + (1 - Y_) * np.log(1 - probs))
        
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        dL_dscores = probs - (Y_ == 1)
        gs = dL_dscores

        dscores_dw = np.transpose(X)
        dscores_db = 1

        grad_w = (1/N) * np.dot(X.T, gs)
        grad_b = (1/N) * np.sum(gs)

        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return w, b

def binlogreg_classify(X, w, b):
    scores = np.dot(X, w) + b
    probs = sigma_func(scores)
    return probs

def binlogreg_decfun(w,b):
    def classify(X):
      return binlogreg_classify(X, w, b)
    return classify

if __name__=="__main__":
    np.random.seed(200)

    X, Y_ = data.sample_gauss_2d(2, 100)

    w, b = binlogreg_train(X, Y_)

    probs = binlogreg_classify(X, w, b)
    Y = (probs >= 0.5).astype(int)

    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    Yr = sorted(Y_)
    AP = data.eval_AP(Yr)
    print (accuracy, recall, precision, AP)

    decfun = binlogreg_decfun(w,b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, 0.5, 256, 256)
    
    data.graph_data(X, Y_, Y)

    plt.show()
import numpy as np
import matplotlib.pyplot as plt
import random

class Random2DGaussian:
    minx = 0
    maxx = 10
    miny = 0
    maxy = 10

    def __init__(self):
        self.mu = None
        self.sigma = None

    def get_sample(self, n):
        self.mu = np.random.random_sample(2) * 10

        eigvalx = (np.random.random_sample()*(Random2DGaussian.maxx - Random2DGaussian.minx)/5)**2
        eigvaly = (np.random.random_sample()*(Random2DGaussian.maxy - Random2DGaussian.miny)/5)**2
        D = np.diag([eigvalx, eigvaly])

        theta = np.random.random_sample()*2*np.pi
        R = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]

        self.sigma = np.dot(np.dot(np.transpose(R), D), R)
        samples = np.random.multivariate_normal(self.mu, self.sigma, size=n)
        return samples

def sample_gauss_2d(C, N):
    X = np.zeros((N * C, 2))
    Y_2 = np.zeros((N * C, 1))

    for i in range(C):
        gaussian = Random2DGaussian()
        sample = gaussian.get_sample(N)
        
        for j in range(N):
            X[i * N + j] = sample[j]
            Y_2[i * N + j] = i
    
    Y_ = Y_2.astype(int)
    return X, Y_

def eval_perf_binary(Y, Y_):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for y, y_ in zip(Y, Y_):
        if y == 1 and y_ == 1:
            TP += 1
        elif y == 0 and y_ == 1:
            FP += 1
        elif y == 0 and y_ == 0:
            TN += 1
        elif y == 1 and y_ == 0:
            FN += 1

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return accuracy, precision, recall

def eval_perf_multi(Y, Y_):
    N = np.max(Y_) + 1
    confusion_matrix = np.zeros((N, N), dtype=int)

    for y, y_ in zip(Y, Y_):
        confusion_matrix[y, y_] += 1

    accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)

    return accuracy, confusion_matrix, precision, recall

def eval_AP(Yr):
    N = len(Yr)
    TP = np.sum(Yr)
    FP = N - TP   

    sum1 = 0
    sum2 = np.sum(Yr)

    for i in Yr:
        if TP + FP != 0:
            precision = TP / (TP + FP)
        else:
            precision = 0

        if i:
            sum1 += precision

        i2 = np.array(i)
        TP -= i2.item()
        FP -= not i2.item()

    if sum2 != 0:
        return sum1/sum2
    else:
        return 0
    
def graph_data(X, Y_, Y):
    colors2 = ["gray" if y_ == 0 else "white" for y_ in Y_]
    colors = np.array(colors2)

    tocno_klasificirano = np.where(Y_ == Y)[0]

    netocno_klasificirano = np.where(Y_ != Y)[0]

    plt.scatter(x = X[tocno_klasificirano, 0], y = X[tocno_klasificirano, 1],
                c = colors[tocno_klasificirano], marker = 'o', edgecolors = 'black')
    
    plt.scatter(x = X[netocno_klasificirano, 0], y = X[netocno_klasificirano, 1],
                c = colors[netocno_klasificirano], marker = 's', edgecolors = 'black')

def graph_surface(fun, rect, offset, width, height):
    x = np.linspace(rect[0][0], rect[1][0], width) 
    y = np.linspace(rect[0][1], rect[1][1], height)
    
    X, Y = np.meshgrid(x, y)

    grid = np.stack((X.flatten(), Y.flatten()), axis=1)

    Z = fun(grid).reshape((width, height))
    
    maxval = max(np.max(Z) - offset, - (np.min(Z) - offset))
    
    plt.pcolormesh(X, Y, Z, vmin = offset - maxval, vmax = offset + maxval)
        
    plt.contour(X, Y, Z, colors = "black", levels = [offset])

def myDummyDecision(X):
    scores = X[:,0] + X[:,1] - 5
    return scores

if __name__=="__main__":
    np.random.seed(100)

    X,Y_ = sample_gauss_2d(2, 100)
  
    Y = myDummyDecision(X)>0.5
    
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(myDummyDecision, bbox, 0.5, 256, 256)

    graph_data(X, Y_, Y) 
  
    plt.show()
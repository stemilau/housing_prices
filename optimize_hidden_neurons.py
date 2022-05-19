import numpy as np

from main import generatePredictions
import matplotlib.pyplot as plt

loss = []
med_hidden = []

def goldenSearchNumberOfHiddenNeurons(x_lower, x_upper, epsilon=0.05):
    c = x_upper - 0.618 * (x_upper - x_lower)
    d = x_lower + 0.618 * (x_upper - x_lower)
    no_iter = 0
    while abs((x_lower - x_upper)) > epsilon:
        c = x_lower - 0.618 * (x_lower - x_upper)
        d = x_upper + 0.618 * (x_lower - x_upper)

        predict_c, val_loss = generatePredictions(0.8, round(c), True)
        predict_d = generatePredictions(0.8, round(d), False)

        if predict_c < predict_d:
            x_upper = x_upper
            x_lower = round(d)
        else:
            x_lower = x_lower
            x_upper = round(c)
        no_iter += 1
        loss.append(val_loss)
        med_hidden.append((x_upper+x_lower)/2)
    return x_lower, no_iter


if __name__ == "__main__":
    hidden_neurons, no_iter = goldenSearchNumberOfHiddenNeurons(8, 32, 1)
    print(f"Number of hidden neurons found by golden search: {hidden_neurons}, in {no_iter} iterations.")
    mse, hist = generatePredictions(0.8, hidden_neurons, True)

    loss = np.asarray(loss)
    med_hidden = np.asarray(med_hidden)
    print(loss)
    print(med_hidden)
    plt.plot(med_hidden,loss,'o')
    plt.show()

    print(f"With a MSE of :{mse}")

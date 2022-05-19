import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd


def readAndParseDataset(train_data_percentage):
    # The csvFile is of type list[list]
    # where each nested list contains 13 numerical values,
    # that signify properties of interest for a house. The length of the dataset is of 506 instances.
    csvFile = pd.read_csv("./Houses.csv").values
    np.random.shuffle(csvFile)  # Shuffling data to prevent the model to learn the order of the training variable.

    # Separating the data from the labels
    # The network has to predict the houses price values in thousands of dollars (with respect to the medValue)
    x_data = [entry[0:13] for entry in csvFile]  # list comprehension slicing an entry into the first 12 parameters
    y_data = [entry[13] for entry in csvFile]  # and here isolating the 13th parameter (medValue)

    # Separating the data into training data and testing data
    # where training_data = 80% of init data, and testing_data = 20% of init data.
    len_train_data = int(train_data_percentage * len(x_data))

    x_train_data = np.asarray(x_data[0:len_train_data])  # slicing the data from start to the 80%
    y_train_data = np.asarray(y_data[0:len_train_data])  # then converting to ndarray

    x_test_data = np.asarray(x_data[len_train_data:])  # slicing the data from 80% to 100%
    y_test_data = np.asarray(y_data[len_train_data:])  # then converting to ndarray
    return len_train_data, (x_train_data, y_train_data), (x_test_data, y_test_data)


def defineModel(number_of_neurons_in_hidden_layer):
    model = Sequential()  # Defining our model as Sequential, because it is a linear stack of layers.

    # Adding the input Layer, where we specify the number of features as 13, and the number of neurons as 8,
    # with the activation function as relu.
    model.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))

    # Adding a hidden Layer, with a number of 16 neurons.
    model.add(Dense(number_of_neurons_in_hidden_layer, kernel_initializer='normal'))

    # Adding an output layer, as a single neuron, which represents the predicted value for a House.
    # There is no activation function for this layer because it is a regression problem, and we are
    # interested in predicting numerical values.
    model.add(Dense(1, kernel_initializer='normal'))

    # We compile the model using the optimizer adam, and mse as a loss function.
    model.compile(loss="mean_squared_error", optimizer="adam", metrics="accuracy")

    return model


def meanSquaredError(y_test_data, predictions, len_train_data):
    mse = 0
    # 506 is the total number of data entries in the dataset. By subtracting
    # the length of the training data, we obtain the length of the testing data.
    for i in range(506 - len_train_data):
        mse += 1 / 100 * (y_test_data[i] - predictions[i]) ** 2
    return mse


def generatePredictions(percentage_of_data_for_training, number_of_neurons_in_hidden_layer, return_loss):
    len_train_data, (x_train_data, y_train_data), (x_test_data, y_test_data) = readAndParseDataset(
        percentage_of_data_for_training)
    ann = defineModel(number_of_neurons_in_hidden_layer)

    # Training the model with a number of iterations over the training dataset of 100,
    # and a number of samples per gradient update of 16. Setting verbose = 2 for more logging
    # info during the training process.
    ann.fit(x_train_data, y_train_data, epochs=100, batch_size=16, verbose=0)

    val_loss, val_acc = ann.evaluate(x_test_data, y_test_data)

    predictions = ann.predict(x_test_data)

    if (return_loss is True):
        return meanSquaredError(y_test_data, predictions, len_train_data), val_loss
    else:
        return meanSquaredError(y_test_data, predictions, len_train_data)

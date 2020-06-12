# neuralnetworksproject

Author: Juraj Síč

This is an implementation of neural network for classifing [pictures of handwritten digits](http://yann.lecun.com/exdb/mnist/) for a project in [PV021 Neural Networks](https://is.muni.cz/predmet/fi/podzim2020/PV021). The

```
./RUN
```

script compiles the sources and runs the training of neural network where it is assumed that training input data are in files `data/mnist_train_vectors.csv` and `data/mnist_train_labels.csv` and the test data are in `data/mnist_test_vectors.csv`. Each line of `data/mnist_train_vectors.csv` should encode the picture of the digit, with pixel represented as a value between 0 and 255. Each line of `data/mnist_train_labels.csv` is then a corresponding value of the handwritten digit. The script results in two files:

- trainPredictions - output labels for training vectors

- actualTestPredictions - output labels for test vectors

The program can be also run on its own as

```
src/.nn [size of layers] [starting learning rate] [number of epochs] [output file for train labels] [output file for test labels]
```

where [size of layers] is a sequence of numbers which represents the number of hidden layers in each layer (the number of layers is variable).

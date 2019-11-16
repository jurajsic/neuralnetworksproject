# neuralnetworksproject

Author: Juraj Síč (433287)

This is an implementation of neural network for classifing pictures of handwritten digits. The

./RUN

script compiles the sources and runs the training of neural network which results in two files:

trainPredictions - output labels for training vectors

actualTestPredictions - output labels for test vectors

The program can be run on its own as

src/.nn [size of layers] [starting learning rate] [number of epochs] [output file for train labels] [output file for test labels]

where [size of layers] is a sequence of numbers which represents the number of hidden layers in each layer (the number of layers is variable).
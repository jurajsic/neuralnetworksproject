CXX = g++
CXXFLAGS = -std=c++17 -pedantic -Wall -Wextra -g
RM = rm -f

SRCS = neuron.cpp neuronconnection.cpp neuralnetwork.cpp main.cpp 
OBJS = $(subst .cpp,.o,$(SRCS))

all: nn

main.o: main.cpp

neuron.o: neuron.hpp neuron.cpp

neuronconnection.o: neuron.hpp neuronconnection.cpp

neuralnetwork.o: neuralnetwork.hpp neuralnetwork.cpp

nn: $(OBJS)
	$(CXX) $(CXXFLAGS) -o nn $(OBJS)

clean:
	$(RM) $(OBJS)

distclean: clean
	$(RM) nn
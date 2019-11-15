CXX = g++
CXXFLAGS = -std=c++17 -pedantic -Wall -Wextra
DEBOPT = -o3
RM = rm -f

SRCS = neuron.cpp neuronconnection.cpp neuralnetwork.cpp main.cpp 
OBJS = $(subst .cpp,.o,$(SRCS))

all: CXXFLAGS += -o3
all: nn

debug: CXXFLAGS += -g
debug: nn

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
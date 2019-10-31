
CXX = clang++
CXXFLAGS = -std=c++17 -O3 -Wall
LIBS =
LDFLAGS =

NN: NN.o
	@echo "Building application NN"
	${CXX} ${CXXFLAGS} ${LIBS} ${LDFLAGS} -o NN.out NN.o

NN.o: NN.cpp
	@echo "Building object file NN.o"
	${CXX} ${CXXFLAGS} ${LIBS} ${LDFLAGS} -c -o NN.o NN.cpp

clean:
	rm NN.o
	rm NN.out

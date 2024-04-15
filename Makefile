CPP := $(wildcard *.cpp) $(wildcard */*.cpp)
INCLUDE_FOLDER := ./include
FLAGS := -Wall

main: $(CPP)
	g++ $(CPP) -o main -I$(INCLUDE_FOLDER) $(FLAGS)


.SUFFIXES : .cpp.o
CC = mpiicpc
CFLAGS = -O2 -std=c++11 -fPIC -fp-model=precise
CFLAGS += -Wall
ARCH = 
EXES = main.e
OBJS = tdma_parallel.o main.o
SRCS = $(OBJS:.o=.cpp)

all: $(EXES)

$(EXES): $(OBJS)
	$(CC) -o $(EXES) $(OBJS)

.cpp.o:
	$(CC) $(CFLAGS) -c $< -o $@

main.o: main.cpp tdma_parallel.h
tdma_parallel.o: tdma_parallel.h tdma_parallel.cpp

clean:
	rm -f *.o $(EXES)



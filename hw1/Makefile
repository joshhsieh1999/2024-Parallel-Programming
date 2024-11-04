CC = mpicc
CXX = mpicxx

CXXFLAGS = -O3 -lm
CFLAGS = -O3 -lm

TARGETS = hw1

SOURCES = hw1.cc

.PHONY: all
all: $(TARGETS)

hw1: $(SOURCES)
	$(CXX) $(CXXFLAGS) -o hw1 $(SOURCES)

.PHONY: clean
clean:
	rm -f $(TARGETS)

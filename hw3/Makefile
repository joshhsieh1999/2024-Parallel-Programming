CC = gcc
CXX = g++
NVCC = nvcc
HIPCC = hipcc

CXXFLAGS = -lm -O3 -ffast-math -march=native -mtune=native -funroll-loops -falign-loops -finline-functions -fopenmp
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" \
          -arch=sm_61 \
          -Xptxas -warn-spills \
          -Xcompiler "-fopenmp -march=native -funroll-loops"
HIPCCFLAGS = -std=c++11 -O3 --offload-arch=gfx90a -fopenmp -march=native -funroll-loops

LDFLAGS = -lm
EXES = hw3-1 hw3-2 hw3-3

.PHONY: all clean

all: $(EXES)

clean:
	rm -f $(EXES) *_profile.* tmp.opt *.hip *.cu.hip

# hw3-1 target
hw3-1: hw3-1.cc
	$(CXX) $(CXXFLAGS) -o $@ $^

# hw3-2 targets
hw3-2: hw3-2.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-2-amd: hw3-2.hip
	$(HIPCC) $(HIPCCFLAGS) $(LDFLAGS) -o $@ $<

# hw3-3 targets
hw3-3: hw3-3.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-3-amd: hw3-3.hip
	$(HIPCC) $(HIPCCFLAGS) $(LDFLAGS) -o $@ $<

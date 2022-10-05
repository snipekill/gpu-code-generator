SRCS = gpu_run.cuh gpu_run.cu main.cu kernel.cu

REQUIRED_FILES = main.cu
export KERNEL ?= kernel.cu

cnn: $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o, $^) $(LDFLAGS)

clean:
	$(RM) cnn

CXX = nvcc # specify your compiler here
LDFLAGS += # specify your library linking options here
CXXFLAGS += -std=c++11 -O3 $(LDFLAGS)



# Makefile for Seeker
#
# Author : Jean-Luc PONS, Chapoly1305, William Flores

SRC = IntGroup.cpp main.cpp Random.cpp \
 Timer.cpp Int.cpp IntMod.cpp Point.cpp SECP224r1.cpp \
 Vanity.cpp GPU/GPUGenerate.cpp \
 hash/sha256.cpp hash/sha512.cpp \
 hash/sha256_sse.cpp Wildcard.cpp PrefixThreadPool.cpp
OBJDIR = obj

ifdef gpu
OBJET = $(addprefix $(OBJDIR)/, \
 IntGroup.o main.o Random.o Timer.o Int.o \
 IntMod.o Point.o SECP224r1.o Vanity.o GPU/GPUGenerate.o \
 hash/sha256.o hash/sha512.o \
 hash/sha256_sse.o \
 GPU/GPUEngine.o Wildcard.o PrefixThreadPool.o)
else
OBJET = $(addprefix $(OBJDIR)/, \
 IntGroup.o main.o Random.o Timer.o Int.o \
 IntMod.o Point.o SECP224r1.o Vanity.o \
 hash/sha256.o hash/sha512.o \
 hash/sha256_sse.o Wildcard.o PrefixThreadPool.o)
endif

CXX = g++
CUDA = /usr/local/cuda-12
CXXCUDA = g++
NVCC = $(CUDA)/bin/nvcc
# nvcc requires joint notation w/o dot, i.e. "5.2" -> "52"
ccap = $(shell echo $(CCAP) | tr -d '.')

ifdef gpu
ifdef debug
CXXFLAGS = -DWITHGPU -m64 -mssse3 -Wno-write-strings -g -I. -I$(CUDA)/include
else
CXXFLAGS = -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I$(CUDA)/include
endif
LFLAGS = -lpthread -L$(CUDA)/lib64 -lcudart
else
ifdef debug
CXXFLAGS = -m64 -mssse3 -Wno-write-strings -g -I. -I$(CUDA)/include
else
CXXFLAGS = -m64 -mssse3 -Wno-write-strings -O2 -I. -I$(CUDA)/include
endif
LFLAGS = -lpthread
endif

#--------------------------------------------------------------------

ifdef gpu
ifdef debug
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) -G -g -lineinfo                -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90 -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
else
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) -O3 --use_fast_math -Xptxas -v -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90 -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
endif
endif

$(OBJDIR)/%.o : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

all: Seeker

Seeker: $(OBJET)
	@echo Making Seeker...
	$(CXX) $(OBJET) $(LFLAGS) -o Seeker
ifndef debug
	@echo Stripping Seeker...
	strip Seeker
endif

$(OBJET): | $(OBJDIR) $(OBJDIR)/GPU $(OBJDIR)/hash

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/GPU: $(OBJDIR)
	cd $(OBJDIR) && mkdir -p GPU

$(OBJDIR)/hash: $(OBJDIR)
	cd $(OBJDIR) && mkdir -p hash

clean:
	@echo Cleaning...
	@rm -f obj/*.o
	@rm -f obj/GPU/*.o
	@rm -f obj/hash/*.o
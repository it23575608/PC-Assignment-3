# Makefile for Mandelbrot CUDA Implementation
# Course: SE3082 - Parallel Computing

# Compiler and Flags
CC = nvcc
CFLAGS = -O3 -std=c++11

# Targets
TARGET = mandelbrot_cuda
SRC = mandelbrot_cuda.cu
OBJ = $(SRC:.cu=.o)

# Default rule
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rules for Windows and Unix-like environments
clean:
	@del /f $(TARGET).exe $(OBJ) *.pgm 2>nul || rm -f $(TARGET) $(OBJ) *.pgm

.PHONY: all clean

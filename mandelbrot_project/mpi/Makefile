# Makefile for Mandelbrot MPI Implementation
# Course: SE3082 - Parallel Computing

# Compiler and Flags
CC = mpicc
CFLAGS = -O3 -Wall -Wextra -std=c99

# Targets
TARGET = mandelbrot_mpi
SRC = mandelbrot_mpi.c
OBJ = $(SRC:.c=.o)

# Default rule
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rules for Windows and Unix-like environments
clean:
	@del /f $(TARGET).exe $(OBJ) *.pgm 2>nul || rm -f $(TARGET) $(OBJ) *.pgm

.PHONY: all clean

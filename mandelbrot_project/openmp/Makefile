# Makefile for Mandelbrot OpenMP Implementation
# Course: SE3082 - Parallel Computing

# Compiler and Flags
CC = gcc
CFLAGS = -O3 -Wall -Wextra -fopenmp -std=c99

# Targets
TARGET = mandelbrot_openmp
SRC = mandelbrot_openmp.c
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

CC=gcc
CFLAGS=-c -Wall -I /usr/local/cuda/include/
LDFLAGS=-L /usr/local/cuda/lib64/ -lOpenCL
SOURCES=
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLES=devices platforms

all: $(SOURCES) $(EXECUTABLES)

devices: devices.o $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.c.o:
	$(CC) $(CFLAGS) $< -o $@

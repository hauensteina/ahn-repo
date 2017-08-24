CC=gcc
CFLAGS=-Wall -I /usr/local/cuda/include/
LDFLAGS=-L /usr/local/cuda/lib64/ -lOpenCL
SOURCES=
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLES=devices platforms

all: $(EXECUTABLES)

clean:
	rm -f $(EXECUTABLES) $(addsuffix .o,$(EXECUTABLES))

devices: $(@:=.c) $(OBJECTS)
	$(CC) $(@:=.c) $(CFLAGS) $(LDFLAGS) $(OBJECTS) -o $@

platforms: $(@:=.c) $(OBJECTS)
	$(CC) $(@:=.c) $(CFLAGS) $(LDFLAGS) $(OBJECTS) -o $@

.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

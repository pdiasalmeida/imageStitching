CC=			g++
CFLAGS=		-c -g -O0 -Wall `pkg-config --cflags opencv libconfig++`
LDFLAGS=	`pkg-config --libs opencv libconfig++`

SOURCES=	createPanorama.cpp

OBJECTS=	$(SOURCES:.cpp=.o)

all: createPanorama

createPanorama: $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJECTS) createPanorama
CC=			g++
CFLAGS=		-c -g -O0 -Wall -std=c++11 `pkg-config --cflags opencv libconfig++`
LDFLAGS=	`pkg-config --libs opencv libconfig++`

SOURCES=	src/FeatureHandler.cpp src/Stitching.cpp \
			src/createPanorama.cpp

OBJECTS=	$(SOURCES:.cpp=.o)

all: createPanorama

createPanorama: $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJECTS) createPanorama
#!/bin/bash

CC=g++

SRCDIR=src
HEADDIR=include
LIBDIR=obj
BINDIR=bin

CFLAGS= -ansi -O5 -Wall -std=c++11
LDFLAGS= -ansi -lm -Wall -std=c++11
EXEC=community convert hierarchy

SRC= $(wildcard $(SRCDIR)/*.cpp)
OBJ1= $(SRC:$(SRCDIR)/%.cpp=$(LIBDIR)/graph_binary.o) $(SRC:$(SRCDIR)/%.cpp=$(LIBDIR)/community.o)
OBJ2= $(SRC:$(SRCDIR)/%.cpp=$(LIBDIR)/graph.o)
OBJ3 = $(SRC:$(SRCDIR)/%.cpp=$(LIBDIR)/graph.o) $(SRC:$(SRCDIR)/%.cpp=$(LIBDIR)/graph_binary.o) $(SRC:$(SRCDIR)/%.cpp=$(LIBDIR)/community.o)

all: $(EXEC)
Debug: CFLAGS += -DDEBUG -g
Debug: LDFLAGS += -DDEBUG -g
Debug: $(EXEC)

community : $(OBJ1) $(SRC:$(SRCDIR)/%.cpp=$(LIBDIR)/main_community.o)
	$(CC)  -o $(BINDIR)/$@ $^ $(LDFLAGS)

convert : $(OBJ2) $(SRC:$(SRCDIR)/%.cpp=$(LIBDIR)/main_convert.o)
	$(CC)  -o $(BINDIR)/$@ $^ $(LDFLAGS)

hierarchy : $(SRC:$(SRCDIR)/%.cpp=$(LIBDIR)/main_hierarchy.o)
	$(CC)  -o $(BINDIR)/$@ $^ $(LDFLAGS)

high_degree : $(OBJ3) $(SRC:$(SRCDIR)/%.cpp=$(LIBDIR)/main_high_degree.o)
	$(CC)  -o $(BINDIR)/$@ $^ $(LDFLAGS)

inverse : $(OBJ3) $(SRC:$(SRCDIR)/%.cpp=$(LIBDIR)/main_compare.o)
	$(CC)  -o $(BINDIR)/$@ $^ $(LDFLAGS)

reverse : $(OBJ3) $(SRC:$(SRCDIR)/%.cpp=$(LIBDIR)/main_reverse.o)
	$(CC)  -o $(BINDIR)/$@ $^ $(LDFLAGS)

overlap : $(OBJ3) $(SRC:$(SRCDIR)/%.cpp=$(LIBDIR)/main_overlap.o)
	$(CC)  -o $(BINDIR)/$@ $^ $(LDFLAGS)

##########################################
# Generic rules
##########################################

$(LIBDIR)/%.o: $(SRCDIR)/%%.cpp $(HEADDIR)/%.h
	$(CC)  -o $@ -c $< $(CFLAGS)

$(LIBDIR)/%.o: $(SRCDIR)/%.cpp
	$(CC)  -o $@ -c $< $(CFLAGS)

cleanDebug:
	rm -f $(LIBDIR)/*.o $(LIBDIR)/*~ $(SRCDIR)/*~
clean:
	rm -f $(LIBDIR)/*.o $(LIBDIR)/*~ $(SRCDIR)/*~

// File: graph.h
// -- simple graph handling header file
//-----------------------------------------------------------------------------
// Community detection
// Based on the article "Fast unfolding of community hierarchies in large networks"
// Copyright (C) 2008 V. Blondel, J.-L. Guillaume, R. Lambiotte, E. Lefebvre
//
// This program must not be distributed without agreement of the above mentionned authors.
//-----------------------------------------------------------------------------
// Author   : E. Lefebvre, adapted by J.-L. Guillaume and then Anthony Perez and Nicolas Dugué for directed modularity
//-----------------------------------------------------------------------------
// see readme.txt for more details

#ifndef GRAPH_H
#define GRAPH_H

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <limits>
#define WEIGHTED   0
#define UNWEIGHTED 1

using namespace std;

class Graph {
public:
vector<vector<pair<unsigned int,float> > > links_out;
vector<vector<pair<unsigned int,float> > > links_in;
vector<unsigned long long int> correspondance;

Graph (char* in_filename, char *filename, char* filename_w, int type, bool do_renumber, unsigned int nodes);

unsigned long long int maj_corresp(unsigned int dest, unsigned long long int cpt);

void clean(int type);
void display(int type);
void display_binary(char *filename, char *filename_w, int type, bool do_renumber);
};

#endif // GRAPH_H

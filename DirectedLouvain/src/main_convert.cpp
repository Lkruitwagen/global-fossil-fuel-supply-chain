// File: main_convert.cpp
// -- conversion of a graph from ascii to binary, sample main file
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

#include "../include/graph.h"
#include <string>

//using namespace std;

char *infile   = NULL;
char *outfile  = NULL;
char *outfile_w  = NULL;
int type       = UNWEIGHTED;
bool do_renumber = false;
unsigned int nodes=100000;

void
usage(char *prog_name, const char *more) {
        cerr << more;
        cerr << "usage: " << prog_name << " -i input_file -o outfile [-r] [-w outfile_weight][-n nodes_number]" << endl << endl;
        cerr << "read the graph and convert it to binary format." << endl;
        cerr << "-r\tnodes are renumbered from 0 to nb_nodes-1 (the order is kept)." << endl;
        cerr << "-n\tto roughly specify how much nodes contain your graph (improves memory usage)" << endl;
        cerr << "-w filename\tread the graph as a weighted one and writes the weights in a separate file." << endl;
        cerr << "-h\tshow this usage message." << endl;
        exit(0);
}

void
parse_args(int argc, char **argv) {
        for (int i = 1; i < argc; i++) {
                if(argv[i][0] == '-') {
                        switch(argv[i][1]) {
                        case 'i':
                                if (i==argc-1)
                                        usage(argv[0], "Infile missing\n");
                                infile = argv[i+1];
                                i++;
                                break;
                        case 'o':
                                if (i==argc-1)
                                        usage(argv[0], "Outfile missing\n");
                                outfile = argv[i+1];
                                i++;
                                break;
                        case 'w':
                                if (i==argc-1)
                                      usage(argv[0], "Weight file missing\n");
                                type = WEIGHTED;
                                outfile_w = argv[i+1];
                                i++;
                                break;
                        case 'r':
                                do_renumber=true;
                                break;
                        case 'n':
                                if (i==argc-1)
                                      usage(argv[0], "Number of nodes missing\n");
                                nodes = std::stoi(argv[i+1]);
                                i++;
                                break;
                        default:
                                usage(argv[0], "Unknown option\n");
                        }
                } else {
                        usage(argv[0], "More than one filename\n");
                }
        }
        if (infile==NULL || outfile==NULL)
                usage(argv[0], "In or outfile missing\n");
}

int
main(int argc, char **argv) {
        parse_args(argc, argv);
        Graph* g;

        g = new Graph(infile, outfile, outfile_w, type, do_renumber, nodes);


}

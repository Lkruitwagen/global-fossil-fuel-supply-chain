// File: graph_binary.cpp
// -- graph handling source
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

#include <sys/mman.h>
#include <fstream>
#include <sstream>
#include <string>
#include <string.h>
#include "../include/graph_binary.h"
#include "math.h"

Graph::Graph() {
        nb_nodes     = 0;
        nb_links_out     = 0;
        nb_links_in     = 0;
        total_weight = 0;
}

Graph::Graph(char *filename, char *filename_w, int type, bool renumbered) {
        ifstream finput;
        finput.open(filename,fstream::in | fstream::binary);

        cerr << "number of nodes" << endl;
        // Read number of nodes on 4 bytes
        finput.read((char *)&nb_nodes, sizeof(int));
        assert(finput.rdstate() == ios::goodbit);
        cerr << "done: " << nb_nodes << endl;

        // Read cumulative out degree sequence: 8 bytes for each node
        // cum_degree[0]=degree(0); cum_degree[1]=degree(0)+degree(1), etc.
        cerr << "degrees out" << endl;
        degrees_out.resize(nb_nodes);
        finput.read((char *)&degrees_out[0], nb_nodes*sizeof(long));
        cerr << "done : " << degrees_out[nb_nodes - 1] << endl;

// Read links_out: 4 bytes for each link
//cerr << "links_out" << endl;
        nb_links_out=degrees_out[nb_nodes-1];
        links.resize(nb_links_out);
        finput.read((char *)(&links[0]), (long)nb_links_out*sizeof(unsigned int));
//	cout << "done" << endl;

        // Read cumulative in degree sequence: 8 bytes for each node
        // cum_degree[0]=degree(0); cum_degree[1]=degree(0)+degree(1), etc.
        cerr << "degrees in" << endl;
        degrees_in.resize(nb_nodes);
        finput.read((char *)&degrees_in[0], nb_nodes*sizeof(long));
        cerr << "done : " << degrees_in[nb_nodes - 1] << endl;

        // Read links_in: 4 bytes for each link
        //cerr << "links in" << endl;
        nb_links_in=degrees_in[nb_nodes-1];
        links_in.resize(nb_links_in);
        finput.read((char *)(&links_in[0]), (long)nb_links_in*sizeof(unsigned int));
        //cout << "done" << endl;

        // Read correspondance of labels
        if(renumbered) {
                cerr << "correspondance" << endl;
                correspondance.resize(nb_nodes);
                finput.read((char *)(&correspondance[0]), nb_nodes*sizeof(unsigned long long int));
        }
        //cout << "done" << endl;

        // IF WEIGHTED : read weights: 4 bytes for each link (each link is counted twice)
        weights.resize(0);
        weights_in.resize(0);

        total_weight=0;
        if (type==WEIGHTED) {
                cerr << "Weights reading" << endl;
                ifstream finput_w;
                finput_w.open(filename_w,fstream::in | fstream::binary);
                weights.resize(nb_links_out);
                finput_w.read((char *)&weights[0], nb_links_out*sizeof(double));
                weights_in.resize(nb_links_in);
                finput_w.read((char *)&weights_in[0], nb_links_in*sizeof(double));
                cerr << "Done" << endl;
        }

        // Compute total weight
        for (unsigned int i=0; i<nb_nodes; i++) {
                total_weight += out_weighted_degree(i);
        }
}

Graph::Graph(int n, int m, float t, int *d, int *l, float *w) {
/*  nb_nodes     = n;
   nb_links     = m;
   total_weight = t;
   degrees_out      = d;
   links        = l;
   weights      = w;*/
}

void
Graph::display() {
/*  for (unsigned int node=0 ; node<nb_nodes ; node++) {
    pair<vector<unsigned int>::iterator, vector<double>::iterator > p = neighbors(node);
    for (unsigned int i=0 ; i<nb_neighbors_out(node) ; i++) {
      if (node<=*(p.first+i)) {
   if (weights.size()!=0)
    cout << node << " " << *(p.first+i) << " " << *(p.second+i) << endl;
   else
    cout << node << " " << *(p.first+i) << endl;
      }
    }
   }*/
        for (unsigned int node=0; node<nb_nodes; node++) {
                pair<vector<unsigned int>::iterator, vector<double>::iterator > p = neighbors(node);
                cout << node << ":";
                for (unsigned int i=0; i<nb_neighbors_out(node); i++) {
                        if (true) {
                                if (weights.size()!=0)
                                        cout << " (" << *(p.first+i) << " " << *(p.second+i) << ")";
                                else
                                        cout << " " << *(p.first+i);
                        }
                }
                cout << endl;
        }
}

/* Methode de reecriture du fichier */
void
Graph::writeFile(string outNeighbors, string inNeighbors) {

        ofstream foutput;
        foutput.open(outNeighbors.c_str(),fstream::out | fstream::binary);

        cout << "Nombre de noeuds : " << nb_nodes << endl;

        /* On recupere les voisins sortants */
        for(unsigned int node=0; node < nb_nodes; node++) {

                pair<vector<unsigned int>::iterator, vector<double>::iterator > p = neighbors(node);

                for(unsigned int i = 0; i < nb_neighbors_out(node); i++) {

                        foutput << correspondance[node] << " " << correspondance[*(p.first+i)] << endl;

                }

        }

        foutput.close();

        ofstream foutputIn;
        foutputIn.open(inNeighbors.c_str(), fstream::out | fstream::binary);

        /* On recupere les voisins entrants */
        for(unsigned int node=0; node < nb_nodes; node++) {

                pair<vector<unsigned int>::iterator, vector<double>::iterator > p1 = in_neighbors(node);

                for(unsigned int i = 0; i < nb_neighbors_in(node); i++) {

                        foutputIn << correspondance[node] << " " << correspondance[*(p1.first+i)] << endl;

                }

        }

}

void
Graph::computeOverlap(string fileName) {

        ofstream foutput;
        foutput.open(fileName.c_str(), fstream::out | fstream::binary);
        foutput.precision(15);

        unsigned int deg;
        float overlap = 0;
        double ratio = 0.000000;
        unsigned int in, out;
        for(unsigned int node = 0; node < nb_nodes; node++) {

                if(node == 0) deg = 0; else deg = node - 1;

                unsigned int* outNeighbors;
                unsigned int* inNeighbors;
                if(node == 0) {
                        outNeighbors = &links[0];
                        inNeighbors = &links_in[0];
                } else {
                        outNeighbors = &links[degrees_out[deg]];
                        inNeighbors = &links_in[degrees_in[deg]];

                }
                vector<unsigned long long int> inter(min(nb_neighbors_out(node), nb_neighbors_in(node)));
                vector<unsigned long long int>::iterator it;

                sort(outNeighbors, outNeighbors + nb_neighbors_out(node));
                sort(inNeighbors, inNeighbors + nb_neighbors_in(node));

                out = nb_neighbors_out(node);
                in = nb_neighbors_in(node);

                if(out > 0 && in > 0) {
                        it = set_intersection(outNeighbors, outNeighbors + out, inNeighbors, inNeighbors + in, inter.begin());
                        inter.resize(it-inter.begin());

                        ratio = (float)out / in;
                        overlap = max(inter.size()/(float)out, inter.size()/(float)in);

                }
                else {
                        ratio = 0;
                        overlap = 0;
                }
                // 17-02-2014 - ANTO : ajout de la taille de l'intersection dans le fichier de sortie
                foutput << correspondance[node] << ";" << out << ";" << in << ";" << ratio << ";" << inter.size() << ";" << overlap << endl;
                inter.clear();
                vector<unsigned long long int>().swap( inter );
        }

        foutput.close();

}

int Graph::nb_high_degree() {

        unsigned int cpt = 0;
        unsigned int degree = 0;
        unsigned int verif = 0;

        for(unsigned int node = 0; node < nb_nodes; node++) {
                degree = nb_neighbors_in(node);
                verif += degree;

                if (degree >= 10000)

                        cpt++;

        }

        cout << verif << endl;
        return cpt;

}

void
Graph::display_reverse() {
        ofstream foutput;
        foutput.open("GRAPHE-REVERSE", ios::out);
        for (unsigned int node=0; node<nb_nodes; node++) {
                pair<vector<unsigned int>::iterator, vector<double>::iterator > p = neighbors(node);
                for (unsigned int i=0; i<nb_neighbors_out(node); i++) {
                        //if (node>*(p.first+i)) {
                        if (weights.size()!=0)
                                foutput << *(p.first+i) << "\t" << node << "\t" << *(p.second+i) << endl;
                        else
                                foutput << correspondance[node] << "\t" << correspondance[*(p.first+i)] << endl;
                        //}
                }
        }
        foutput.close();
}


bool
Graph::check_symmetry() {
        int error=0;
        for (unsigned int node=0; node<nb_nodes; node++) {
                pair<vector<unsigned int>::iterator, vector<double>::iterator > p = neighbors(node);
                for (unsigned int i=0; i<nb_neighbors_out(node); i++) {
                        unsigned int neigh = *(p.first+i);
                        double weight = *(p.second+i);

                        pair<vector<unsigned int>::iterator, vector<double>::iterator > p_neigh = neighbors(neigh);
                        for (unsigned int j=0; j<nb_neighbors_out(neigh); j++) {
                                unsigned int neigh_neigh = *(p_neigh.first+j);
                                double neigh_weight = *(p_neigh.second+j);

                                if (node==neigh_neigh && weight!=neigh_weight) {
                                        cout << node << " " << neigh << " " << weight << " " << neigh_weight << endl;
                                        if (error++==10)
                                                exit(0);
                                }
                        }
                }
        }
        return (error==0);
}


void
Graph::display_binary(char *outfile) {
        ofstream foutput;
        foutput.open(outfile,fstream::out | fstream::binary);

        foutput.write((char *)(&nb_nodes),4);
        foutput.write((char *)(&degrees_out[0]),4*nb_nodes);
        foutput.write((char *)(&links[0]),8*nb_links_out);
        foutput.write((char *)(&degrees_in[0]),4*nb_nodes);
        foutput.write((char *)(&links_in[0]),8*nb_links_in);
}

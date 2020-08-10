#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <map>

#include "../include/graph_binary.h"
#include "../include/graph.h"

/* Convertit un fichier a l'envers, puis compare au fichier existant */

int main(int argc, char** argv) {

	/* Extraction du fichier normal depuis le fichier binaire */
	Graph gHighDegree = Graph(argv[1], NULL, UNWEIGHTED, true);
	cout << "nb high degree : " << gHighDegree.nb_high_degree() << endl;
}

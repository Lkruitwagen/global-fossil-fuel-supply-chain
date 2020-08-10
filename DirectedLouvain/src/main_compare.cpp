#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <map>

#include "../include/graph_binary.h"

/* Convertit un fichier a l'envers, puis compare au fichier existant */

int main(int argc, char** argv) {

	/* Extraction du fichier normal depuis le fichier binaire */
	Graph gInverse = Graph(argv[1], NULL, UNWEIGHTED, true);
	gInverse.writeFile("graphes/outNeighbors.txt", "graphes/inNeighbors.txt");

}

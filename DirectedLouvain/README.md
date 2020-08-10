# DirectedLouvain

The algorithm used in this package is the same that was developped by V. Blondel, J.-L. Guillaume, R. Lambiotte, E. Lefebvre and was downloaded based on the [Louvain algorithm webpage] (https://sites.google.com/site/findcommunities/) (**[2]**).
The algorithm was then adjusted to optimize the directed modularity of Leich and Newman instead of the classic modularity.
These modifications were mostly made by [Anthony perez] (http://www.univ-orleans.fr/lifo/membres/Anthony.Perez), and a few by [Nicolas Dugué] (http://www.univ-orleans.fr/lifo/membres/Nicolas.Dugue/).

The directed modularity is proved to be more efficient in the case of directed graphs as shown in [Directed Louvain : maximizing modularity in directed networks] (https://hal.archives-ouvertes.fr/hal-01231784) (**[3]**) and was also succesfully used in [A community role approach to assess social capitalists visibility in the Twitter network] (https://hal.archives-ouvertes.fr/hal-01163741) with Vincent Labatut and Anthony Perez (**[1]**).

**The README below is from the Louvain algorithm : our package works in a similar way**

-----------------------------------------------------------------------------



**Convert**

This package offers a set of functions to use in order to compute 
communities on graphs weighted or unweighted. A typical sequence of 
actions is:

Conversion from a text format (each line contains a couple "src dest")

    ./convert -i graph.txt -o graph.bin
  
This program can also be used to convert weighted graphs (each line contain
a triple "src dest w") using -w option:

    ./convert -i graph.txt -o graph.bin -w graph.weights
  
Finally, nodes can be renumbered from 0 to nb_nodes - 1 using -r option
(less space wasted in some cases):

    ./convert -i graph.txt -o graph.bin -r

-----------------------------------------------------------------------------
**Compute communities**

Computes communities and displays hierarchical tree:

    ./community graph.bin -l -1 -v > graph.tree

To ensure a faster computation (with a loss of quality), one can use
the -q option to specify that the program must stop if the increase of
modularity is below epsilon for a given iteration or pass:

    ./community graph.bin -l -1 -q 0.0001 > graph.tree

The program can deal with weighted networks using -w option:

    ./community graph.bin -l -1 -w graph.weights > graph.tree
    
In this specific case, the convertion step must also use the -w option.

The program can also start with any given partition using -p option

    ./community graph.bin -p graph.part -v
  
-----------------------------------------------------------------------------
**Display communities information**

Displays information on the tree structure (number of hierarchical
levels and nodes per level):

    ./hierarchy graph.tree

Displays the belonging of nodes to communities for a given level of
the tree:

    ./hierarchy graph.tree -l 2 > graph_node2comm_level2

-----------------------------------------------------------------------------

Known bugs or restrictions:
- the number of nodes is stored on 4 bytes and the number of links on 8 bytes.

-----------------------------------------------------------------------------

## References
* **[1]** Nicolas Dugué, Vincent Labatut, Anthony Perez. A community role approach to assess social capitalists visibility in the Twitter network. Social Network Analysis and Mining, Springer, 2015, 5, pp.26.
* **[2]** BLONDEL, Vincent D., GUILLAUME, Jean-Loup, LAMBIOTTE, Renaud, et al. Fast unfolding of communities in large networks. Journal of Statistical Mechanics: Theory and Experiment, 2008, vol. 2008, no 10, p. P10008.
* **[3]** Nicolas Dugué, Anthony Perez. Directed Louvain : maximizing modularity in directed networks. [Research Report] Université d'Orléans. 2015.

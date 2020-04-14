# Global Fossil Fuel Supply Chain

A repo for building a network implementation of the global fossil fuel supply chain using asset level data. A project in support of the [Oxford Martin Programme on the Post-Carbon Transition](https://www.oxfordmartin.ox.ac.uk/post-carbon/).

![Global fossil fuel infrastructure](image_assets/global_infrastructure.png)

## Research Applications
The energy system is inherently spatial. Spatially localised energy demand is increasingly met with intermittent, spatially localised energy supply. Energy infrastructure is path-depends and creates technology- and greenhouse-gas emissions lock-in. Understanding the spatial layout of conventional energy infrastructure, and geographies of resource supply and demand is critical for designing efficient and robust pathways to net-zero carbon emissions.

This repo takes raw data and constructs it into a graph which can be interrogated with network analysis. Research questions will then follow:
- How might we identify geospatial sensitive intervention points in the fossil fuel supply chain?
- What is the most efficient way to decarbonise: extremities first (savings in the parasitic emission of infrastructure), or central core first (i.e. strand the extremities)?
- Can we find robust pathways to a carbon-free energy system? What about strategies robust to the interdiction of an industry that continues to build?
- Is the data and model prescient? Can we fit it to historic emissions or pricing data?

## Repo Structure

/globalenergydemand contains scripts for calculating and allocating global energy demand:
- [degreedays.py](globalenergydemand/degreedays.py) - contains a class for calculating heating and cooling degree day baselines for anywhere in the world
- [euclideanallocation.py](globalenergydemand/euclideanallocation.py) - contains a class for the spatial allocation of any country's land mass to urban population centers in that country, creating representative population clusters for any country.
- [energyallocation.py](globalenergydemand/energyallocation.py) - contains a class for energy allocation (coal, oil, and natural gas) to any population clusters created in the euclidean allocaiton
- [vincenty_py3](globalenergydemand/vincenty_py.py) - some simple vincenty and vincenty_inverse functions for forward and backwards vincenty calculations.

## Setup

### Environment

We use [conda](https://docs.conda.io/en/latest/miniconda.html) for environment management. Create a new environment:

    conda create -n ffsc python=3.7

Activate your conda environment:

    conda activate ffsc

Clone and change directory into this repo:

    git clone https://github.com/Lkruitwagen/global-fossil-fuel-supply-chain.git
    cd global-fossil-fuel-supply-chain

Install pip package manager to the environment if it isn't already:

    conda install pip

Install the project packages:

    pip install -r requirements.txt

We need the bleeding edge branch of GeoPandas (until 0.8 is released) which significantly speeds up spatial joins:

    pip install https://github.com/geopandas/geopandas/archive/master.zip

On a fresh linux install you will also require the following:

    sudo apt-get update
    sudo apt-get install python3-dev build-essential libspatialindex-dev openjdk-8-jre


### Database - Neo4J
1. Note if ssh tunnelling you will need to redirect port 8678 and 8888
1. [Install neo4j server](https://neo4j.com/docs/operations-manual/current/installation/linux/). We use Neo4j 3.5 in our experiments, 
but everything described here should also work with Neo4j 4.0

### Environment Variables

Save the environment variables we need in activation and deactivation scripts in the conda environment. Follow the [conda instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#setting-environment-variables) for your os, and adapt the following:

    cd $CONDA_PREFIX
    mkdir -p ./etc/conda/activate.d
    mkdir -p ./etc/conda/deactivate.d
    touch ./etc/conda/activate.d/env_vars.sh
    touch ./etc/conda/deactivate.d/env_vars.sh

edit `./etc/conda/activate.d/env_vars.sh` as follows:

    #!/bin/sh
    export USE_PYGEOS=1
    export PYTHONPATH=~/path/to/repo/global-fossil-fuel-supply-chain

Leave `<YOURPASSWORD>` blank if you haven't password protected your database (default).

edit `./etc/conda/deactivate.d/env_vars.sh` as follows:

    #!/bin/sh

    unset PYTHONPATH
    unset USE_PYGEOS

Save and close both files.


## Useage

### Data

Data must be [downloaded](https://drive.google.com/file/d/1LWXT3WyNpMS8xmdFzStbUyQlzdPLGhv_/view?usp=sharing) and unzipped in a folder `data` in the main directory.

### Running the pipeline
1. Execute the pipeline using `kedro run`. For more information on how to run the pipeline, see the [Kedro Docs](https://kedro.readthedocs.io/en/stable/).

As some of the data files are quite big, we recommend having around 50GB of RAM available. The pipeline will write a set of 
CSV files to the `results/output` folder. These files are pre-formatted to be used with the Neo4j importer.

### Importing files into Neo4j
1. Make sure Neo4j is shut down. The installer might start up neo4j under a different user (e.g. Neo4j). 
In this case, you might want to find the process under which Neo4j runs using `sudo ps -a | grep neo4j`. Find the PID of the process and kill it using `sudo kill`.
2. As Neo4j's files may be restricted, you want to do the next steps as root.
3. Delete Neo4j's data folder from the old database. On Linux, this is stored under `/var/lib/neo4j/data`.
4. Import the data by executing the `import.sh` script, which you can find under `src/neo4j_commands` in this repository.
5. After the import is complete, restart Neo4j using `neo4j start`

After import, you should see a message like this. The raw graph has about 11 million nodes and 20.6 million relationships.
![](image_assets/import_complete.png)

### Simplifying the graph
The graph construction pipeline treats each segment edge as a separate node. 
This creates a lot of pipeline and railway nodes which are not needed for most analysis. To simplify the graph, we have provided a set of simplification queries.
Once you have imported the data into Neo4j and launched the database, run the Cypher queries in `src/neo4j_commands/graph_simplification.cypher` one by one. 
The graph simplification will create a direct relationship between railway and pipeline nodes which are otherwise connected through a string of segments. 
Afterwards, the individual nodes and relationships making up the segments can be deleted.
![](image_assets/simplified_edge.png)



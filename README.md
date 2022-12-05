# Fall 2022 CS170 Final Project
This is the repository for the Fall 2022 CS170 final project for the team TheKarpOfEdmond.

The goal is to partition a graph into disjoint sets in order to minimize a cost function that involves the weights between vertices in the same set, the total number of sets, and the relative balance of vertices in each set. If you want to learn more about this project, you can look at the spec here: [Project Spec](https://cs170.org/assets/pdf/project.pdf).

## How to Generate Outputs
Before proceeding, ensure that you have downloaded the inputs from this link: [Inputs](https://cs170.org/assets/misc/inputs.zip). Make sure that you unzip the `assets_misc_inputs.zip` file and place the resulting `inputs` directory on the same level as `main.py`.

First, in `main.py` choose the initial solution heuristic and local search algorithm by changing the global variables at the top of the file.

You can choose one of the following options for the initial solution heuristic:
* SPECTRAL: Uses a spectral clustering algorithm from scikit-learn to partition the graph into k disjoint sets
* GREEDY: Uses a greedy algorithm to union sets together until a locally optimum score is acheived.
* RANDOM: Randomly assigns every vertex to a set numbered between 1 and k (inclusive)

You can choose one of the following options for the local search algorithm:
* FM: The Fidducia-Mattheyses Algorithm
* ANNEALING: A simulated annealing approach

After selecting the relevant values, run the `main.py` file using a Python interpreter.

If the GREEDY heuristic is selected, the outputs will be generated in a folder called `outputs1`. This folder will then be compressed into a file called `outputs1.tar`.

If the SPECTRAL or RANDOM heuristics have been selected, outputs will be generated in folder called `outputs2`, `outputs3`, ..., `outputs16`, and the compressed into files `outputs2.tar`, `outputs3.tar`, ..., `outputs16.tar`. Each file `outputsk.tar` contains the found partition of every input graph into `k` disjoint sets, with `k` ranging between 2 and 16.

## Requirements
`python >= 3.6` is needed for `starter.py`, which contains a handful of utility functions, to run properly; it was, nevertheless, developed in python 3.9.  
The other dependencies needed for this project are `networkx`, `numpy`, `matplotlib`, `tqdm`, and `scikit-learn`. These dependencies can all be installed through pip.

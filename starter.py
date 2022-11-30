import networkx as nx
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from pathlib import Path
from tqdm.auto import tqdm
import tarfile
from collections import defaultdict

# Scoring constants
MAX_WEIGHT = 1000
MAX_EDGES = 10000
N_SMALL = 100
N_MEDIUM = 300
N_LARGE = 1000
K_EXP = 0.5
K_COEFFICIENT = 100
B_EXP = 70

INPUT_SIZE_LIMIT = 1000000
OUTPUT_SIZE_LIMIT = 10000


def write_input(G: nx.Graph, path: str, overwrite: bool=False):
    assert overwrite or not os.path.exists(path), \
        'File already exists and overwrite set to False. Move file or set overwrite to True to proceed.'
    if validate_input(G):
        with open(path, 'w') as fp:
            json.dump(nx.node_link_data(G), fp)


def read_input(path: str):
    assert os.path.getsize(path) < INPUT_SIZE_LIMIT, 'This input file is too large'
    with open(path) as fp:
        G = nx.node_link_graph(json.load(fp), multigraph=False)
        if validate_input(G):
            return G


def write_output(G: nx.Graph, path: str, overwrite=False):
    assert overwrite or not os.path.exists(path), \
        'File already exists and overwrite set to False. Move file or set overwrite to True to proceed.'
    if validate_output(G):
        with open(path, 'w') as fp:
            json.dump([G.nodes[v]['team'] for v in range(G.number_of_nodes())], fp)


def read_output(G: nx.Graph, path: str):
    assert os.path.getsize(path) < OUTPUT_SIZE_LIMIT, 'This output file is too large'
    with open(path) as fp:
        l = json.load(fp)
        assert isinstance(l, list), 'Output partition must be a list'
        assert set(G) == set(range(len(l))), 'Output does not match input graph'
        nx.set_node_attributes(G, {v: l[v] for v in G}, 'team')
        if validate_output(G):
            return G


def validate_graph(G: nx.Graph):
    assert not G.is_directed(), 'G should not be directed'
    assert set(G) == set(range(G.number_of_nodes())), 'Nodes must be numbered from 0 to n-1'
    return True


def validate_input(G: nx.Graph):
    for n, d in G.nodes(data=True):
        assert not d, 'Nodes cannot have data'
    for u, v, d in G.edges(data=True):
        assert u != v, 'Edges should be between distinct vertices (a penguin is experiencing inner-conflict)'
        assert set(d) == {'weight'}, 'Edge must only have weight data'
        assert isinstance(d['weight'], int), 'Edge weights must be integers'
        assert d['weight'] > 0, 'Edge weights must be positive'
        assert d['weight'] <= MAX_WEIGHT, f'Edge weights cannot be greater than {MAX_WEIGHT}'
    assert G.number_of_edges() <= MAX_EDGES, 'Graph has too many edges'
    assert sum(d for u, w, d in G.edges(data='weight')) >= MAX_WEIGHT*MAX_EDGES*0.05, \
        f'There must be at least {MAX_WEIGHT*MAX_EDGES*0.05} edge weight in the input.'
    assert not G.is_multigraph()
    return validate_graph(G)


def validate_output(G: nx.Graph):
    for n, d in G.nodes(data=True):
        assert set(d) == {'team'}, 'Nodes must have team data'
        assert isinstance(d['team'], int), 'Team identifier must be an integer'
        assert d['team'] > 0, 'Team identifier must be greater than 0'
        assert d['team'] <= G.number_of_nodes(), 'Team identifier unreasonably large'
    return validate_graph(G)


def score(G: nx.Graph, separated=False):
    output = [G.nodes[v]['team'] for v in range(G.number_of_nodes())]
    teams, counts = np.unique(output, return_counts=True)

    k = np.max(teams)
    b = np.linalg.norm((counts / G.number_of_nodes()) - 1 / k, 2)
    C_w = sum(d for u, v, d in G.edges(data='weight') if output[u] == output[v])

    if separated:
        return C_w, K_COEFFICIENT * math.exp(K_EXP * k), math.exp(B_EXP * b)
    return C_w + K_COEFFICIENT * math.exp(K_EXP * k) + math.exp(B_EXP * b)
    
def visualize(G: nx.Graph):
    output = G.nodes(data='team', default=0)
    partition = dict()
    for n, t in output:
        if t not in partition:
            partition[t] = []
        partition[t].append(n)

    pos = dict()
    circle_size = len(partition) * 0.5
    for k, v in partition.items():
        pos.update(nx.shell_layout(G, nlist=[v], center=(circle_size*math.cos(math.tau*k / len(partition)),
                                                         circle_size*math.sin(math.tau*k / len(partition)))))

    crossing_edges = [e for e in G.edges(data='weight') if output[e[0]] != output[e[1]]]
    within_edges = [e for e in G.edges(data='weight') if output[e[0]] == output[e[1]]]
    max_weight = max(nx.get_edge_attributes(G, name='weight').values())

    nx.draw_networkx_nodes(G, pos, node_color=[output[n] for n in G],
                           cmap=cm.get_cmap('tab20b'))
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="white")

    nx.draw_networkx_edges(G, pos, edgelist=crossing_edges, edge_color=[x[2] for x in crossing_edges],
                           edge_cmap=cm.get_cmap('Blues'), edge_vmax=max_weight*1.5, edge_vmin=max_weight*-0.2)
    nx.draw_networkx_edges(G, pos, width=2, edgelist=within_edges, edge_color=[x[2] for x in within_edges],
                           edge_cmap=cm.get_cmap('Reds'), edge_vmax=max_weight*1.5, edge_vmin=max_weight*-0.2)

    plt.tight_layout()
    plt.axis("off")
    plt.show()


def run(solver, in_file: str, out_file: str, overwrite: bool=False):
    instance = read_input(in_file)
    output = solver(instance)
    if output:
        instance = output
    write_output(instance, out_file, overwrite)
    print(f"{str(in_file)}: cost", score(instance))


def run_all(solver, in_dir, out_dir, overwrite: bool=False):
    for file in tqdm([x for x in os.listdir(in_dir) if x.endswith('.in')]):
        run(solver, str(Path(in_dir) / file), str(Path(out_dir) / f"{file[:-len('.in')]}.out"), overwrite)


def tar(out_dir, overwrite=False):
    path = f'{os.path.basename(out_dir)}.tar'
    assert overwrite or not os.path.exists(path), \
        'File already exists and overwrite set to False. Move file or set overwrite to True to proceed.'
    with tarfile.open(path, 'w') as fp:
        fp.add(out_dir)

def get_b_and_b_norm(G: nx.graph):
    output = [G.nodes[v]['team'] for v in range(G.number_of_nodes())]
    teams, counts = np.unique(output, return_counts=True)

    k = np.max(teams)
    b = counts / G.number_of_nodes() - 1 / k
    b_norm = np.linalg.norm(b, 2)
    return b, b_norm
    
def cost(G: nx.graph, vertex: int, new_team: int, b: np.array = None, b_norm: int = None):
    if b is None or b_norm is None:
        b, b_norm = get_b_and_b_norm(G)
    old_team = G.nodes[vertex]["team"]
    b_i = b[old_team]
    b_j = b[new_team]
    V = G.number_of_nodes()
    new_Cp = math.exp(70 * (b_norm ** 2 - b_i ** 2 - b_j ** 2 + (b_i - 1 / V) ** 2 + (b_j + 1 / V) ** 2) ** (1/2))
    new_Cw = 0
    for neighbor in G.neighbors(vertex):
        if G.nodes[neighbor]["team"] != new_team:
            new_Cw += G[vertex][neighbor]["weight"]
    return new_Cp + new_Cw

def solve(G: nx.Graph):
    # Assign a team to v with G.nodes[v]['team'] = team_id
    # Access the team of v with team_id = G.nodes[v]['team']
    G_intermediate = G
    num_nodes = len(G.nodes)
    num_teams = 2
 
    # Partition first half to team 0
    for u in range(num_nodes//2):
        G.nodes[u]['team'] = 0
 
    # Partition first half to team 1
    for v in range(num_nodes//2, num_nodes):
        G.nodes[v]['team'] = 1
 
    unmarked = set(list(G.nodes))
    gains = []
    while len(unmarked) != 0:
        best_cost = float('inf')
        swap_pair = None
        for u in unmarked:
            for team in range(num_teams):
                cost_if_swapped = cost(G_intermediate, u, team)
                if cost_if_swapped < best_cost:
                    swap_pair = (u, team, cost_if_swapped)
                    best_cost = cost_if_swapped
        G_intermediate = swap(G_intermediate, swap_pair[0], swap_pair[1])
        unmarked.remove(swap_pair[0])
        gains.append(swap_pair)
 
    smallest_swap_score = min(gains, key=lambda x: x[2])
 
    for v, team, swap_score in gains:
        G = swap(G, v, team)
        if swap_score == smallest_swap_score:
            break
    return G
 
def swap(G: nx.graph, v: int, team: int):
    G.nodes[v]["team"] = team
    return G
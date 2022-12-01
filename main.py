from starter import *
from sklearn.cluster import spectral_clustering
K_VALUE = 4

def main():
    G = read_input('inputs\\large1.in')
    G_naive = read_input('inputs\\large1.in')
    solve(G)
    validate_output(G)
    calculated_score = score(G)
    # visualize(G)
    #solve_naive(G_naive)
    spectralSolve(G_naive, K_VALUE)
    validate_output(G_naive)
    score_naive = score(G_naive)
    print("score: ", calculated_score)
    print("naive score: ", score_naive)
    print("better by ", score_naive - calculated_score)

def cost(G: nx.graph, vertex: int, new_team: int, weight_score: int = None, teams_score: int = None, balance_score: int = None,  b: np.array = None, b_norm: int = None, ):
    if b is None or b_norm is None:
        b, b_norm = get_b_and_b_norm(G)
    if weight_score is None or teams_score is None or balance_score is None:
        weight_score, teams_score, balance_score = score(G, separated=True)
    old_team = G.nodes[vertex]["team"]
    b_i = b[old_team - 1]
    b_j = b[new_team - 1]
    V = G.number_of_nodes()
    
    b[old_team - 1] -= 1 / V
    b[new_team - 1] += 1 / V

    new_weight_score = weight_score
    if old_team == new_team:
        new_balance_score = balance_score
    else:
        b_norm = np.sqrt(b_norm ** 2 - b_i ** 2 - b_j ** 2 + (b_i - 1 / V) ** 2 + (b_j + 1 / V) ** 2)
        new_balance_score = math.exp(70 * b_norm)
        for neighbor in G.neighbors(vertex):
            if G.nodes[neighbor]["team"] == new_team:
                new_weight_score += G[vertex][neighbor]["weight"]
            if G.nodes[neighbor]["team"] == old_team:
                new_weight_score -= G[vertex][neighbor]["weight"]
    return new_weight_score, teams_score, new_balance_score, b, b_norm

def solve(G: nx.Graph):
    #solve_naive(G) # Will be replaced with approximation
    spectralSolve(G, K_VALUE)
    local_search(G)

def local_search(G: nx.graph):
    teams = list(map(int, get_teams_and_counts(G)[0]))
    i = 0
    curr_weight_score, curr_teams_score, curr_balance_score = score(G, separated=True)
    curr_b, curr_b_norm = get_b_and_b_norm(G)
    while True:
        old_score = curr_weight_score + curr_teams_score + curr_balance_score
        print(f"{i=}, {old_score=}")
        unmarked = set(list(G.nodes))
        size = G.number_of_nodes()
        while len(unmarked) != 0:
            print(f"{size - len(unmarked)}/{size}")
            best_cost = float('inf')
            swap_pair = None
            for u in unmarked:
                for team in teams:
                    weight_score, teams_score, balance_score, b, b_norm = cost(G, u, team, curr_weight_score, curr_teams_score, curr_balance_score, curr_b, curr_b_norm)
                    cost_if_swapped = weight_score + teams_score + balance_score
                    if cost_if_swapped < best_cost:
                        swap_pair = (u, team, weight_score, teams_score, balance_score, b, b_norm)
                        best_cost = cost_if_swapped
            u, team, curr_weight_score, curr_teams_score, curr_balance_score, curr_b, curr_b_norm = swap_pair
            swap(G, u, team)
            unmarked.remove(u)
        if curr_weight_score + curr_teams_score + curr_balance_score == old_score:
            break
        i += 1
    return G

def solve_naive(G: nx.graph):
    num_nodes = len(G.nodes)

    # Partition first half to team 1
    for u in range(num_nodes//2):
        G.nodes[u]['team'] = 1

    # Partition first half to team 2
    for v in range(num_nodes//2, num_nodes):
        G.nodes[v]['team'] = 2

def randomSolve(G: nx.Graph, k: int):
    for u in G.nodes:
        team = np.random.randint(1, k)
        G.nodes[u]["team"] = team
    
def spectralSolve(G: nx.Graph, k: int):
    adjMatrix = nx.to_numpy_array(G)
    beta = 10
    eps = 1e-6
    adjMatrix = np.exp(-beta * adjMatrix / adjMatrix.std()) + eps
    teams = spectral_clustering(adjMatrix, n_clusters=k, eigen_solver="arpack", eigen_tol=1e-7, assign_labels="discretize")
    for u in G.nodes:
        G.nodes[u]["team"] = int(teams[u]) + 1
    
def get_teams_and_counts(G: nx.graph):
    output = [G.nodes[v]['team'] for v in range(G.number_of_nodes())]
    k = np.max(output)
    teams = list(range(1, k+1))
    counts = [0] * k
    for team in output:
        counts[team - 1] += 1
    return np.array(teams), np.array(counts)

def get_b_and_b_norm(G: nx.graph):
    teams, counts = get_teams_and_counts(G)

    k = np.max(teams)
    b = (counts / G.number_of_nodes()) - 1 / k
    b_norm = np.linalg.norm(b, 2)
    return b, b_norm
 
def swap(G: nx.graph, v: int, team: int):
    G.nodes[v]["team"] = team
    return G

if __name__ == '__main__':
    main()
from starter import *
from sklearn.cluster import spectral_clustering
import multiprocessing as mp
import os

def main():
    k_values = range(11, 17)
    for k in k_values:
        if not os.path.exists(f"outputs{k}/"):
            os.makedirs(f"outputs{k}/")
        run_all_parallel(k)

def solve(k: int):
    def inner(G: nx.Graph):
        #greedySolve(G)
        spectralSolve(G, k)
        local_search(G)
    return inner

def run_parallel(pair):
    in_file, k = pair
    run(solve(k), "inputs/" + in_file, f"outputs{k}/" + in_file[:-len(".in")] + ".out", overwrite=False)

def run_all_parallel(k: int):
    input_files = [x for x in os.listdir("inputs") if x.endswith('.in')]
    inputs_with_k = tqdm([(f, k) for f in input_files])
    threads = mp.cpu_count()
    print("k:", k)
    print("Threads:", threads)
    with mp.Pool(threads - 3) as p:
        p.map(run_parallel, inputs_with_k)
    tar(f"outputs{k}")

def cost(G: nx.graph, vertex: int, new_team: int, weight_score: int = None, teams_score: int = None, balance_score: int = None,  b: np.array = None, b_norm: int = None, ):
    if b is None or b_norm is None:
        b, b_norm = get_b_and_b_norm(G)
    if weight_score is None or teams_score is None or balance_score is None:
        weight_score, teams_score, balance_score = score(G, separated=True)
    old_team = G.nodes[vertex]["team"]
    b_i = b[old_team - 1]
    b_j = b[new_team - 1]
    V = G.number_of_nodes()
    
    # b[old_team - 1] -= 1 / V
    # b[new_team - 1] += 1 / V

    new_weight_score = weight_score
    if old_team == new_team:
        new_balance_score = balance_score
    else:
        inside = b_norm ** 2 - b_i ** 2 - b_j ** 2 + (b_i - 1 / V) ** 2 + (b_j + 1 / V) ** 2
        if inside < 0:
            inside = 0
        b_norm = np.sqrt(inside)
        new_balance_score = math.exp(70 * b_norm)
        for neighbor in G.neighbors(vertex):
            if G.nodes[neighbor]["team"] == new_team:
                new_weight_score += G[vertex][neighbor]["weight"]
            if G.nodes[neighbor]["team"] == old_team:
                new_weight_score -= G[vertex][neighbor]["weight"]
        if np.random.random() < 0.001:
            swap(G, vertex, new_team)
            actual_weight, actual_teams, actual_balance = score(G, separated=True)
            # print("real:", actual_balance)
            # print("calculated:", new_balance_score)
            swap(G, vertex, old_team)
    return new_weight_score, teams_score, new_balance_score, b, b_norm

def local_search(G: nx.graph):
    teams = list(map(int, get_teams_and_counts(G)[0]))
    if len(teams) == 1:
        return G
    i = 0
    curr_weight_score, curr_teams_score, curr_balance_score = score(G, separated=True)
    old_score = curr_weight_score + curr_teams_score + curr_balance_score
    curr_b, curr_b_norm = get_b_and_b_norm(G)
    while True:
        G_intermediate = G.copy()
        # print(f"{i=}, {old_score=}")
        unmarked = set(list(G.nodes))
        size = G.number_of_nodes()
        swaps = []
        while len(unmarked) != 0:
            # print(f"{size - len(unmarked)}/{size}")
            best_cost = float('inf')
            swap_pair = None
            curr_b, curr_b_norm = get_b_and_b_norm(G_intermediate)
            for u in unmarked:
                old_team = G.nodes[u]["team"]
                for team in teams:
                    if team != old_team:
                        try:
                            weight_score, teams_score, balance_score, b, b_norm = cost(G_intermediate, u, team, curr_weight_score, curr_teams_score, curr_balance_score, curr_b, curr_b_norm)
                            cost_if_swapped = weight_score + teams_score + balance_score
                        except:
                            continue
                        if cost_if_swapped < best_cost:
                            swap_pair = (u, team)
                            best_cost = cost_if_swapped
            if best_cost == float("inf"):
                break
            u, team = swap_pair
            swap(G_intermediate, u, team)
            curr_weight_score, curr_teams_score, curr_balance_score = score(G_intermediate, separated=True)
            curr_b, curr_b_norm = get_b_and_b_norm(G_intermediate)
            swaps.append((u, team, curr_weight_score + curr_teams_score + curr_balance_score))
            unmarked.remove(u)
        # If every possible swap overflowed
        if swaps is None:
            return G
        lowest_of_swaps = min(swaps, key=lambda x: x[2])[2]
        if lowest_of_swaps >= old_score:
            break
        for u, team, score_of_swap in swaps:
            swap(G, u, team)
            if score_of_swap == lowest_of_swaps:
                old_score = score_of_swap
                break
        i += 1
    return G

def update_b(G: nx.graph, b: np.array, u: int, new_team: int):
    old_team = G.nodes[u]["team"]
    V = G.number_of_nodes()
    b[old_team - 1] -= 1 / V
    b[new_team - 1] += 1 / V

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

def greedySolve(G: nx.Graph):
    for u in G.nodes():
        swap(G, u, u)
    G2 = G.copy()
    A = nx.to_numpy_array(G).astype(float)
    np.fill_diagonal(A, float("inf"))
    numTeams = G.number_of_nodes()
    updates = []
    teamsToVertices = {i : {i} for i in range(numTeams)}
    Cw, Ck, Cb = score(G, separated=True)
    b, b_norm = get_b_and_b_norm(G)

    while numTeams > 1:
       i, j = np.unravel_index(np.argmin(A), A.shape)
       i, j = min(i, j), max(i, j)
       # print(f"{i=}, {j=}, {A[i, j]=}")
       # print(A)
       A[i, :] += A[j, :]
       A[:, i] += A[:, j]
       A[j, :] = float("inf")
       A[:, j] = float("inf")
       Cw, Ck, Cb, b, b_norm = mergeTeamsAndCost(G2, i, j, teamsToVertices, Cw, Ck, Cb, b, b_norm)
       # print(teamsToVertices)
       mergedScore = Cw + Ck + Cb
       updates.append((mergedScore, i, j))
       numTeams -= 1
    # print("numTeams:", numTeams)
    # print("updates:", updates)
    bestUpdate = np.argmin(np.array(updates)[:, 0])
    # print("k:", len(updates) - bestUpdate - 1)
    teamsToVertices = {i : {i} for i in range(G.number_of_nodes())}
    for index in range(bestUpdate + 1):
        _, i, j = updates[index]
        mergeTeams(G, i, j, teamsToVertices)

    teams = list(set(G.nodes[u]["team"] for u in G.nodes()))
    teams.sort()
    teamsMap = {team : index + 1 for index, team in enumerate(teams)}
    for u in G.nodes():
        team = G.nodes[u]["team"]
        swap(G, u, teamsMap[team])

    return G

def mergeTeams(G: nx.Graph, i: int, j: int, teamsToVertices):
    for u in teamsToVertices[j]:
        swap(G, u, i)
    teamsToVertices[i].update(teamsToVertices[j])
    teamsToVertices[j] = None

def mergeTeamsAndCost(G: nx.Graph, i: int, j: int, teamsToVertices, Cw: int, Ck: int, Cb: int, b: np.array, b_norm: float):
    """
    for u in teamsToVertices[j]:
        Cw, Ck, Cb, b, b_norm = cost(G, u, i, Cw, Ck, Cb, b, b_norm)
        swap(G, u, i)
    teamsToVertices[i].update(teamsToVertices[j])
    teamsToVertices[j] = None
    return Cw, Ck, Cb, b, b_norm
    """
    mergeTeams(G, i, j, teamsToVertices)
    Cw, Ck, Cb = score2(G, separated=True)
    return Cw, Ck, Cb, None, None
    

def get_teams_and_counts(G: nx.graph):
    output = [G.nodes[v]['team'] for v in range(G.number_of_nodes())]
    k = np.max(output)
    teams = list(range(1, k+1))
    counts = [0] * k
    for team in output:
        counts[team - 1] += 1
    return np.array(teams), np.array(counts)

def get_b_and_b_norm(G: nx.graph):
    output = [G.nodes[v]['team'] for v in range(G.number_of_nodes())]
    teams, counts = np.unique(output, return_counts=True)

    k = np.max(teams)
    b = (counts / G.number_of_nodes()) - 1 / k
    b_norm = np.linalg.norm(b, 2)
    return b, b_norm
 
def swap(G: nx.graph, v: int, team: int):
    G.nodes[v]["team"] = team
    return G

def score2(G: nx.Graph, separated=False):
    output = [G.nodes[v]['team'] for v in range(G.number_of_nodes())]
    teams, counts = np.unique(output, return_counts=True)

    k = len(teams)
    b = np.linalg.norm((counts / G.number_of_nodes()) - 1 / k, 2)
    C_w = sum(d for u, v, d in G.edges(data='weight') if output[u] == output[v])

    if separated:
        return C_w, K_COEFFICIENT * math.exp(K_EXP * k), math.exp(B_EXP * b)
    return C_w + K_COEFFICIENT * math.exp(K_EXP * k) + math.exp(B_EXP * b)

if __name__ == '__main__':
    main()
from starter import *
from sklearn.cluster import spectral_clustering
import multiprocessing as mp
import os

def main():
    G = read_input("inputs/small6.in")
    G2 = G.copy()
    solve(2)(G)
    spectralSolve(G2, 2)
    print("solve", score(G))
    print("spectral", score(G2))
    print("difference", score(G2) - score(G))

def run_parallel(pair):
    in_file, k = pair
    run(solve(k), "inputs/" + in_file, f"outputs{k}/" + in_file[:-len(".in")] + ".out", overwrite=False)

def run_all_parallel(k: int):
    input_files = [x for x in os.listdir("inputs") if x.endswith('.in')]
    inputs_with_k = tqdm([(f, k) for f in input_files])
    threads = mp.cpu_count()
    print("k:", k)
    print("Threads:", threads)
    with mp.Pool(threads - 1) as p:
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
    
    b[old_team - 1] -= 1 / V
    b[new_team - 1] += 1 / V

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

        swap(G, vertex, new_team)
        actual_weight, actual_teams, actual_balance = score(G, separated=True)
        actual_b, actual_b_norm = get_b_and_b_norm(G)
        assert new_weight_score == actual_weight, f"{new_weight_score=} != {actual_weight=}"
        assert np.abs(new_balance_score / actual_balance - 1) < 0.01, f"{new_balance_score=} != {actual_balance=}, {b=}"
        #assert b == actual_b, f"{b=} != {actual_b=}"
        #assert b_norm == actual_b_norm, f"{b_norm=} != {actual_b_norm=}"
        swap(G, vertex, old_team)
    return new_weight_score, teams_score, new_balance_score, b, b_norm

def solve(k: int):
    def inner(G: nx.Graph):
        spectralSolve(G, k)
        local_search(G)
    return inner

def local_search(G: nx.graph):
    teams = list(map(int, get_teams_and_counts(G)[0]))
    i = 0
    curr_weight_score, curr_teams_score, curr_balance_score = score(G, separated=True)
    curr_b, curr_b_norm = get_b_and_b_norm(G)
    old_score = curr_weight_score + curr_teams_score + curr_balance_score
    while True:
        print(f"{i=}, {old_score=}")
        print(score(G))
        unmarked = set(list(G.nodes))
        size = G.number_of_nodes()
        swaps = []
        stop = False
        while len(unmarked) != 0:
            print(f"{size - len(unmarked)}/{size}")
            best_cost = float('inf')
            swap_pair = None
            for u in unmarked:
                for team in teams:
                    if G.nodes[u]["team"] == team:
                        continue
                    try:
                        #weight_score, teams_score, balance_score, b, b_norm = cost(G, u, team, curr_weight_score, curr_teams_score, curr_balance_score, curr_b, curr_b_norm)
                        weight_score, teams_score, balance_score, b, b_norm = cost(G, u, team, curr_weight_score, curr_teams_score, curr_balance_score)
                    except OverflowError:
                        continue
                    cost_if_swapped = weight_score + teams_score + balance_score
                    if cost_if_swapped < best_cost:
                        swap_pair = (u, team, weight_score, teams_score, balance_score, b, b_norm)
                        best_cost = cost_if_swapped
            unmarked.remove(u)
            if swap_pair == None:
                continue
            u, team, curr_weight_score, curr_teams_score, curr_balance_score, curr_b, curr_b_norm = swap_pair
            old_team = G.nodes[u]["team"]
            swap(G, u, team)

            swaps.append((u, old_team, best_cost))
        print(swaps)
        best_score_overall = min(swaps, key=lambda x: x[2])[2]
        prev_best_overall = best_score_overall
        if best_score_overall >= old_score:
            stop = True
            best_score_overall = -1
        print("score before swapping bacK", score(G))
        for u, old_team, curr_cost in swaps[::-1]:
            if curr_cost == best_score_overall:
                old_score = curr_cost
                print("broke:", u, old_team, curr_cost)
                break
            print("swapped: ", u, old_team, curr_cost, best_score_overall)
            swap(G, u, old_team)
        print(best_score_overall)
        print(prev_best_overall)
        if stop or i == 0:
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
    teams, counts = get_teams_and_counts(G)

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
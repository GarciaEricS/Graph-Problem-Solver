from starter import *

def main():
    G = read_input('small1.in')
    G_naive = read_input('small1.in')
    solve(G)
    validate_output(G)
    calculated_score = score(G)
    # visualize(G)
    solve_naive(G_naive)
    validate_output(G_naive)
    score_naive = score(G_naive)
    print("score: ", calculated_score)
    print("naive score: ", score_naive)
    print("better by ", score_naive - calculated_score)

def cost(G: nx.graph, vertex: int, new_team: int, b: np.array = None, b_norm: int = None, weight_score: int = None, teams_score: int = None, balance_score: int = None):
    old_team = G.nodes[vertex]["team"]
    swap(G, vertex, new_team)
    new_cost = score(G)
    swap(G, vertex, old_team)
    return new_cost



    if b is None or b_norm is None:
        b, b_norm = get_b_and_b_norm(G)
    if weight_score is None or teams_score is None or balance_score is None:
        weight_score, teams_score, balance_score = score(G, separated=True)
    old_team = G.nodes[vertex]["team"]
    b_i = b[old_team - 1]
    b_j = b[new_team - 1]
    V = G.number_of_nodes()
    new_balance_score = math.exp(70 * (b_norm ** 2 - b_i ** 2 - b_j ** 2 + (b_i - 1 / V) ** 2 + (b_j + 1 / V) ** 2) ** (1/2))
    new_weight_score = weight_score
    for neighbor in G.neighbors(vertex):
        if G.nodes[neighbor]["team"] == new_team:
            new_weight_score -= G[vertex][neighbor]["weight"]
        if G.nodes[neighbor]["team"] == old_team:
            new_weight_score += G[vertex][neighbor]["weight"]
    return new_weight_score + teams_score + new_balance_score

def solve(G: nx.Graph):
    solve_naive(G) # Will be replaced with approximation
    local_search(G)

def local_search(G: nx.graph):
    G_intermediate = G
    teams = list(get_teams_and_counts(G)[0]) # This does not work
    teams = [1, 2]
    unmarked = set(list(G.nodes))
    gains = []
    while len(unmarked) != 0:
        best_cost = float('inf')
        swap_pair = None
        for u in unmarked:
            old_team = G.nodes[u]["team"]
            for team in teams:
                if team == old_team:
                    continue
                cost_if_swapped = cost(G_intermediate, u, team)
                if cost_if_swapped < best_cost:
                    swap_pair = (u, team, cost_if_swapped)
                    best_cost = cost_if_swapped
        swap(G_intermediate, swap_pair[0], swap_pair[1])
        unmarked.remove(swap_pair[0])
        gains.append(swap_pair)

    smallest_swap_score = min(gains, key=lambda x: x[2])[2]

    #print(gains)
    #print(smallest_swap_score)

    for v, team, swap_score in gains:
        swap(G, v, team)
        #print("swapped")
        if swap_score == smallest_swap_score:
            break
    return G

def solve_naive(G: nx.graph):
    num_nodes = len(G.nodes)

    # Partition first half to team 1
    for u in range(num_nodes//2):
        G.nodes[u]['team'] = 1

    # Partition first half to team 2
    for v in range(num_nodes//2, num_nodes):
        G.nodes[v]['team'] = 2
    
def get_teams_and_counts(G: nx.graph):
    output = [G.nodes[v]['team'] for v in range(G.number_of_nodes())]
    teams, counts = np.unique(output, return_counts=True)
    return teams, counts

def get_b_and_b_norm(G: nx.graph):
    teams, counts = get_teams_and_counts(G)

    k = np.max(teams)
    b = counts / G.number_of_nodes() - 1 / k
    b_norm = np.linalg.norm(b, 2)
    return b, b_norm
 
def swap(G: nx.graph, v: int, team: int):
    old_team = G.nodes[v]["team"]
    #if old_team == team:
    #    print("swapped")
    G.nodes[v]["team"] = team
    return G

if __name__ == '__main__':
    main()
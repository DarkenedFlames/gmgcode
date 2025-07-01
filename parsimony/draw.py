from gmgcode.groups.canons import Dih, canonicals
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def normalize(x, n):
    G = Dih(n)
    return min(g(x) for g in G)

def draw_parsimony(n=12, k=3, edge_type="parsimony", save_as=None):
    prime_sets = list(canonicals(n, k))
    index_map = {p: i for i, p in enumerate(prime_sets)}
    node_count = len(prime_sets)
    adj = np.zeros((node_count, node_count), dtype=int)

    def int_to_bits(n, width):
        return format(n, f'0{width}b')

    def get_parsimonious_neighbors(x):
        bits = list(int_to_bits(x, n))
        neighbors = set()
        for i in range(n):
            if bits[i] == '1':
                for offset in [-1, 1]:
                    j = (i + offset) % n
                    if bits[j] == '0':
                        new_bits = bits.copy()
                        new_bits[i] = '0'
                        new_bits[j] = '1'
                        neighbors.add(int(''.join(new_bits), 2))
        return neighbors

    def get_split_neighbors(x):
        bits = list(int_to_bits(x, n))
        neighbors = set()
        for i in range(n):
            if bits[i] == '1':
                for offset in [-1, 1]:
                    j = (i + offset) % n
                    if bits[j] == '0':
                        new_bits = bits.copy()
                        new_bits[j] = '1'
                        neighbors.add(int(''.join(new_bits), 2))
        return neighbors

    def get_hamming_neighbors(x):
        return {x ^ (1 << i) for i in range(n)}

    for i, p in enumerate(prime_sets):
        neighbors = set()
        if edge_type == "parsimony":
            neighbors |= get_parsimonious_neighbors(p)
        elif edge_type == "split":
            neighbors |= get_split_neighbors(p)
        elif edge_type == "all_parsimony":
            neighbors |= get_parsimonious_neighbors(p)
            neighbors |= get_split_neighbors(p)
        elif edge_type == "hamming":
            neighbors |= get_hamming_neighbors(p)
        else:
            raise ValueError(f"Unknown edge type: {edge_type}")

        for q in neighbors:
            q_norm = normalize(q, n)
            if q_norm in index_map:
                j = index_map[q_norm]
                adj[i, j] = 1
                adj[j, i] = 1

    G = nx.from_numpy_array(adj)
    labels = {i: int_to_bits(prime_sets[i], n) for i in range(node_count)}
    title = f"{edge_type.replace('_', ' ').capitalize()} Voice Leading on canonical sets of cardinality {k} in {n}-ET"


    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=40)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, labels, font_size=6)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    if save_as:
        plt.savefig(save_as, dpi=300)
        print(f"Graph saved to: {save_as}")
        plt.close()
    else:
        plt.show()


    return G
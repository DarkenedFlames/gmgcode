from ..pcs_object import PCSet
import networkx as nx
import matplotlib.pyplot as plt

# --- Core Set Definitions ---
def get_sets(n, k):
    sets = [i for i in range(1 << n) if i.bit_count() == k]
    canons = PCSet(tuple(range(k)), n).transform.canonicals
    anco = [i for i in canons if PCSet(i, n).intervallic.cohemitonia == 0]
    return sets, canons, anco

# --- Neighbor Function ---
def prime_neighbors(x, n):
    return {PCSet(i, n).transform.prime for i in PCSet(x, n).transform.neighbors}

# --- Graph Builder ---
def draw_parsimony(n=12, k=3, mode='canons', save_as=None, width_px=1920, height_px=1080, dpi=150):
    k = min(k, n-k)
    # Build Graph
    sets, canons, anco = get_sets(n, k)
    match mode:
        case 'all'   : C, title_adjust = sets  , 'All'
        case 'canons': C, title_adjust = canons, 'Canonical'
        case 'anco'  : C, title_adjust = anco  , 'Cluster-Free'
        case _       : raise ValueError(f"Invalid mode: '{mode}'.")

    idx_of = {x: i for i, x in enumerate(C)}
    adj = [[] for _ in C]
    for x in C:
        adj[idx_of[x]] = [idx_of[y] for y in prime_neighbors(x, n) if y in idx_of]
    
    G = nx.Graph()
    for i, x in enumerate(C):
        G.add_node(i, label=x, layer=round(PCSet(x, n).intervallic.variation, 3))
    for i, xs in enumerate(adj):
        for j in xs:
            G.add_edge(i, j)

    # Plot Graph
    figsize = (width_px / dpi, height_px / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    pos = nx.multipartite_layout(G, subset_key='layer')
    labels = nx.get_node_attributes(G, 'label')

    nx.draw(G, pos, ax=ax, with_labels=False, node_size=2000, node_color='mediumpurple')
    nx.draw_networkx_labels(G, pos, labels, font_size=20, font_family='serif', ax=ax)

    xs, ys = zip(*pos.values())
    x_margin = (max(xs) - min(xs)) * 0.05
    y_margin = (max(ys) - min(ys)) * 0.2
    ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
    ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)


    title = 'Multipartite Layout: Parsimonious Voice-Leading Graph of ' + title_adjust + f' {k}-sets in {n}-ET by Spectra Variation'
    ax.set_title(title, fontsize=15, pad=20, fontfamily='serif')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    plt.tight_layout()
    if save_as:
        fig.savefig(save_as, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
    else:
        plt.show()

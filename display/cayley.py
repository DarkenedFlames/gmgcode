import matplotlib.pyplot as plt
import networkx as nx
import math
from matplotlib.patches import FancyArrowPatch

def draw_cayley(n):
    superscript_map = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³',
        '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷',
        '8': '⁸', '9': '⁹'
    }
    
    def to_superscript(n):
        return ''.join(superscript_map[d] for d in str(n))
    
    def format_r(i):
        return 'e' if i == 0 else f'r{to_superscript(i)}'

    inner_labels = [format_r(i) for i in range(n)]
    outer_labels = [label + 's' for label in inner_labels]

    G = nx.DiGraph()
    G.add_nodes_from(inner_labels + outer_labels)

    theta_edges = []
    r_edges = []

    for i in range(n):
        u = inner_labels[i]
        v = inner_labels[(i + 1) % n]
        G.add_edge(u, v)
        theta_edges.append((u, v))


    for i in range(n):
        u = outer_labels[i]
        v = outer_labels[(i + 1) % n]  # <== this change reverses arrow direction
        G.add_edge(u, v)
        theta_edges.append((u, v))
    
    for i in range(n):
        inner = inner_labels[i]
        outer = outer_labels[i]
        G.add_edge(inner, outer)
        G.add_edge(outer, inner)
        r_edges.append((inner, outer))
        r_edges.append((outer, inner))

    def get_circular_positions(labels, radius, top_index=0):
        angle_step = 2 * math.pi / len(labels)
        return {
            label: (
                radius * math.cos((i - top_index) * angle_step + math.pi/2),
                radius * math.sin((i - top_index) * angle_step + math.pi/2)
            )
            for i, label in enumerate(labels)
        }

    pos = {
        **get_circular_positions(inner_labels, 0.625, top_index=0),
        **get_circular_positions(outer_labels, 1, top_index=0)
    }

    fig, ax = plt.subplots(figsize=(8, 8))

    nx.draw_networkx_nodes(
        G, pos,
        node_color='white',
        edgecolors='black',
        linewidths=2.0,
        node_size=1600 if n <= 12 else 1000,
        ax=ax
    )

    nx.draw_networkx_labels(G, pos, font_size=16 if n <= 12 else 10, font_weight='bold', ax=ax)

    def draw_adjusted_arrow(ax, p1, p2, color, curvature=0.0):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(dx, dy)
        shrink = 0.1 / length
        new_p1 = (p1[0] + dx * shrink, p1[1] + dy * shrink)
        new_p2 = (p2[0] - dx * shrink, p2[1] - dy * shrink)

        arrow = FancyArrowPatch(
            posA=new_p1, posB=new_p2,
            arrowstyle='-|>',
            color=color,
            linewidth=1.5,
            mutation_scale=15,
            connectionstyle=f"arc3,rad={curvature}",
        )
        ax.add_patch(arrow)

    for u, v in theta_edges:
        draw_adjusted_arrow(ax, pos[u], pos[v], color='purple', curvature=0)

    for u, v in r_edges:
        draw_adjusted_arrow(ax, pos[u], pos[v], color='orange', curvature=0)

    ax.set_aspect('equal')
    ax.axis('off')
    plt.title(f"Cayley Graph of D{n}", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.patheffects as path_effects


def draw_clock(
    n_nodes,
    active_nodes=(),
    background='none',
    active_color='white',
    inactive_color='black',
    show_numbers=True,
    dotted_axes=None,
    axis_color='black',
    arrow_coords=None,
    arrow_color='black',
    bidirectional_arrows=False,
    center_text=None,
    center_text_color='black',
    center_text_size=40,
    save_as=None
):
    # Create graph
    G = nx.cycle_graph(n_nodes)
    layout_radius = 1.5

    # Get evenly spaced clockwise angles, starting from top (12 o'clock)
    theta = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    pos = {
        i: (
            layout_radius * np.cos(-theta[i] + np.pi/2),  # flip angle for clockwise, shift π/2 to top
            layout_radius * np.sin(-theta[i] + np.pi/2)
        )
        for i in range(n_nodes)
    }

    # Dynamic node sizing with cap
    base_size = 3750
    node_size = min(base_size * (12 / n_nodes)**1.5, 3750)

    # Color setup
    node_colors = [active_color if i in active_nodes else inactive_color for i in G.nodes()]
    node_edges = [inactive_color if i in active_nodes else active_color for i in G.nodes()]

    # Begin drawing
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='white', width=2)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        edgecolors=node_edges,
        linewidths=3,
        node_size=node_size
    )

    # Label active nodes only, with numbers in clockwise order
    if show_numbers:
        for i in range(n_nodes):
            if i in active_nodes:
                x, y = pos[i]
                ax.text(
                    x, y, str(i),
                    fontsize=40, color='black',
                    family='serif',
                    ha='center', va='center',
                    path_effects=
    [path_effects.withStroke(linewidth=2, foreground='black')]
                )

    # Dotted axes through specified node pairs (e.g., i to i + n/2)
    if dotted_axes:
        for i in dotted_axes:
            a = np.array(pos[i])
            b = np.array(pos[(i + n_nodes // 2) % n_nodes])
            ax.plot([a[0], b[0]], [a[1], b[1]],
                    linestyle='dotted', color=axis_color, lw=5, zorder=0)

    # Arrows between coordinates
    if arrow_coords:
        for src, dst in arrow_coords:
            ax.annotate("",
                        xy=pos[dst], xycoords='data',
                        xytext=pos[src], textcoords='data',
                        arrowprops=dict(
                            arrowstyle="->",
                            color=arrow_color,
                            lw=4,
                            shrinkA=0, shrinkB=0,
                            connectionstyle='arc3,rad=0.0'
                        ),
                        zorder=4)
            if bidirectional_arrows:
                ax.annotate("",
                            xy=pos[src], xycoords='data',
                            xytext=pos[dst], textcoords='data',
                            arrowprops=dict(
                                arrowstyle="->",
                                color='white',
                                lw=4,
                                shrinkA=0, shrinkB=0,
                                connectionstyle='arc3,rad=0.0'
                            ),
                            zorder=4)
            

    # Adjust view to include large nodes
    node_radius = (node_size ** 0.5) / 3000
    padding = 0.5 + node_radius
    limit = layout_radius + padding
    
    # Optional center text
    if center_text is not None:
        ax.text(
            0, 0, str(center_text),
            color=center_text_color,
            fontsize=center_text_size,
            family='serif',
            ha='center', va='center',
            path_effects=[path_effects.withStroke(linewidth=2, foreground='black')]
        )
    
    
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    if save_as:
        plt.savefig(
            save_as, 
            dpi=200,    
            facecolor=fig.get_facecolor(), 
            bbox_inches='tight'
        )
        plt.close(fig)
        
    else:
        plt.show()
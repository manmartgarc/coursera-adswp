# -*- coding: utf-8 -*-
"""
Created on Sun May 27 01:36:54 2018

@author: manma
"""

import networkx as nx

# %%
def answer_one():
    G = nx.read_edgelist('email_network.txt',
                         create_using=nx.MultiDiGraph(),
                         data=[('time', int)],
                         nodetype=str)
    return G
answer_one()

# %%
def answer_two():
    G = answer_one()
    n_employees = len(G.nodes())
    n_emails = len(G.edges())
    return (n_employees, n_emails)
answer_two()

# %%
def answer_three():
    G = answer_one()
    part_a = nx.is_strongly_connected(G)
    part_b = nx.is_weakly_connected(G)
    return (part_a, part_b)
answer_three()

# %%
def answer_four():
    G = answer_one()
    weakly_connected = sorted(nx.weakly_connected_components(G))
    max_wc = len(max(weakly_connected, key=len))
    return max_wc
answer_four()

# %%
def answer_five():
    G = answer_one()
    strongly_connected = sorted(nx.strongly_connected_components(G))
    max_sc = len(max(strongly_connected, key=len))
    return max_sc
answer_five()

# %%
def answer_six():
    G = answer_one()
    G_sc = max(nx.strongly_connected_component_subgraphs(G), key=len)
    return G_sc
answer_six()

# %%
def answer_seven():
    G_sc = answer_six()
    mean_dist = nx.average_shortest_path_length(G_sc)
    return mean_dist
answer_seven()

# %%
def answer_eight():
    G_sc = answer_six()
    diameter = nx.diameter(G_sc)
    return diameter
answer_eight()

# %%
def answer_nine():
    G_sc = answer_six()
    periphery = nx.periphery(G_sc)
    return set(periphery)
answer_nine()

# %%
def answer_ten():
    G_sc = answer_six()
    center = nx.center(G_sc)
    return set(center)
answer_ten()

# %%
def answer_eleven():
    scores = {}
    G_sc = answer_six()
    diameter = nx.diameter(G_sc)
    periphery = nx.periphery(G_sc)
    for peripheral in periphery:
        shortest = nx.shortest_path_length(G_sc, peripheral)
        current = len([k for k,v in shortest.items() if v == diameter])
        scores[peripheral] = current
    return (str(max(scores)), scores[max(scores)])
answer_eleven()

# %%
def answer_twelve():
    nodes_to_cut = set()
    G_sc = answer_six()
    target = answer_eleven()[0]
    sources = nx.center(G_sc)
    for source in sources:
        min_nodes_to_cut = nx.minimum_node_cut(G_sc, s=source, t=target)
        nodes_to_cut.update(min_nodes_to_cut)
    n_min_nodes_to_cut = len(nodes_to_cut)
    return n_min_nodes_to_cut
answer_twelve()

# %%
def answer_thirteen():
    G_sc = answer_six()
    G_sc_un = G_sc.to_undirected()
    G_un = nx.Graph(G_sc_un)
    return G_un
answer_thirteen()

# %%
def answer_fourteen():
    G_un = answer_thirteen()
    transitivity = nx.transitivity(G_un)
    avg_clust = nx.average_clustering(G_un)
    return (transitivity, avg_clust)
answer_fourteen()

# %%
# Draw for fun
import matplotlib.pyplot as plt

G_un = answer_thirteen()
plt.figure(figsize=(10,7))

pos = nx.spring_layout(G_un)
node_color = [G_un.degree(v) for v in G_un]
edge_width = [0.0015*G_un[u][v]['time'] for u,v in G_un.edges()]

nx.draw_networkx(G_un, pos, node_color=node_color, alpha=0.7,
                 edge_width=edge_width, with_labels=False,
                 edge_color='.4', cmap=plt.cm.inferno)


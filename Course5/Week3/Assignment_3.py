import networkx as nx

path = ('C:/Users/manma/Google Drive/GitHub/'
        'Coursera-Applied-Data-Science-with-Python/Course5/Week3/')
G1 = nx.read_gml(path + 'friendships.gml')

def answer_one():
    """
    Find the degree centrality, closeness centrality, and normalized betweeness
    centrality (excluding endpoints) of node 100.

    This function should return a tuple of floats
    (degree_centrality, closeness_centrality, betweenness_centrality).
    """
    deg_c = nx.degree_centrality(G1)[100]
    clo_c = nx.closeness_centrality(G1)[100]
    bet_c = nx.betweenness_centrality(G1)[100]
    return (deg_c, clo_c, bet_c)

def answer_two():

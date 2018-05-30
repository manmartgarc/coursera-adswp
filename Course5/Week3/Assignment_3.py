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
    """
    Suppose you are employed by an online shopping website and are tasked with
    selecting one user in network G1 to send an online shopping voucher to.
    We expect that the user who receives the voucher will send it to their
    friends in the network. You want the voucher to reach as many nodes
    as possible. The voucher can be forwarded to multiple users at the
    same time, but the travel distance of the voucher is limited to one step,
    which means if the voucher travels more than one step in this network,
    it is no longer valid.
    Apply your knowledge in network centrality to select the best
    candidate for the voucher.
    """

    clo_c_all = nx.degree_centrality(G1)
    max_degree_node = max(clo_c_all.items(), key=lambda x: x[1])
    return max_degree_node[0]
    

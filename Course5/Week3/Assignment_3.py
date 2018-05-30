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

    deg_c_all = nx.degree_centrality(G1)
    max_degree_node = max(deg_c_all.items(), key=lambda x: x[1])
    return max_degree_node[0]

def answer_three():
    """
    Now the limit of the voucher’s travel distance has been removed.
    Because the network is connected, regardless of who you pick, every node
    in the network will eventually receive the voucher. However, we now want
    to ensure that the voucher reaches the nodes in the lowest average number
    of hops.

    How would you change your selection strategy? Write a function to tell
    us who is the best candidate in the network under this condition.
    """

    clo_c_all = nx.closeness_centrality(G1)
    max_clo_node = max(clo_c_all.items(), key=lambda x: x[1])
    return max_clo_node[0]

def answer_four():
    """
    Assume the restriction on the voucher’s travel distance is still removed,
    but now a competitor has developed a strategy to remove a person from the
    network in order to disrupt the distribution of your company’s voucher.
    Your competitor is specifically targeting people who are often bridges of
    information flow between other pairs of people. Identify the single
    riskiest person to be removed under your competitor’s strategy?
    """

    bet_c_all = nx.betweenness_centrality(G1)
    max_bet_node = max(bet_c_all.items(), key=lambda x: x[1])
    return max_bet_node[0]

G2 = nx.read_gml(path + 'blogs.gml')

def answer_five():
    """
    Apply the Scaled Page Rank Algorithm to this network.
    Find the Page Rank of node 'realclearpolitics.com' with damping value 0.85.
    """

    pg_rank = nx.pagerank(G2, alpha=0.85)
    pg_rank_node = pg_rank['realclearpolitics.com']
    return pg_rank_node

def answer_six():
    """
    Apply the Scaled Page Rank Algorithm to this network with damping
    value 0.85. Find the 5 nodes with highest Page Rank.
    """

    pg_rank = nx.pagerank(G2, alpha=0.85)
    sorted_pg_rank = sorted(pg_rank.items(), reverse=True, key=lambda x: x[1])
    top_5 = sorted_pg_rank[:5]
    top_5_blogs = [blog for blog, pg_rank in top_5]
    return top_5_blogs

def answer_seven():
    """
    Apply the HITS Algorithm to the network to find the hub and authority
    scores of node 'realclearpolitics.com'.
    """

    hits = nx.hits(G2)

import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms import bipartite

# This is the set of employees
employees = set(['Pablo',
                 'Lee',
                 'Georgia',
                 'Vincent',
                 'Andy',
                 'Frida',
                 'Joan',
                 'Claude'])

# This is the set of movies
movies = set(['The Shawshank Redemption',
              'Forrest Gump',
              'The Matrix',
              'Anaconda',
              'The Social Network',
              'The Godfather',
              'Monty Python and the Holy Grail',
              'Snakes on a Plane',
              'Kung Fu Panda',
              'The Dark Knight',
              'Mean Girls'])

def plot_graph(G, weight_name=None):
    '''
    G: a networkx G
    weight_name: name of the attribute for plotting edge weights (if G is weighted)
    '''
    #%matplotlib notebook
    import matplotlib.pyplot as plt

    plt.figure()
    pos = nx.spring_layout(G)
    edges = G.edges()
    weights = None

    if weight_name:
        weights = [int(G[u][v][weight_name]) for u,v in edges]
        labels = nx.get_edge_attributes(G,weight_name)
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
        nx.draw_networkx(G, pos, edges=edges, width=weights);
    else:
        nx.draw_networkx(G, pos, edges=edges);
        
# %%
def answer_one():
    df = pd.read_table('Employee_Movie_choices.txt')
    G = nx.from_pandas_dataframe(df, '#Employee', 'Movie')
    return G
answer_one()

# %%
def answer_two():
    G_2 = answer_one()
    G_2.add_nodes_from(employees, type='employee')
    G_2.add_nodes_from(movies, type='movie')
    return G_2

# %%
def answer_three():
    G_2 = answer_two()
    G_2_weighted = nx.bipartite.weighted_projected_graph(G_2, employees)
    return G_2_weighted

# %%
def answer_four():
    relationships = pd.read_table('Employee_Relationships.txt',
                                  header=None)
    relationships.rename(columns={0:'from',
                                  1:'to',
                                  2:'relationships'}, inplace=True)
    common_movies = pd.DataFrame.from_dict(answer_three().edges(data=True))
    common_movies[2] = common_movies[2].apply(lambda x: x['weight'])
    
    for index, row in relationships.iterrows():
        e1a = row[0]
        e2a = row[1]
        for index_2, row_2 in common_movies.iterrows():
            e1b = row_2[0]
            e2b = row_2[1]
            if ((e1a == e1b) & (e2a == e2b)) | ((e1a == e2b) & (e2a == e1b)):
                relationships.loc[index, 'common_movies'] = row_2[2]
    relationships['common_movies'].fillna(0, inplace=True)
    corr = relationships.corr()['common_movies'][0]
    return corr
answer_four()
    
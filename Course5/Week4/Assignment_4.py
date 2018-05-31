import networkx as nx
import pandas as pd
import numpy as np
import pickle

path = ('C:/Users/manma/Google Drive/GitHub/'
        'Coursera-Applied-Data-Science-with-Python/Course5/Week4/')
P1_Graphs = pickle.load(open(path + 'A4_graphs', 'rb'))

def graph_identification():
    for i, G in enumerate(P1_Graphs, start=1):
        degrees = list(G.degree().values())
        mean_d = np.mean(degrees)
        min = np.min(degrees)
        max = np.max(degrees)
        mean_sp = nx.average_shortest_path_length(G)
        mean_cl = nx.average_clustering(G)
        print('G{} - d_mean: {:.2f}, d_min: {}, d_max: {}, '
              'mean_sp: {:.2f}, mean_cl: {:.2f}'
              .format(i, mean_d, min, max, mean_sp, mean_cl))
    return ['PA', 'SW_L', 'SW_L', 'PA', 'SW_H']

G = nx.read_gpickle(path + 'email_prediction.txt')
print(nx.info(G))

def salary_predictions():
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import roc_auc_score

    df = pd.DataFrame(index=G.nodes())
    df['department'] = pd.Series(nx.get_node_attributes(G, 'Department'))
    df['degree'] = pd.Series(G.degree())
    df['clust'] = pd.Series(nx.clustering(G))
    df['deg_c'] = pd.Series(nx.degree_centrality(G))
    df['close_c'] = pd.Series(nx.closeness_centrality(G))
    df['btwn_c'] = pd.Series(nx.betweenness_centrality(G))
    df['pg_rank'] = pd.Series(nx.pagerank(G, alpha=0.85))
    df['hits_hub'] = pd.Series(nx.hits(G)[0])
    df['hits_aut'] = pd.Series(nx.hits(G)[1])
    df['m_salary'] = pd.Series(nx.get_node_attributes(G, 'ManagementSalary'))

    keep = df[~df['m_salary'].isnull()]
    hold = df[df['m_salary'].isnull()]
    X_keep = keep.drop(columns='m_salary')
    y_keep = keep['m_salary']
    X_hold = hold.drop(columns='m_salary')

    X_train, X_test, y_train, y_test = train_test_split(X_keep, y_keep,
                                                        random_state=1337)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    X_hold_scaled = scaler.fit_transform(X_hold)

    param_grid = {'C':[10 ** a for a in range(-6,2)],
                  'gamma':[10 ** a for a in range(-6, -2)],
                  'class_weight':[None, 'balanced'],
                  'kernel':['rbf', 'linear']}
    svc = SVC(probability=True, random_state=0)
    clf = GridSearchCV(svc, param_grid, scoring='roc_auc')
    clf.fit(X_train_scaled, y_train)
    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test_scaled)[:, 1])
    answer = pd.Series(data=clf.predict_proba(X_hold_scaled)[:, 1],
                       index=X_hold.index)
    return answer

def new_connections_predictions():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    future_connections = pd.read_csv(path
                                     + 'Future_Connections.csv',
                                     index_col=0,
                                     converters={0:eval})

    def communities(row):
        """
        Check to whether are in the same department or notself.
        Vectorized for rows, use with pd.DataFrame.apply(x, axis=1)
        """
        nodes = row.name
        a = nodes[0]
        b = nodes[1]
        comm_a = G.node[a]['Department']
        comm_b = G.node[b]['Department']
        if comm_a == comm_b:
            return 1
        else:
            return 0

    future_connections['same_comm'] = future_connections.apply(communities,
                                                               axis=1)
    # For Soundarajan-Hopcroft algorithms.
    for node in G.nodes():
       G.node[node]['community'] = G.node[node]['Department']

    pa = list(nx.preferential_attachment(G))
    pa_df = pd.DataFrame(index=[(i[0], i[1]) for i in pa],
                         data={'pref_att':[i[2] for i in pa]})

    cn = [(e[0], e[1], len(list(nx.common_neighbors(G, e[0], e[1]))))
          for e in nx.non_edges(G)]
    cn_df = pd.DataFrame(index=[(i[0], i[1]) for i in cn],
                         data={'comm_neigh':[i[2] for i in cn]})

    cnsh = list(nx.cn_soundarajan_hopcroft(G))
    cnsh_df = pd.DataFrame(index=[(i[0], i[1]) for i in cnsh],
                         data={'sh_comm_neigh':[i[2] for i in cnsh]})

    ra = list(nx.resource_allocation_index(G))
    ra_df = pd.DataFrame(index=[(i[0], i[1]) for i in ra],
                         data={'reso_alloc':[i[2] for i in ra]})

    rash = list(nx.ra_index_soundarajan_hopcroft(G))
    rash_df = pd.DataFrame(index=[(i[0], i[1]) for i in rash],
                         data={'sh_reso_alloc':[i[2] for i in rash]})

    jc = [i for i in nx.jaccard_coefficient(G)]
    jc_df = pd.DataFrame(index=[(i[0], i[1]) for i in jc],
                         data={'jacc_coeff':[i[2] for i in jc]})

    for df in [pa_df, cn_df, cnsh_df, ra_df, rash_df, jc_df]:
        future_connections = future_connections.merge(df, how='left',
                                                      left_index=True,
                                                      right_index=True)

    keep = future_connections[~future_connections['Future Connection'].isnull()]
    hold = future_connections[future_connections['Future Connection'].isnull()]

    X_keep = keep.drop('Future Connection', axis=1)
    y_keep = keep['Future Connection']
    X_hold = hold.drop('Future Connection', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_keep, y_keep,
                                                        random_state=0)
    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)

    # Check on ROC_AUC performance.
    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    probs = clf.predict_proba(X_hold)[:, 1]
    answer = pd.Series(index=X_hold.index,
                       data=probs)
    return answer

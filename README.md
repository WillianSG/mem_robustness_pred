# SNNs Robustness
Learning to Predict Memory Robustness from Spiking Neural Networks.

## dataset

**Load dataset**

```py
import pickle

dataset = pickle.load(open('RCN_dataset.pickle', 'rb'))

dataset['features'] # axis labels (metrics)
dataset['X'] 		# 300 datapoints (13D vectors)
dataset['Y'] 		# 300 target values (maximal connections drops)
```

**Features**

0. **nonrecurrent_count**: number of non-reciprocal edges (a->b but not b->a).
1. **recurrent_count**: number of reciprocal edges (a->b and b->a).
2. **cliques_count**: number of maximal cliques within the graph.
3. **k_edge_connect**: k value at which the graph becomes disconnected.
4. **clique_size_avg**: average maximal clique size (number of nodes).
5. **clique_size_std**: standard deviation of _clique_size_avg_.
6. **cs_max**: biggest maximal clique.
7. **in_degree_centrality_avg**: average in-degree centrality of nodes.
8. **in_degree_centrality_std**: standard deviation of _in_degree_centrality_avg_.
9. **out_degree_centrality_avg**: average out-degree centrality of nodes.
10. **out_degree_centrality_std**: standard deviation of _out_degree_centrality_avg_.
11. **between_centrality_avg**: average betweenness-centrality of nodes.
12. **between_centrality_std**: standard deviation of _between_centrality_avg_.

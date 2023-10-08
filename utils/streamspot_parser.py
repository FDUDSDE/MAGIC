import networkx as nx
from tqdm import tqdm
import json
raw_path = '../data/streamspot/'
processed_path = '../data/streamspot/'

NUM_GRAPHS = 600
node_type_dict = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
edge_type_dict = ['i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                  'q', 't', 'u', 'v', 'w', 'y', 'z', 'A', 'C', 'D', 'E', 'G']

for i in tqdm(range(NUM_GRAPHS)):
    f = open(raw_path + str(i + 1) + '.json', 'r', encoding='utf-8')
    G = nx.node_link_graph(json.load(f))
    for n in G.nodes():
        G.nodes[n]['type'] = node_type_dict.index(G.nodes[n]['type'])
    for e in G.edges():
        G.edges[e]['type'] = edge_type_dict.index(G.edges[e]['type'])
    f.close()
    f = open(raw_path + str(i + 1) + '.json', 'w', encoding='utf-8')
    json.dump(nx.node_link_data(G), f)


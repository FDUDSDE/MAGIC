import networkx as nx
from tqdm import tqdm
import json
raw_path = '../data/streamspot/'

NUM_GRAPHS = 600
node_type_dict = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
edge_type_dict = ['i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                  'q', 't', 'u', 'v', 'w', 'y', 'z', 'A', 'C', 'D', 'E', 'G']
node_type_set = set(node_type_dict)
edge_type_set = set(edge_type_dict)

count_graph = 0
with open(raw_path + 'all.tsv', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    g = nx.DiGraph()
    node_map = {}
    count_node = 0
    for line in tqdm(lines):
        src, src_type, dst, dst_type, etype, graph_id = line.strip('\n').split('\t')
        graph_id = int(graph_id)
        if src_type not in node_type_set or dst_type not in node_type_set:
            continue
        if etype not in edge_type_set:
            continue
        if graph_id != count_graph:
            count_graph += 1
            for n in g.nodes():
                g.nodes[n]['type'] = node_type_dict.index(g.nodes[n]['type'])
            for e in g.edges():
                g.edges[e]['type'] = edge_type_dict.index(g.edges[e]['type'])
            f1 = open(raw_path + str(count_graph) + '.json', 'w', encoding='utf-8')
            json.dump(nx.node_link_data(g), f1)
            assert graph_id == count_graph
            g = nx.DiGraph()
            count_node = 0
        if src not in node_map:
            node_map[src] = count_node
            g.add_node(count_node, type=src_type)
            count_node += 1
        if dst not in node_map:
            node_map[dst] = count_node
            g.add_node(count_node, type=dst_type)
            count_node += 1
        if not g.has_edge(node_map[src], node_map[dst]):
            g.add_edge(node_map[src], node_map[dst], type=etype)
    count_graph += 1
    for n in g.nodes():
        g.nodes[n]['type'] = node_type_dict.index(g.nodes[n]['type'])
    for e in g.edges():
        g.edges[e]['type'] = edge_type_dict.index(g.edges[e]['type'])
    f1 = open(raw_path + str(count_graph) + '.json', 'w', encoding='utf-8')
    json.dump(nx.node_link_data(g), f1)

import json
import sys
import argparse
import os
import tqdm
import pickle as pkl
import networkx as nx
from tqdm import tqdm

valid_node_type = ['file', 'process_memory', 'task', 'mmaped_file', 'path', 'socket', 'address', 'link']


def read_single_graph(file_name, threshold):
    graph = []
    edge_cnt = 0
    with open(file_name, 'r') as f:
        for line in f:
            try:
                edge = line.strip().split("\t")
                new_edge = [edge[0], edge[1]]
                attributes = edge[2].strip().split(":")
                source_node_type = attributes[0]
                destination_node_type = attributes[1]
                edge_type = attributes[2]
                edge_order = attributes[3]

                new_edge.append(source_node_type)
                new_edge.append(destination_node_type)
                new_edge.append(edge_type)
                new_edge.append(edge_order)
                graph.append(new_edge)
                edge_cnt += 1
            except:
                print("{}".format(line))
    f.close()
    graph.sort(key=lambda e: e[5])
    if len(graph) <= threshold:
        return graph
    else:
        return graph[:threshold]


def process_graph(name, threshold):
    graph = read_single_graph(name, threshold)
    result_graph = nx.DiGraph()
    cnt = 0
    for num, edge in enumerate(graph):
        cnt += 1
        src, dst, src_type, dst_type, edge_type = edge[:5]
        if True:# src_type in valid_node_type and dst_type in valid_node_type:
            if not result_graph.has_node(src):
                result_graph.add_node(src, type=src_type)
            if not result_graph.has_node(dst):
                result_graph.add_node(dst, type=dst_type)
            if not result_graph.has_edge(src, dst):
                result_graph.add_edge(src, dst, type=edge_type)
                if bidirection:
                    result_graph.add_edge(dst, src, type='reverse_{}'.format(edge_type))
    return cnt, result_graph


node_type_list = []
edge_type_list = []
node_type_dict = {}
edge_type_dict = {}


def format_graph(g, name):
    new_g = nx.DiGraph()
    node_map = {}
    node_cnt = 0
    for n in g.nodes:
        node_map[n] = node_cnt
        new_g.add_node(node_cnt, type=g.nodes[n]['type'])
        node_cnt += 1
    for e in g.edges:
        new_g.add_edge(node_map[e[0]], node_map[e[1]], type=g.edges[e]['type'])
    for n in new_g.nodes:
        node_type = new_g.nodes[n]['type']
        if not node_type in node_type_dict:
            node_type_list.append(node_type)
            node_type_dict[node_type] = 1
        else:
            node_type_dict[node_type] += 1
    for e in new_g.edges:
        edge_type = new_g.edges[e]['type']
        if not edge_type in edge_type_dict:
            edge_type_list.append(edge_type)
            edge_type_dict[edge_type] = 1
        else:
            edge_type_dict[edge_type] += 1
    for n in new_g.nodes:
        new_g.nodes[n]['type'] = node_type_list.index(new_g.nodes[n]['type'])
    for e in new_g.edges:
        new_g.edges[e]['type'] = edge_type_list.index(new_g.edges[e]['type'])
    with open('{}.json'.format(name), 'w', encoding='utf-8') as f:
        json.dump(nx.node_link_data(new_g), f)


if __name__ == "__main__":
    bidirection = False
    threshold = 10000000 # infinity
    interaction_dict = []
    graph_cnt = 0
    result_graphs = []
    cnt = 0
    input = "../data/unicorn/processed/"
    base = "../data/unicorn/final/"

    for i in tqdm(range(150)):
        single_cnt, result_graph = process_graph('{}{}.log'.format(input, i + 1), threshold)
        format_graph(result_graph, '{}{}'.format(base, i))
        cnt += single_cnt

    print(cnt // 150)
    print(len(node_type_list))
    print(node_type_dict)
    print(len(edge_type_list))
    print(edge_type_dict)


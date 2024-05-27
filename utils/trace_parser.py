import argparse
import json
import os
import random
import re

from tqdm import tqdm
import networkx as nx
import pickle as pkl


node_type_dict = {}
edge_type_dict = {}
node_type_cnt = 0
edge_type_cnt = 0

metadata = {
    'trace':{
        'train': ['ta1-trace-e3-official-1.json.0', 'ta1-trace-e3-official-1.json.1', 'ta1-trace-e3-official-1.json.2', 'ta1-trace-e3-official-1.json.3'],
        'test': ['ta1-trace-e3-official-1.json.0', 'ta1-trace-e3-official-1.json.1', 'ta1-trace-e3-official-1.json.2', 'ta1-trace-e3-official-1.json.3', 'ta1-trace-e3-official-1.json.4']
    },
    'theia':{
            'train': ['ta1-theia-e3-official-6r.json', 'ta1-theia-e3-official-6r.json.1', 'ta1-theia-e3-official-6r.json.2', 'ta1-theia-e3-official-6r.json.3'],
            'test': ['ta1-theia-e3-official-6r.json.8']
    },
    'cadets':{
            'train': ['ta1-cadets-e3-official.json','ta1-cadets-e3-official.json.1', 'ta1-cadets-e3-official.json.2', 'ta1-cadets-e3-official-2.json.1'],
            'test': ['ta1-cadets-e3-official-2.json']
    }
}


pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')
pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_type = re.compile(r'type\":\"(.*?)\"')
pattern_time = re.compile(r'timestampNanos\":(.*?),')
pattern_file_name = re.compile(r'map\":\{\"path\":\"(.*?)\"')
pattern_process_name = re.compile(r'map\":\{\"name\":\"(.*?)\"')
pattern_netflow_object_name = re.compile(r'remoteAddress\":\"(.*?)\"')


def read_single_graph(dataset, malicious, path, test=False):
    global node_type_cnt, edge_type_cnt
    g = nx.DiGraph()
    print('converting {} ...'.format(path))
    path = '../data/{}/'.format(dataset) + path + '.txt'
    f = open(path, 'r')
    lines = []
    for l in f.readlines():
        split_line = l.split('\t')
        src, src_type, dst, dst_type, edge_type, ts = split_line
        ts = int(ts)
        if not test:
            if src in malicious or dst in malicious:
                if src in malicious and src_type != 'MemoryObject':
                    continue
                if dst in malicious and dst_type != 'MemoryObject':
                    continue

        if src_type not in node_type_dict:
            node_type_dict[src_type] = node_type_cnt
            node_type_cnt += 1
        if dst_type not in node_type_dict:
            node_type_dict[dst_type] = node_type_cnt
            node_type_cnt += 1
        if edge_type not in edge_type_dict:
            edge_type_dict[edge_type] = edge_type_cnt
            edge_type_cnt += 1
        if 'READ' in edge_type or 'RECV' in edge_type or 'LOAD' in edge_type:
            lines.append([dst, src, dst_type, src_type, edge_type, ts])
        else:
            lines.append([src, dst, src_type, dst_type, edge_type, ts])
    lines.sort(key=lambda l: l[5])

    node_map = {}
    node_type_map = {}
    node_cnt = 0
    node_list = []
    for l in lines:
        src, dst, src_type, dst_type, edge_type = l[:5]
        src_type_id = node_type_dict[src_type]
        dst_type_id = node_type_dict[dst_type]
        edge_type_id = edge_type_dict[edge_type]
        if src not in node_map:
            node_map[src] = node_cnt
            g.add_node(node_cnt, type=src_type_id)
            node_list.append(src)
            node_type_map[src] = src_type
            node_cnt += 1
        if dst not in node_map:
            node_map[dst] = node_cnt
            g.add_node(node_cnt, type=dst_type_id)
            node_type_map[dst] = dst_type
            node_list.append(dst)
            node_cnt += 1
        if not g.has_edge(node_map[src], node_map[dst]):
            g.add_edge(node_map[src], node_map[dst], type=edge_type_id)
    return node_map, g


def preprocess_dataset(dataset):
    id_nodetype_map = {}
    id_nodename_map = {}
    for file in os.listdir('../data/{}/'.format(dataset)):
        if 'json' in file and not '.txt' in file and not 'names' in file and not 'types' in file and not 'metadata' in file:
            print('reading {} ...'.format(file))
            f = open('../data/{}/'.format(dataset) + file, 'r', encoding='utf-8')
            for line in tqdm(f):
                if 'com.bbn.tc.schema.avro.cdm18.Event' in line or 'com.bbn.tc.schema.avro.cdm18.Host' in line: continue
                if 'com.bbn.tc.schema.avro.cdm18.TimeMarker' in line or 'com.bbn.tc.schema.avro.cdm18.StartMarker' in line: continue
                if 'com.bbn.tc.schema.avro.cdm18.UnitDependency' in line or 'com.bbn.tc.schema.avro.cdm18.EndMarker' in line: continue
                if len(pattern_uuid.findall(line)) == 0: print(line)
                uuid = pattern_uuid.findall(line)[0]
                subject_type = pattern_type.findall(line)

                if len(subject_type) < 1:
                    if 'com.bbn.tc.schema.avro.cdm18.MemoryObject' in line:
                        subject_type = 'MemoryObject'
                    if 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in line:
                        subject_type = 'NetFlowObject'
                    if 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
                        subject_type = 'UnnamedPipeObject'
                else:
                    subject_type = subject_type[0]

                if uuid == '00000000-0000-0000-0000-000000000000' or subject_type in ['SUBJECT_UNIT']:
                    continue
                id_nodetype_map[uuid] = subject_type
                if 'FILE' in subject_type and len(pattern_file_name.findall(line)) > 0:
                    id_nodename_map[uuid] = pattern_file_name.findall(line)[0]
                elif subject_type == 'SUBJECT_PROCESS' and len(pattern_process_name.findall(line)) > 0:
                    id_nodename_map[uuid] = pattern_process_name.findall(line)[0]
                elif subject_type == 'NetFlowObject' and len(pattern_netflow_object_name.findall(line)) > 0:
                    id_nodename_map[uuid] = pattern_netflow_object_name.findall(line)[0]
    for key in metadata[dataset]:
        for file in metadata[dataset][key]:
            if os.path.exists('../data/{}/'.format(dataset) + file + '.txt'):
                continue
            f = open('../data/{}/'.format(dataset) + file, 'r', encoding='utf-8')
            fw = open('../data/{}/'.format(dataset) + file + '.txt', 'w', encoding='utf-8')
            print('processing {} ...'.format(file))
            for line in tqdm(f):
                if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                    edgeType = pattern_type.findall(line)[0]
                    timestamp = pattern_time.findall(line)[0]
                    srcId = pattern_src.findall(line)

                    if len(srcId) == 0: continue
                    srcId = srcId[0]
                    if not srcId in id_nodetype_map:
                        continue
                    srcType = id_nodetype_map[srcId]
                    dstId1 = pattern_dst1.findall(line)
                    if len(dstId1) > 0 and dstId1[0] != 'null':
                        dstId1 = dstId1[0]
                        if not dstId1 in id_nodetype_map:
                            continue
                        dstType1 = id_nodetype_map[dstId1]
                        this_edge1 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId1) + '\t' + str(
                            dstType1) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                        fw.write(this_edge1)

                    dstId2 = pattern_dst2.findall(line)
                    if len(dstId2) > 0 and dstId2[0] != 'null':
                        dstId2 = dstId2[0]
                        if not dstId2 in id_nodetype_map.keys():
                            continue
                        dstType2 = id_nodetype_map[dstId2]
                        this_edge2 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId2) + '\t' + str(
                            dstType2) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                        fw.write(this_edge2)
            fw.close()
            f.close()
    if len(id_nodename_map) != 0:
        fw = open('../data/{}/'.format(dataset) + 'names.json', 'w', encoding='utf-8')
        json.dump(id_nodename_map, fw)
    if len(id_nodetype_map) != 0:
        fw = open('../data/{}/'.format(dataset) + 'types.json', 'w', encoding='utf-8')
        json.dump(id_nodetype_map, fw)


def read_graphs(dataset):
    malicious_entities = '../data/{}/{}.txt'.format(dataset, dataset)
    f = open(malicious_entities, 'r')
    malicious_entities = set()
    for l in f.readlines():
        malicious_entities.add(l.lstrip().rstrip())

    preprocess_dataset(dataset)
    train_gs = []
    for file in metadata[dataset]['train']:
        _, train_g = read_single_graph(dataset, malicious_entities, file, False)
        train_gs.append(train_g)
    test_gs = []
    test_node_map = {}
    count_node = 0
    for file in metadata[dataset]['test']:
        node_map, test_g = read_single_graph(dataset, malicious_entities, file, True)
        assert len(node_map) == test_g.number_of_nodes()
        test_gs.append(test_g)
        for key in node_map:
            if key not in test_node_map:
                test_node_map[key] = node_map[key] + count_node
        count_node += test_g.number_of_nodes()

    if os.path.exists('../data/{}/names.json'.format(dataset)) and os.path.exists('../data/{}/types.json'.format(dataset)):
        with open('../data/{}/names.json'.format(dataset), 'r', encoding='utf-8') as f:
            id_nodename_map = json.load(f)
        with open('../data/{}/types.json'.format(dataset), 'r', encoding='utf-8') as f:
            id_nodetype_map = json.load(f)
        f = open('../data/{}/malicious_names.txt'.format(dataset), 'w', encoding='utf-8')
        final_malicious_entities = []
        malicious_names = []
        for e in malicious_entities:
            if e in test_node_map and e in id_nodetype_map and id_nodetype_map[e] != 'MemoryObject' and id_nodetype_map[e] != 'UnnamedPipeObject':
                final_malicious_entities.append(test_node_map[e])
                if e in id_nodename_map:
                    malicious_names.append(id_nodename_map[e])
                    f.write('{}\t{}\n'.format(e, id_nodename_map[e]))
                else:
                    malicious_names.append(e)
                    f.write('{}\t{}\n'.format(e, e))
    else:
        f = open('../data/{}/malicious_names.txt'.format(dataset), 'w', encoding='utf-8')
        final_malicious_entities = []
        malicious_names = []
        for e in malicious_entities:
            if e in test_node_map:
                final_malicious_entities.append(test_node_map[e])
                malicious_names.append(e)
                f.write('{}\t{}\n'.format(e, e))

    pkl.dump((final_malicious_entities, malicious_names), open('../data/{}/malicious.pkl'.format(dataset), 'wb'))
    pkl.dump([nx.node_link_data(train_g) for train_g in train_gs], open('../data/{}/train.pkl'.format(dataset), 'wb'))
    pkl.dump([nx.node_link_data(test_g) for test_g in test_gs], open('../data/{}/test.pkl'.format(dataset), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDM Parser')
    parser.add_argument("--dataset", type=str, default="trace")
    args = parser.parse_args()
    if args.dataset not in ['trace', 'theia', 'cadets']:
        raise NotImplementedError
    read_graphs(args.dataset)


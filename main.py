import json
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import yaml

def load_config(cfg):
    with open(cfg, 'r') as f:
        return yaml.safe_load(f)

def clean_name(name):
    return name.split('\n')[0]

def get_input_list(inputs):
    input_list = []
    for i in inputs:
        if type(i) == list:
            for j in i:
                vals = j.get('value')
                for v in vals:
                    input_list.append(clean_name(j.get('name')))
        else:
            vals = i.get('value')
            for v in vals:
                input_list.append(clean_name(v.get('name')))
    return input_list

def get_graph_data(input):
    with open(input, 'r') as f:
        return json.load(f)
    
def process_node(n, node_idx, connection_dict, DG, link=False):
    node_name = clean_name(n.get('name'))
    theseNodes, theseEdges = [], []
    input_names = []
    if len(n.get('inputs')) > 0:
        input_names = get_input_list(n.get('inputs'))

    outputs = n.get('outputs')
    for o in outputs:
        if len(o.get('value')) > 1:
                print('Multiple link outputs not supported yet... Only processing the first one for now...')
        connection_dict[clean_name(o.get('value')[0].get('name'))] = node_idx
    DG.add_node(node_idx, name=node_name, op=n.get('type'))

    if link:
        DG.add_edge(node_idx-1, node_idx)

    for input_name in input_names:
        if input_name in connection_dict.keys():
            if connection_dict[input_name] != node_idx:
                DG.add_edge(connection_dict[input_name], node_idx)

def build_graph(graph_data, debug=False):
    DG = nx.DiGraph()
    nodes = graph_data.get('nodes')

    node_idx = 0
    connection_dict = dict()
    for n in nodes:
        process_node(n, node_idx, connection_dict, DG)
        node_idx += 1 

        for link in n.get('chain'):
            process_node(link, node_idx, connection_dict, DG, link=True)
            node_idx += 1 
        
        if debug:
            pos = nx.spectral_layout(DG, scale=0.5)
            nx.draw(DG, pos,with_labels=True)
            plt.savefig(f"node{node_idx}.png")
            plt.clf()
    
    if debug:
        pos = nx.spectral_layout(DG, scale=0.5)
        nx.draw(DG, pos,with_labels=True)
        plt.savefig(f"out.png")
        node_data = DG.nodes(data=True)
        for node, data in node_data:
            print(f"Node {node}: {data}")

    return DG

def main():
    cfg = load_config(args.config)
    graph_data = get_graph_data(cfg.get('input_graph_json'))
    graph = build_graph(graph_data, debug=cfg.get('debug_graph', False))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=False, help="Configuration YAML file", dest="config", default="cfg.yaml")
    args = parser.parse_args()
    main()
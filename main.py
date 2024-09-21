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

def main():
    cfg = load_config(args.config)
    with open(cfg["input"], 'r') as f:
        data = json.load(f)
    DG = nx.DiGraph()
    nodes = data.get('nodes')

    node_idx = 0
    connection_dict = dict()
    for n in nodes:
        node_name = clean_name(n.get('name'))
        input_names = []
        if len(n.get('inputs')) > 0:
            input_names = get_input_list(n.get('inputs'))

        outputs = n.get('outputs')
        for o in outputs:
            if len(o.get('value')) > 1:
                    print('Multiple link outputs not supported yet... Only processing the first one for now...')
            connection_dict[clean_name(o.get('value')[0].get('name'))] = node_idx
        DG.add_node(node_idx, name=node_name, op=n.get('type'))
        for input_name in input_names:
            if input_name in connection_dict.keys():
                if connection_dict[input_name] != node_idx:
                    DG.add_edge(connection_dict[input_name], node_idx)
        parent_idx = node_idx
        node_idx += 1 

        #chain_connection_dict = dict()
        for link in n.get('chain'):
            node_name = clean_name(link.get('name'))
            outputs = link.get('outputs')
            for o in outputs:
                if len(o.get('value')) > 1:
                    print('Multiple link outputs not supported yet... Only processing the first one for now...')
                connection_dict[clean_name(o.get('value')[0].get('name'))] = node_idx
            DG.add_node(node_idx, name=node_name, op=link.get('type'))
            if node_idx - parent_idx == 1:
                DG.add_edge(parent_idx, node_idx)
            else:
                DG.add_edge(node_idx-1, node_idx)
            if node_name in connection_dict.keys():
                if connection_dict[node_name] != node_idx:
                    DG.add_edge(connection_dict[node_name], node_idx)
            node_idx += 1 
        
        pos = nx.spectral_layout(DG, scale=0.5)
        nx.draw(DG, pos,with_labels=True)
        plt.savefig(f"node{node_idx}.png")
        plt.clf()
    
    pos = nx.spectral_layout(DG, scale=0.5)
    nx.draw(DG, pos,with_labels=True)
    plt.savefig(f"out.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Configuration YAML file", dest="config")
    args = parser.parse_args()
    main()
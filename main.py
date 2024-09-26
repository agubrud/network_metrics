import json
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import yaml
import Op

def load_config(cfg):
    with open(cfg, 'r') as f:
        return yaml.safe_load(f)

def clean_name(name):
    return name.split('\n')[0]

def get_name_list(inputs):
    name_list = []
    value_list = []
    for i in inputs:
        if type(i) == list:
            for j in i:
                vals = j.get('value')
                for v in vals:
                    name_list.append(clean_name(j.get('name')))
        else:
            vals = i.get('value')
            for v in vals:
                name_list.append(clean_name(v.get('name')))
                if v.get('type') is not None and 'shape' in v.get('type').keys():
                    value_list.append(next(iter(v.get('type').get('shape').values())))
    return name_list, value_list

def execute_graph(graph):
    #for node in nx.dfs_preorder_nodes(graph):
    start_node = graph.nodes[0]
    input_dims = start_node.get('op_input_dims')

    op = Op.Op(start_node.get('name'), start_node.get('op_type').get('name'), start_node.get('op_attr'), input_dims)
    print(f"{str([])}, {0}, {op.name}, {op.input_dims}, {op.op_count}, {op.output_dims}")
    start_node['op_output_dims'] = op.output_dims
    traversed_nodes = [0]
    for src, node in nx.bfs_edges(graph, 0):
        #print(src,node)
        if node in traversed_nodes:
            continue
        cur_node = graph.nodes[node]
        input_dims = cur_node.get('op_input_dims')
        
        # grabs only the source of the incoming edge
        #cur_node['wt_size'] = op.instance.wt_size
        #print(cur_node['wt_size'])
  
        input_nodes = [u for u, v in graph.edges if v == node]
        if len(input_nodes) > 1:
            # catchup at branch join
            # find all nodes that haven't yet been traversed that come into the current junction, process them
            for i in input_nodes:
                if i not in traversed_nodes:
                    tmp = i
                    catchup_node = i
                    untraversed_stack = [i]
                    while tmp not in traversed_nodes:
                        untraversed_input_nodes = [u for u, v in graph.edges if v == tmp]
                        tmp = untraversed_input_nodes[0]
                        untraversed_stack.append(tmp)
                    untraversed_stack.pop()
                    while len(untraversed_stack) > 0 :
                        cur_catchup_node = untraversed_stack.pop()
                        catchup_cur_node = graph.nodes[cur_catchup_node]
                        catchup_input_nodes = [u for u, v in graph.edges if v == cur_catchup_node]
                        for cn in catchup_input_nodes:
                            prev_node = graph.nodes[cn]

                            # TODO: figure out how to handle ops that come together from a branch where the other path has not been evaluated
                            c_input_dims = prev_node.get('op_output_dims')

                        op = Op.Op(catchup_cur_node.get('name'), catchup_cur_node.get('op_type').get('name'), catchup_cur_node.get('op_attr'), c_input_dims)
                        print(f"{str(catchup_input_nodes)}, {cur_catchup_node}, {op.name}, {op.input_dims}, {op.op_count}, {op.output_dims}")
                        catchup_cur_node['op_output_dims'] = op.output_dims
                        traversed_nodes.append(cur_catchup_node)


        for n in input_nodes:
            prev_node = graph.nodes[n]

            # TODO: figure out how to handle ops that come together from a branch where the other path has not been evaluated
            input_dims = prev_node.get('op_output_dims', input_dims)

        op = Op.Op(cur_node.get('name'), cur_node.get('op_type').get('name'), cur_node.get('op_attr'), input_dims)
        print(f"{str(input_nodes)}, {node}, {op.name}, {op.input_dims}, {op.op_count}, {op.output_dims}")
        cur_node['op_output_dims'] = op.output_dims
        traversed_nodes.append(node)

def get_graph_data(input):
    with open(input, 'r') as f:
        return json.load(f)
    
def process_node(n, node_idx, connection_dict, DG, link=False):
    node_name = clean_name(n.get('name'))
    input_names = []
    if len(n.get('inputs')) > 0:
        input_names, input_vals = get_name_list(n.get('inputs'))

    if len(n.get('outputs')) > 0:
        output_names, _ = get_name_list(n.get('outputs'))
        for o in output_names:
            connection_dict[clean_name(o)] = node_idx
    DG.add_node(node_idx, name=node_name, op_attr=n.get('attributes'), op_type=n.get('type'), op_input_dims=input_vals)

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
        nx.write_gexf(DG, "graph.gexf")

    return DG

def main():
    cfg = load_config(args.config)
    graph_data = get_graph_data(cfg.get('input_graph_json'))
    graph = build_graph(graph_data, debug=cfg.get('debug_graph', False))
    execute_graph(graph)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=False, help="Configuration YAML file", dest="config", default="cfg.yaml")
    args = parser.parse_args()
    main()
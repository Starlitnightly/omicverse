# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:remove_cycle.py
# @Software:PyCharm
# @Created Time:2024/1/23 5:57 PM
import dgl
import networkx as nx

def remove_cycle(graph):
    nx_g = dgl.to_networkx(graph)
    while True:
        try:
            cycle = nx.find_cycle(nx_g, orientation='original')
            print(f'Detect cycle: {cycle}')
            for u, v, _, _ in cycle:
                # Remove edge from the DGL graph
                graph.remove_edges(graph.edge_ids(u, v))
            nx_g = dgl.to_networkx(graph)
        except:
            print('no cycle detected')
            break
    return graph


if __name__=='__main__':
    graph=dgl.load_graphs(r'kb_acyclic_reg_cxg.dgl')[0][0]
    regulate_subg = dgl.edge_type_subgraph(graph, etypes=['regulate'])
    acyclic_regulate_graph=remove_cycle(regulate_subg)
    dgl.save_graphs(r'kb_acyclic_reg_cxg.dgl',acyclic_regulate_graph)


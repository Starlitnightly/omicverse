import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from .model.aug import random_aug
from .utils import coords2adjacentmat
from timeit import default_timer as timer
from collections import OrderedDict
from tqdm import trange

def train_seq(graphs, args, dump_epoch_list, out_prefix, model):
    """The CAST MARK training function

    Args:
        graphs (List[Tuple(str, dgl.Graph, torch.Tensor)]): List of 3-member tuples, each tuple represents one tissue sample, containing sample name, a DGL graph object, and a feature matrix in the torch.Tensor format
        args (model_GCNII.Args): the Args object contains training parameters
        dump_epoch_list (List): A list of epoch id you hope training snapshots to be dumped, for debug use, empty by default
        out_prefix (str): file name prefix for the snapshot files
        model (model_GCNII.CCA_SSG): the GNN model

    Returns:
        Tuple(Dict, List, CCA_SSG): returns a 3-member tuple, a dictionary containing the graph embeddings for each sample, a list of every loss value, and the trained model object
    """    
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)

    loss_log = []
    time_now = timer()
    
    t = trange(args.epochs, desc='', leave=True)
    for epoch in t:

        with torch.no_grad():
            if epoch in dump_epoch_list:
                model.eval()
                dump_embedding = OrderedDict()
                for name, graph, feat in graphs:
                    # graph = graph.to(args.device)
                    # feat = feat.to(args.device)
                    dump_embedding[name] = model.get_embedding(graph, feat)
                torch.save(dump_embedding, f'{out_prefix}_embed_dict_epoch{epoch}.pt')
                torch.save(loss_log, f'{out_prefix}_loss_log_epoch{epoch}.pt')
                print(f"Successfully dumped epoch {epoch}")

        losses = dict()
        model.train()
        optimizer.zero_grad()
        # print(f'Epoch: {epoch}')

        for name_, graph_, feat_ in graphs:
            with torch.no_grad():
                N = graph_.number_of_nodes()
                graph1, feat1 = random_aug(graph_, feat_, args.dfr, args.der)
                graph2, feat2 = random_aug(graph_, feat_, args.dfr, args.der)

                graph1 = graph1.add_self_loop()
                graph2 = graph2.add_self_loop()

            z1, z2 = model(graph1, feat1, graph2, feat2)

            c = torch.mm(z1.T, z2)
            c1 = torch.mm(z1.T, z1)
            c2 = torch.mm(z2.T, z2)

            c = c / N
            c1 = c1 / N
            c2 = c2 / N

            loss_inv = - torch.diagonal(c).sum()
            iden = torch.eye(c.size(0), device=args.device)
            loss_dec1 = (iden - c1).pow(2).sum()
            loss_dec2 = (iden - c2).pow(2).sum()
            loss = loss_inv + args.lambd * (loss_dec1 + loss_dec2)
            loss.backward()
            optimizer.step()
            
        # del graph1, feat1, graph2, feat2        
        loss_log.append(loss.item())
        time_step = timer() - time_now
        time_now += time_step
        # print(f'Loss: {loss.item()} step time={time_step:.3f}s')
        t.set_description(f'Loss: {loss.item():.3f} step time={time_step:.3f}s')
        t.refresh()
    
    model.eval()
    with torch.no_grad():
        dump_embedding = OrderedDict()
        for name, graph, feat in graphs:
            dump_embedding[name] = model.get_embedding(graph, feat)
    return dump_embedding, loss_log, model

# graph construction tools
def delaunay_dgl(sample_name, df, output_path,if_plot=True,strategy_t = 'convex'):
    coords = np.column_stack((np.array(df)[:,0],np.array(df)[:,1]))
    delaunay_graph = coords2adjacentmat(coords,output_mode = 'raw',strategy_t = strategy_t)
    if if_plot:
        positions = dict(zip(delaunay_graph.nodes, coords[delaunay_graph.nodes,:]))
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        nx.draw(
            delaunay_graph,
            positions,
            ax=ax,
            node_size=1,
            node_color="#000000",
            edge_color="#5A98AF",
            alpha=0.6,
        )
        plt.axis('equal')
        plt.savefig(f'{output_path}/delaunay_{sample_name}.png')
    import dgl
    return dgl.from_networkx(delaunay_graph)
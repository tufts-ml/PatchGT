import torch
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch
from torch_geometric.nn import avg_pool_x
from torch_scatter import scatter
from torch_geometric.data import Data as gData
import pdb
import networkx as nx
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
from scipy.sparse import csgraph


def eigencluster_connect(dataset, tresh, device, normalized = True, patch_self_loop = False, autok = False, max_cluster = 6,
                         fixed_k = False, constant_k = 3):
    def find_k(vals, max):
        if len(vals) > 1:
            m = min(max, len(vals))
            if m == len(vals):
                m = m-1

            s = []
            for i in range(m):
                s.append(vals[i + 1] - vals[i])
            s = np.array(s)

            k = np.argmax(s)


            return k + 1
        else:
            print('auto assigned as 1')
            return 1

    def index_adj(edge_index, num_nodes):
        _, n = edge_index.shape
        adj = np.zeros((num_nodes, num_nodes))
        glist = []

        for i in range(n):
            s = edge_index[0, i]
            t = edge_index[1, i]
            adj[s, t] = 1
            adj[t, s] = 1
            glist.append((s, t))

        return adj, glist
    def is_connected(data):
        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)

        num_components, component = sp.csgraph.connected_components(adj)
        if num_components ==1:
            return True
        else:
            return False

    if not autok and not fixed_k:
        print('start clusterting......, the tresh hold ratio is {}'.format(tresh))

    elif not fixed_k and autok:
        print('start clusterting......, the autok maximum cluster is {}'.format(max_cluster))
    else:
        print('start clusterting......, the constant number of cluster is {}'.format(constant_k))

    color_list=[]
    kl=[]
    center_list = []
    sender_list = []
    receiver_list = []
    #group_number_list = []
    l = len(dataset)
    max_length = 1
    for i in tqdm(range(l)):
        g=dataset[i]
        edge_index = g.edge_index
        num_nodes = g.num_nodes
        edge_index = edge_index.numpy()

        A, elist = index_adj(edge_index, num_nodes)


        # graph laplacian
        if not normalized:
            # diagonal matrix
            D = np.diag(A.sum(axis=1))
            L = D - A
        else:
            L = csgraph.laplacian(A, normed=True)


        # eigenvalues and eigenvectors
        vals, vecs = np.linalg.eig(L)
        #only keep real part
        vals = vals.real
        vecs = vecs.real
        # if len(vals) < 3:
        #     continue


        vecs = vecs[:, np.argsort(vals)]
        vals = vals[np.argsort(vals)]
        svals = np.sum(vals)

        if not autok and not fixed_k:
            kk = 1
            for k in range(1,num_nodes):

                r= vals[k]
                if r > tresh:
                    kk =k
                    break

        elif not fixed_k and autok:
            kk = find_k(vals, max_cluster)

        else:
            kk = min(constant_k, len(vals))


        #check max_Length
        if kk>max_length:
            max_length = kk


        kmeans = KMeans(n_clusters=kk)
        try:
            kmeans.fit(vecs[:, 1:kk])
        except:
            kmeans.fit(vecs[:, :kk])

        colors = kmeans.labels_



        #only take first 3 dim of centers
        centers = kmeans.cluster_centers_[:,:3] #should be 1:3
        if len(vals) < 3:  #2

            dist = 3 - len(vals) #2
            centers = np.concatenate((centers, np.zeros((kk,dist))),axis=1)

        #senders and receivers for subgraph
        senders =[]
        receivers =[]
        if kk > 1:
            for i in range(kk-1):
                indices_s = np.where(colors == i)[0]
                for j in range(i+1,kk):
                    indices_r = np.where(colors == j)[0]
                    indices_all = torch.tensor(np.concatenate((indices_s,indices_r)), dtype=torch.int64)
                    sg = g.subgraph(indices_all)
                    if is_connected(sg):
                        senders.append(i)
                        senders.append(j)
                        receivers.append(j)
                        receivers.append(i)
                        #add self_loop
                    if patch_self_loop:
                        senders.append(i)
                        senders.append(j)
                        receivers.append(i)
                        receivers.append(j)
        else:
            senders.append(0)
            receivers.append(0)







        color_list.append(torch.tensor(colors,dtype=torch.int64).to(device))
        kl.append(kk)
        center_list.append(torch.tensor(centers,dtype=torch.float32).to(device))
        sender_list.append(senders)
        receiver_list.append(receivers)

    return color_list, kl, center_list, max_length, sender_list, receiver_list


def group_number_count(color_list,kl, device):
    group_number_list = []
    for i in range(len(color_list)):
        color = color_list[i]
        kk=kl[i]
        colors = color.cpu().numpy()
        count_arr = np.bincount(colors)
        if max(colors) == kk-1:
            group_count = [count_arr[i] for i in range(kk)]
        else:
            dis = kk-1-max(colors)
            group_count = [count_arr[i] for i in range(max(colors))]
            for j in range(dis):
                group_count.append(0)
            print('make up for graph {}'.format(i))
        group_number_list.append(torch.tensor(np.array(group_count), device=device))
    return group_number_list






from torch_geometric.data import DataLoader
import numpy as np
from train_TU import train
from args import Args
from torch_geometric.datasets import TUDataset
from model.patch_model import patchGT_TU
from cluster import eigencluster_connect, group_number_count


if __name__ == '__main__':

    args = Args()
    args = args.update_args()

    dataset = TUDataset(root = 'dataset/', name=args.dataset, use_node_attr=True, use_edge_attr=True)
    args.input_embd = dataset.num_node_features
    args.num_tasks = dataset.num_classes
    args.num_node_features = dataset.num_node_features
    args.num_edge_features = dataset.num_edge_features

    idx = np.arange(len(dataset))
    np.random.shuffle(idx)
    l=int(len(dataset)*0.9)
    train_idx = idx[:l]
    test_idx = idx[l:]


    traingraphs = [g for g in dataset[train_idx]]

    testgraphs = [g for g in dataset[test_idx]]

    dataloader_train = DataLoader(traingraphs, batch_size=args.batch_size, shuffle=False, drop_last=True)

    dataloader_test = DataLoader(testgraphs, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    if args.optimizer == 'auroc':
        args.imratio = float((dataloader_train.dataset.data.y.sum() / dataloader_train.dataset.data.y.shape[0]).numpy())


    #preprocess for spectral cluster for training graphs
    print('process dataset...')
    color_list, color_number, center_list, max_length, sender_list, receiver_list = eigencluster_connect(traingraphs,
                                                                                                         args.cluster_bar,
                                                                                                         args.device,
                                                                                                         args.normalized,
                                                                                                         args.self_loop_patch,
                                                                                                         args.autok,
                                                                                                         args.max_cluster,
                                                                                                         args.fixed_k,
                                                                                                         args.constant_k)
    test_color_list, test_color_number, test_center_list, test_max_length, test_sender_list, test_receiver_list = \
        eigencluster_connect(testgraphs, args.cluster_bar, args.device, args.normalized, args.self_loop_patch,
                             args.autok, args.max_cluster,
                             args.fixed_k, args.constant_k)
    print("counting group number...")
    group_number = group_number_count(color_list, color_number, args.device)
    test_group_number = group_number_count(test_color_list, test_color_number, args.device)
    print("finish counting group number!")
    #create models
    model = patchGT_TU(args, num_tasks=args.num_tasks, num_layer=args.gcn_num_layers,
                                                   emb_dim=args.n_embd,
                                                   gnn_type=args.gnn_type, coarse_gnn_type=args.coarse_gnn_type,
                                                   virtual_node=args.virtual_node, residual=args.residual,
                                                   drop_ratio=args.feature_drop, att_drop=args.attn_pdrop, JK="last",
                                                   graph_pooling="mean", cls_token=args.cls_token,
                                                   num_heads=args.n_head, attention_layers=args.n_layer,
                                                   patch_gcn_num_layers=args.patch_gcn_num_layers, in_edge_channels=4,
                                                   patch_pooling=args.coarse_pooling,
                                                   device=args.device, node_dim = args.num_node_features)


    model.to(args.device)
    args.model = model.__class__.__name__

    evaluator = None

    train(args, traingraphs, testgraphs, args.batch_size, args.test_batch_size, color_list, center_list, color_number,
          model, test_color_list, test_color_number, test_center_list, test_max_length, max_length=max_length,
          x_dim=args.n_embd, evaluator=evaluator, sender_list=sender_list,
          receiver_list=receiver_list, test_sender_list=test_sender_list, test_receiver_list=test_receiver_list,
          group_number=group_number, test_group_number=test_group_number)
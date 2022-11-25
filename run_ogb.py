from ogb.graphproppred import PygGraphPropPredDataset
from train_ogb import train
from args import Args
from cluster import eigencluster_connect, group_number_count
from utils import create_models, save_processed_dataset, load_processed_dataset
from ogb.graphproppred import Evaluator
from model.patch_model import patchGT



if __name__ == '__main__':

    args = Args()
    args = args.update_args()

    dataset = PygGraphPropPredDataset(name = args.dataset, root = 'dataset/')
    args.num_node_features = dataset.num_node_features
    args.num_tasks = dataset.num_tasks
    args.eval_metric = dataset.eval_metric


    split_idx = dataset.get_idx_split()


    traingraphs = dataset[split_idx["train"]]
    validationgraphs = dataset[split_idx["valid"]]
    testgraphs = dataset[split_idx["test"]]

    args.num_classes = dataset.num_classes
    evaluator = Evaluator(args.dataset)


    if args.optimizer == 'auroc':
        args.imratio = 0.04





    if  not args.load_processed_dataset:
        print('process dataset...')
        color_list, color_number, center_list, max_length, sender_list, receiver_list = eigencluster_connect(traingraphs,
                args.cluster_bar, args.device, args.normalized, args.self_loop_patch, args.autok,args.max_cluster,
                args.fixed_k, args.constant_k)
        test_color_list, test_color_number, test_center_list, test_max_length, test_sender_list, test_receiver_list =\
            eigencluster_connect(testgraphs, args.cluster_bar, args.device, args.normalized,args.self_loop_patch,args.autok,args.max_cluster,
                             args.fixed_k, args.constant_k)


    else:
        print("load processed dataset directly!")
        color_list, color_number, center_list, max_length, sender_list, receiver_list, \
        test_color_list, test_color_number, test_center_list, test_max_length, test_sender_list, test_receiver_list= load_processed_dataset(args)
        test_color_list, test_color_number, test_center_list, test_max_length, test_sender_list, test_receiver_list = \
            eigencluster_connect(testgraphs, args.cluster_bar, args.device, args.normalized, args.self_loop_patch,
                                 args.autok, args.max_cluster,
                                 args.fixed_k, args.constant_k)



    if args.save_processed_dataset and  not args.load_processed_dataset:
        print('save processed dataset...')
        save_processed_dataset(args, color_list, color_number, center_list, max_length, sender_list, receiver_list,
                               test_color_list, test_color_number, test_max_length, test_sender_list, test_receiver_list)
    print("counting group number...")
    group_number = group_number_count(color_list, color_number, args.device)
    test_group_number = group_number_count(test_color_list, test_color_number,args.device)
    print("finish counting group number!")








    model = patchGT(args, num_tasks=args.num_tasks, num_layer=args.gcn_num_layers, emb_dim=args.n_embd,
                      gnn_type=args.gnn_type, coarse_gnn_type = args.coarse_gnn_type, virtual_node=args.virtual_node, residual=args.residual, drop_ratio=args.feature_drop,att_drop=args.attn_pdrop, JK="last",
                      graph_pooling="mean", cls_token=args.cls_token,
                      num_heads=args.n_head, attention_layers=args.n_layer, patch_gcn_num_layers=args.patch_gcn_num_layers, in_edge_channels=4, patch_pooling=args.coarse_pooling,
                      device=args.device, )

    model.to(args.device)
    args.model = model.__class__.__name__


    train(args, traingraphs, testgraphs, args.batch_size, args.test_batch_size, color_list, center_list, color_number, model, test_color_list, test_color_number, test_center_list, test_max_length,max_length = max_length, x_dim = args.n_embd, evaluator=evaluator, sender_list=sender_list,
          receiver_list=receiver_list, test_sender_list= test_sender_list, test_receiver_list=test_receiver_list, group_number = group_number, test_group_number= test_group_number)
from datetime import datetime
import torch
import argparse
import os

class Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # logging & saving
        self.parser.add_argument('--save_model', default=True, action='store_true', help='Whether to save model')
        self.parser.add_argument('--print_interval', type=int, default=1, help='loss printing batch interval')
        self.parser.add_argument('--epochs_save', type=int, default=100, help='model save epoch interval')
        self.parser.add_argument('--epochs_eval', type=int, default=1, help='model validate epoch interval')




        # setup
        self.parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu',
                                 help='cuda:[d] | cpu')

        self.parser.add_argument('--seed', type=int, default=314, help='random seed to reproduce performance/dataset')

        # dataset
        self.parser.add_argument('--dataset', default="DD", help='Select datase -- ogbg-molhiv | ogbg-molpcba| ogbg-ppa | ogbg-molbace|ogbg-molbbbp | PROTEINS| DD |MUTAG|PTC_MR|COLLAB|Mutagenicity|ENZYMES')
        self.parser.add_argument('--batch_size', type=int, default=64, help='batch size') #DD 6
        self.parser.add_argument('--test_batch_size', type=int, default=16, help='test batch size')
        self.parser.add_argument('--load_processed_dataset', default=False, action='store_true',
                                 help='whether to load dataset')
        self.parser.add_argument('--save_processed_dataset', default=False, action='store_true',
                                 help='whether to load dataset')






        #eigen cluster:
        self.parser.add_argument('--cluster_bar', type=float, default=0.2, help='cluster eigen value threshhold')

        self.parser.add_argument('--autok', default=False, action='store_true', help='whether to use autok, make bar useless')
        self.parser.add_argument('--max_cluster', type=int, default=6, help='only applied for autok')

        self.parser.add_argument('--fixed_k', default=False, action='store_true',
                                 help='whether to use a fixed k for clustering')
        self.parser.add_argument('--constant_k', type=int, default=3, help='only used for autok')


        self.parser.add_argument('--coarse_pooling', default='mean',type=str,
                                 help='coarse graph pooling: mean | max | sum| min')
        #self.parser.add_argument('--split_gnn', default=False, action='store_true', help='whether to run split gnn')
        self.parser.add_argument('--normalized', default=True, action='store_true', help='whether to use normalized laplacian matrix')








        # specific to GNN
        self.parser.add_argument('--gcn_num_layers', type=int, default=4,help='hidden layers of MGN, must greater than 1')
        self.parser.add_argument('--patch_gcn_num_layers', type=int, default=4, help='hidden layers of MGN')
        self.parser.add_argument('--input_embd', type=int, default=4, help='input dimension of MGN')
        self.parser.add_argument('--virtual_node', default=False, help='use virtual_node or not in the first GNN')
        self.parser.add_argument('--residual', default=True, help='use residual or not in the first GNN')
        self.parser.add_argument('--gnn_type', default='gin',
                                 help='graph type: gin | gcn  | deepergcn',type=str)
        self.parser.add_argument('--coarse_gnn_type', default='gcn',type=str,
                                 help='coarse graph type: gin | gcn')
        self.parser.add_argument('--feature_drop', type=float, default=0.001, help='last feature drop out rate')
        self.parser.add_argument('--self_loop_patch', default=False, action='store_true', help='whether to add self-loop for patch graph')





        # specific to Transformer
        self.parser.add_argument('--embd_pdrops', type=float, default=0.2, help='dropout rate')
        self.parser.add_argument('--position', default=False, action='store_true',
                                 help='Whether to use postion')
        self.parser.add_argument('--n_ctx', type=int, default=19, help='?')
        self.parser.add_argument('--n_embd', type=int, default=512, help='node embedding dimension')
        self.parser.add_argument('--layer_norm_epsilon', type=float, default=1e-5, help='cd gralayer_norm_epsilon')
        self.parser.add_argument('--n_layer', type=int, default=4, help='layer of attention')
        self.parser.add_argument('--withz', default=False, help='use z or not')
        self.parser.add_argument('--embd_pdrop', type=float, default=0.0, help='embd drop out rate')
        self.parser.add_argument('--n_head', type=int, default=32, help='transformer head')
        self.parser.add_argument('--attn_pdrop', type=float, default=0.1, help='attention drop rate')  #important
        self.parser.add_argument('--resid_pdrop', type=float, default=0.0, help='residual drop rate')
        self.parser.add_argument('--activation_function', type=str, default="relu",
                                 help='activation type in transformer')
        self.parser.add_argument('--initializer_range', type=float, default=0.02,
                                 help='activation type in transformer')
        self.parser.add_argument('--cls_token', default=True, action='store_true',
                                 help='Whether to use cls_token, if not, use avg of x output')

        #training process
        self.parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
        self.parser.add_argument('--prefetch_factor', type=int, default=2, help='prefetch_factor')
        self.parser.add_argument('--epochs', type=int, default=150, help='training epochs')
        self.parser.add_argument('--optimizer', default='adam',
                                 help='Select optimizer -- auroc | adam')
        self.parser.add_argument('--use_schedule', default=False, action='store_true', help='whether to use schedule')
        self.parser.add_argument('--max_lr', type=float, default=0.001, help='learning rate')
        self.parser.add_argument('--schedule_mode', default='triangular2',
                                 help='Select learnign rate schedule -- triangular2 | exp_range')
        self.parser.add_argument('--FLAG', default=False, action='store_true',
                                 help='whether to use FLAG Data Augmentation, support deepergcn only for current version')
        #for flag training
        self.parser.add_argument('--step-size', type=float, default=1e-2)
        self.parser.add_argument('-m', type=int, default=3)



        # Model load parameters
        self.parser.add_argument('--load_model', default=False, action='store_true', help='whether to load model')
        self.parser.add_argument('--load_model_path', default='output/GRAN_Lung_unif_nobfs_2021_01_24_23_55_12/',
                                 help='load model path')
        self.parser.add_argument('--load_device', default='cuda:0' if torch.cuda.is_available() else 'cpu',
                                 help='load device: cuda:[d] | cpu')
        self.parser.add_argument('--epochs_end', type=int, default=100, help='model in which epoch to load')

    def update_args(self):
        args = self.parser.parse_args()
        args.time = '{0:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())

        # output dataset
        args.dir_output = 'output/'
        args.fname = args.gnn_type + '_' +args.coarse_gnn_type + '_'+ args.dataset + args.time
        args.experiment_path = args.dir_output + args.fname
        args.model_save_path = args.experiment_path + '/' + 'model_save/'
        args.logging_path = args.experiment_path + '/' + 'logging/'

        args.dataset_dir = 'data/' + args.dataset
        args.current_model_save_path = args.model_save_path
        args.logging_epoch_path = args.logging_path + 'epoch_history.csv'

        if not os.path.isdir(args.logging_path):
            os.makedirs(args.logging_path)

        return args


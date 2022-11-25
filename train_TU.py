from torch_geometric.data import DataLoader
import torch.nn.functional as F
import torch
from torch import optim
import pandas as pd
import numpy as np
from collections import defaultdict
from losses import AUCMLoss, PESG
import json
import matplotlib.pyplot as plt
import os


def train_batch(args,batch_g, color_list, center_list, color_number, model,


                max_length, x_dim, optimizer, sender_list, receiver_list, group_number, loss_fn):
    '''

    :param batchg:
    :param label_list:
    :return:
    '''
    g = batch_g.to(args.device)
    batch_size = args.batch_size




    prediction = model(batch_g, color_list, center_list, color_number, batch_size, sender_list, receiver_list, group_number)
    true_y = g.y

    is_labeled = true_y == true_y

    if args.optimizer == 'adam':
        loss = loss_fn(prediction,true_y.view(-1))

    for _, opt in optimizer.items():
        opt.zero_grad()
    loss.backward()
    for _, opt in optimizer.items():
        opt.step()


    return loss.detach()

def train_batch_binary_FLAG(args,batch_g, color_list, center_list, color_number, model,


                max_length, x_dim, optimizer, sender_list, receiver_list,loss_fn):
    '''

    :param batchg:
    :param label_list:
    :return:
    '''
    g = batch_g.to(args.device)
    batch_size = args.batch_size

    true_y = g.y.to(torch.float32)
    is_labeled = true_y == true_y
    forward = lambda perturb: model(batch_g, color_list, center_list, color_number, batch_size, sender_list, receiver_list, perturb).to(torch.float32)[is_labeled]

    target = true_y[is_labeled]
    perturb_shape = (g.x.shape[0], args.n_embd)
    model.train()
    for _, opt in optimizer.items():
        opt.zero_grad()
    perturb = torch.FloatTensor(*perturb_shape).uniform_(-args.step_size, args.step_size).to(args.device)
    perturb.requires_grad_()
    prediction = forward(perturb)





    if args.optimizer == 'adam':
        loss = loss_fn(prediction,target)
    if args.optimizer == 'auroc':
        loss = loss_fn(prediction.to(torch.float32).reshape(-1, 1), target.to(torch.float32).reshape(-1, 1))
    loss /= args.m
    for _ in range(args.m - 1):
        loss.backward()
        perturb_data = perturb.detach() + args.step_size * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0

        prediction = forward(perturb)
        if args.optimizer == 'adam':
            loss = loss_fn(prediction, target)
        if args.optimizer == 'auroc':
            loss = loss_fn(prediction.to(torch.float32).reshape(-1, 1), target.to(torch.float32).reshape(-1, 1))
        loss /= args.m

    loss.backward()
    for _, opt in optimizer.items():
        opt.step()


    return loss.detach()


def train_epoch(args, epoch, dataloader_train, batch_size, color_list, center_list, color_number, model,max_length, x_dim, optimizer, sender_list, receiver_list, group_number, loss_fn):

        model.train()



        total_loss =0.0
        batch_count = len(dataloader_train)
        for batch_id, batch_g in enumerate(dataloader_train):
            #st = time.time()
            batch_color_list = color_list[batch_id*batch_size : batch_id*batch_size+batch_size]
            batch_center_list = center_list[batch_id*batch_size : batch_id*batch_size+batch_size]
            batch_color_number = color_number[batch_id*batch_size : batch_id*batch_size+batch_size]
            batch_sender_list = sender_list[batch_id*batch_size : batch_id*batch_size+batch_size]
            batch_receiver_list = receiver_list[batch_id*batch_size : batch_id*batch_size+batch_size]
            batch_group_number = group_number[batch_id*batch_size : batch_id*batch_size+batch_size]


            batch_loss = train_batch(args, batch_g, batch_color_list, batch_center_list, batch_color_number, model,max_length, x_dim, optimizer, batch_sender_list, batch_receiver_list, batch_group_number,loss_fn)



            total_loss = total_loss + batch_loss
            #spent = time.time() - st
            # if batch_id % args.print_interval == 0:
            #     print('epoch {} batch {}: loss is {}, time spent is {}.'.format(epoch, batch_id, batch_loss, spent), flush=True)


        return total_loss/batch_count

@torch.no_grad()
def test(args, epoch, dataloader_test, test_batch_size, test_color_list, test_center_list, test_color_number, model, test_max_length,x_dim, evaluator, test_sender_list, test_receiver_list, test_group_number):
        model.eval()

        acc_sum = 0.0
        rocauc_sum =0.0
        batch_numer = 0.0
        y_pred_record = []
        y_true_record = []
        correct = 0
        for batch_id, batch_g in enumerate(dataloader_test):
            batch_color_list = test_color_list[batch_id * test_batch_size: batch_id * test_batch_size + test_batch_size]
            batch_center_list = test_center_list[batch_id * test_batch_size: batch_id * test_batch_size + test_batch_size]
            batch_color_number = test_color_number[batch_id * test_batch_size: batch_id * test_batch_size + test_batch_size]
            batch_sender_list = test_sender_list[batch_id * test_batch_size: batch_id * test_batch_size + test_batch_size]
            batch_receiver_list = test_receiver_list[batch_id * test_batch_size: batch_id * test_batch_size + test_batch_size]
            batch_test_group_number = test_group_number[batch_id * test_batch_size: batch_id * test_batch_size + test_batch_size]
            #-----------------------------------------------------------------------------------------test batch
            g = batch_g.to(args.device)
            test_batch_size = args.test_batch_size



            pred = model(batch_g, batch_color_list ,batch_center_list,batch_color_number, test_batch_size, batch_sender_list, batch_receiver_list, batch_test_group_number)  # (batch_size, 1) for binary classification and (batch_size, num_class) for others
            pred = pred.max(1)[1]


            y_true = g.y.view(-1)
            correct = correct + pred.eq(y_true).sum().item()

        acc = correct / len(dataloader_test.dataset)

        print('Epoch: {}/{}, test acc: {:.6f}'.format(epoch, args.epochs, acc))
        return acc


def train(args, traingraphs, testgraphs, batch_size, test_batch_size,color_list, center_list, color_number, model,test_color_list, test_color_number,\
          test_center_list, test_max_length, max_length, x_dim, evaluator,sender_list, receiver_list,test_sender_list, test_receiver_list, group_number, test_group_number):

    #create optimizer
    optimizer = {}

    with open(os.path.join(args.logging_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)



    if args.optimizer == 'adam':
        optimizer['model'] = optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = F.nll_loss
        if args.use_schedule:
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer['model'], base_lr=args.lr, max_lr=args.max_lr, step_size_up=15, mode = args.schedule_mode, cycle_momentum=False)


    if args.optimizer == 'auroc':
        loss_fn = AUCMLoss(imratio=args.imratio, device=args.device) #should be the same in the training batch loss!
        optimizer['model'] = PESG(model = model, a=loss_fn.a, b=loss_fn.b,alpha=loss_fn.alpha,lr=args.lr, device=args.device)




    log_history = defaultdict(list)

    #training process
    epoch = 0
    losses = []
    best_acc =0
    while epoch < args.epochs:
        # shuffle
        if args.optimizer == 'auroc':
            if epoch ==10 or epoch == 20:
                optimizer['model'].update_regularizer(decay_factor=10)

        train_idx = np.arange(len(traingraphs))
        np.random.shuffle(train_idx)

        dataloader_train = DataLoader([traingraphs[i] for i in train_idx], batch_size=args.batch_size, shuffle=False,
                                      drop_last=True)

        dataloader_test = DataLoader(testgraphs, batch_size=args.test_batch_size, shuffle=False, drop_last=True)

        loss = train_epoch(args, epoch, dataloader_train, batch_size,[color_list[i] for i in train_idx], [center_list[i] for i in train_idx], [color_number[i] for i in train_idx], model ,
                max_length, x_dim, optimizer, [sender_list[i] for i in train_idx], [receiver_list[i] for i in train_idx],[group_number[i] for i in train_idx], loss_fn)

        losses.append(loss.item())
        epoch += 1
        print('Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss.cpu().item()))
        if epoch % args.epochs_eval == 0:
            acc = test(args, epoch, dataloader_test, test_batch_size, test_color_list, test_center_list, test_color_number, model,
                 test_max_length, x_dim, evaluator, test_sender_list, test_receiver_list, test_group_number)
            if acc>best_acc:
                best_acc = acc
            log_history['test_acc'].append(acc)
            #log_history['test_{}'.format(args.eval_metric)].append(rocauc)
            df_epoch = pd.DataFrame()
            df_epoch['test_acc'] = np.array(log_history['test_acc'])
            #df_epoch['test_{}'.format(args.eval_metric)] = np.array(log_history['test_{}'.format(args.eval_metric)])
            df_epoch.to_csv(args.logging_epoch_path, index=False)

            # save  plot
            fig, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.plot(losses, label='train loss', color=color)
            # plt.plot(all_val_losses, label='val')

            ax1.set_xlabel('epochs')
            ax1.set_ylabel('loss', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            color = 'tab:blue'
            ax2 = ax1.twinx()
            ax2.set_ylabel('test acc', color='tab:blue')  # we already handled the x-label with ax1
            ax2.plot(df_epoch['test_acc'], color=color, label='test acc')
            ax2.tick_params(axis='y', labelcolor=color)
            plt.legend(loc='best')
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.legend()
            plt.title(
                f'best:{best_acc:3}')
            plt.savefig(os.path.join(args.logging_path, 'loss_curve.png'))
            plt.close()


from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import optimizers
import torch
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics
import geoopt.manifolds.poincare.math as pmath
import pandas as pd
from sklearn.preprocessing import minmax_scale

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def get_geometric_hyp(args):
    if args.dataset == 'amazonphoto': 
        filename = './delta_hyp/local_hyperbolicity_list_' + str(args.dataset) +'_sg1.pkl'
    else:
        filename = './delta_hyp/local_hyperbolicity_list_' + str(args.dataset) +'_sg2.pkl'
    with open(filename, 'rb') as f:
        local_hyperbolicity_list = pickle.load(f) # list type
    local_hyp_array = np.array(local_hyperbolicity_list)
    local_hyp_array = minmax_scale(local_hyp_array)
    local_geometric_hyp = torch.from_numpy(local_hyp_array)
    print('local_geometric_hyp', local_geometric_hyp.size())
    return local_geometric_hyp


def stats_match(args, model, layer_num):
    # match mean
    ave_euc_attn_score = sum(model.encoder.layers[layer_num].att_score[:,1].view(-1))/args.num_nodes
    ave_hyp_score = sum(args.local_geometric_hyp)/args.num_nodes
    loss = (ave_euc_attn_score - ave_hyp_score) ** 2
    return loss


def one_one_match(args, model, layer_num):
    euclidean_attn_score = model.encoder.layers[layer_num].att_score[:,1].view(-1)
    assert euclidean_attn_score.size() == args.local_geometric_hyp.size()
    return torch.pow(torch.mean(torch.pow(euclidean_attn_score - args.local_geometric_hyp, 2)), 1/2)


def W(args, model, layer_num):
    u_euclidean_attn_score = model.encoder.layers[layer_num].att_score[:,1].view(-1)
    v = args.local_geometric_hyp
    # u and v are raw values
    sorted_u, _ = torch.sort(u_euclidean_attn_score)
    sorted_v, _ = torch.sort(v)
    assert sorted_u.size() == sorted_v.size()
    return torch.pow(torch.mean(torch.pow(sorted_u - sorted_v, 2)), 1/2)


def non_uniformity(args, model, layer_num):
     # non uniformity part
    whole_attn_score = model.encoder.layers[layer_num].att_score.clone() # Nx2x1
    square_whole_attn_score = torch.pow(whole_attn_score, 2)
    minimize_for_one_hot = -1*torch.sum(square_whole_attn_score)/whole_attn_score.size()[0]
    return minimize_for_one_hot


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join(os.environ['LOG_DIR'], args.task, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    # Load data
    data = load_data(args, os.path.join(os.environ['DATAPATH'], args.dataset))
    args.n_nodes, args.feat_dim = data['features'].shape

    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = LPModel

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Model and optimizer
    model = Model(args)
    logging.info(str(model))
    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)

    # Train model
    t_total = time.time()
    counter = 0
    total_num_nodes = data['features'].size()[0]
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    best_attn_score = None

    VARIANT = args.variant
    local_geometric_hyp = get_geometric_hyp(args)
    args.local_geometric_hyp = local_geometric_hyp.to(args.device)
    args.num_nodes = total_num_nodes

    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        train_metrics = model.compute_metrics(embeddings, data, 'train', args)
        
        if args.model == 'JSGNN' and VARIANT == 'PLAIN':
            overall_loss = train_metrics['loss']
            if args.dataset == 'amazonphoto':
                print('W plain ', W(args, model, 0))
            else:
                print('W plain ', W(args, model, 0) + W(args, model,1))

        elif args.model == 'JSGNN':
            need_layer_2 = need_layer_3 = False
            if args.num_layers > 2:
                need_layer_2 = True
            if args.num_layers > 3:
                need_layer_3 = True

            if VARIANT == 'MEAN':
                mean_loss = stats_match(args, model, layer_num=0)

                if need_layer_2:
                    mean_loss_new = stats_match(args, model, layer_num=1)
                    mean_loss += mean_loss_new

                if need_layer_3:
                    mean_loss_new = stats_match(args, model, layer_num=2)
                    mean_loss += mean_loss_new
                overall_loss = train_metrics['loss'] + mean_loss

            elif VARIANT == 'PAIRWISE':
                pairwise_loss = one_one_match(args, model, layer_num=0)

                if need_layer_2:
                    pairwise_loss_new = one_one_match(args, model, layer_num=1)
                    pairwise_loss += pairwise_loss_new

                if need_layer_3:
                    mean_loss_new = one_one_match(args, model, layer_num=2)
                    pairwise_loss += pairwise_loss_new
                overall_loss = train_metrics['loss'] + pairwise_loss

            elif VARIANT == 'DISTRIBUTION':
                # print('distribution')
                wasser_loss = W(args, model, layer_num=0)
                non_uniform_loss = non_uniformity(args, model, layer_num=0)
                if need_layer_2:
                    wasser_loss_new = W(args, model, layer_num=1)
                    non_uniform_loss_new = non_uniformity(args, model, layer_num=1)
                    wasser_loss += wasser_loss_new
                    non_uniform_loss += non_uniform_loss_new

                if need_layer_3:
                    mean_loss_new = W(args, model, layer_num=2)
                    non_uniform_loss_new = non_uniformity(args, model, layer_num=2)
                    wasser_loss += wasser_loss_new
                    non_uniform_loss += non_uniform_loss_new
                overall_loss = train_metrics['loss'] + args.lambda_wasser * wasser_loss + args.lambda_uniform * non_uniform_loss
                # print('wasser_loss', wasser_loss)

        else: # Other models
            overall_loss = train_metrics['loss']
        overall_loss.backward()
        # train_metrics['loss'].backward()

        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()
        lr_scheduler.step()
        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                   'lr: {}'.format(lr_scheduler.get_last_lr()[0]),
                                   format_metrics(train_metrics, 'train'),
                                   'time: {:.4f}s'.format(time.time() - t)
                                   ]))
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(data['features'], data['adj_train_norm'])
            val_metrics = model.compute_metrics(embeddings, data, 'val', args)
            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
            if model.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = model.compute_metrics(embeddings, data, 'test', args)

                # TO ANALYSE ATTENTION SCORES LEARNT 
                if args.model == 'JSGNN':
                    best_attn_score = [lyr.att_score.clone().detach().cpu() for lyr in model.encoder.layers]
                    with open('plot_data/' + args.dataset + '_' + args.variant + '_test_attn_score_seed'+ str(args.seed) +'.pkl', 'wb') as f:
                        pickle.dump(best_attn_score, f)
                if args.model == 'GIL':
                    best_attn_score = [(lyr.dist_attn_h_to_e.clone().detach().cpu(), lyr.dist_attn_e_to_h.clone().detach().cpu()) for lyr in model.encoder.layers]
                    with open(args.dataset + '_gil_dist_attention_scores.pkl', 'wb') as f:
                        pickle.dump(best_attn_score, f)

                if isinstance(embeddings, tuple):
                    best_emb = torch.cat((pmath.logmap0(embeddings[0], c=1.0), embeddings[1]), dim=1).cpu()
                else:
                    best_emb = embeddings.cpu()
                if args.save:
                    np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())

                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter == args.patience and epoch > args.min_epochs:
                    logging.info("Early stopping")
                    break

    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        best_emb = model.encode(data['features'], data['adj_train_norm'])
        best_test_metrics = model.compute_metrics(best_emb, data, 'test', args)
        
        # TO ANALYSE ATTENTION SCORES LEARNT
        # if args.model == 'JSGNN':
        #     best_attn_score = [lyr.att_score.clone().detach().cpu() for lyr in model.encoder.layers]
        #     with open(args.dataset + '_test_attn_score_split_eu_hyp.pkl', 'wb') as f:
        #         pickle.dump(best_attn_score, f)
        #     if VARIANT == 'PLAIN':
        #         with open(args.dataset + '_plain_test_attn_score.pkl', 'wb') as f:
        #             pickle.dump(best_attn_score, f)

        # if args.model == 'GIL':
        #     best_attn_score = [(lyr.dist_attn_h_to_e.clone().detach().cpu(), lyr.dist_attn_e_to_h.clone().detach().cpu()) for lyr in model.encoder.layers]
        #     with open(args.dataset + '_gil_dist_attention_scores.pkl', 'wb') as f:
        #         pickle.dump(best_attn_score, f)

    logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
    logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))
    print(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))

    if args.save:
        if isinstance(best_emb, tuple):
            best_emb = torch.cat((pmath.logmap0(best_emb[0], c=1.0), best_emb[1]), dim=1).cpu()
        else:
            best_emb = best_emb.cpu()
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
        if hasattr(model.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
            print('Dumped attention adj: ' + filename)

        json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        logging.info(f"Saved model in {save_dir}")


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    train(args)
    import sys

    sys.exit(0)

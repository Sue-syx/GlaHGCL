# -*- coding: utf-8 -*-
import os
 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from util_functions import *
from preprocessing_data import *
from train_eval import *
from model import CausalGAT, GATNet
import opts
import torch
import logging

if __name__ == '__main__':

    # Arguments
    args = opts.parse_args()
    args.model = "GloLoCon"
    args.dataset = "HMDDv2_0"
    args.lr = 0.002
    args.min_lr = 5e-6
    args.bias = 0.9
    args.use_features = True
    args.wandb_open = False
    args.causal = True

    # root_path = os.path.join('results', args.dataset, args.model, '5CV')
    root_path = os.path.join('results', 'MTI', args.model, '3/')
    os.makedirs(root_path, exist_ok=True)
    logger = logger_config(root_path + 'log.txt', logging_name='MTI')

    if args.wandb_open:
        wandb.init(project="MTI", group="GloLoCon")

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2341)
    seed = 2341
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    hop = 1

    if not args.no_train:
        # Construct model
        logger.info('training.....')
        data_combo = (args.dataset, '', '')
        u_features, v_features, net, labels, u_indices, v_indices, num_list = load_data(args.dataset)
        # print(u_features)
        
        logger.info('preprocessing end.')
        
        adj = torch.tensor(net)
        if args.use_features:
            n_features = u_features.shape[1] + v_features.shape[1]
        else:
            u_features, v_features = None, None
            n_features = 0
        all_indices = (u_indices, v_indices)
        logger.info('begin constructing all_graphs')
        all_graphs = extracting_subgraphs(net, all_indices, labels, hop, u_features, v_features, hop * 2 + 1)

        # print(all_graphs)
        # print(all_graphs.shape,"!!!!!!!!!!!!!!!!")
        # exit()

        data_size = int(len(all_graphs) / 2)
        data_pos = all_graphs[0:data_size]
        data_neg = all_graphs[data_size:]
        test_pos = sample(data_pos, int(0.1 * len(data_pos)))
        train_pos = [item for item in data_pos if item not in test_pos]
        test_neg = sample(data_neg, int(0.1 * len(data_neg)))
        train_neg = [item for item in data_neg if item not in test_neg]
        train_set = train_pos + train_neg
        test_set = test_pos + test_neg

        train_dataset = MyDataset(train_set, root='data/{}{}/{}/train'.format(*data_combo))
        test_dataset = MyDataset(test_set, root='data/{}{}/{}/test'.format(*data_combo))
        logger.info('constructing all_graphs end.')

        sum = 0
        all_results = []
        max_f1 = 0
        num_features = u_features.shape[1] + 4
        _, test_graphs = get_k_fold_data(1, 0, test_dataset)
        test_loader = DataLoader(test_graphs, 1, shuffle=False, num_workers=0)
        for count in range(1):
            logger.info('########', count, ' training.' + '#########')
            if args.causal:
                model = CausalGAT(num_features=num_features, num_classes=2, args=args, head=8)
                # print(model)
                # exit()
                model = model.cuda()
            else:
                model = GATNet(num_features=num_features, num_classes=2, hidden=args.hidden, head=8)
                model = model.cuda()
            # K-fold cross-validation
            K = 5
            all_f1_mean, all_f1_std = 0, 0
            all_accuracy_mean, all_accuracy_std = 0, 0
            all_recall_mean, all_recall_std = 0, 0
            all_precision_mean, all_precision_std = 0, 0
            all_auc_mean, all_auc_std = 0, 0
            all_aupr_mean, all_aupr_std = 0, 0
            truth = []
            probability = []
            f1_s = []
            accuracy_s = []
            recall_s = []
            precision_s = []
            auc_s = []
            aupr_s = []
            max = 0
            best = 0
            for i in range(K):
                logger.info('*' * 25, i + 1, '*' * 25)
                train_graphs, vali_graphs = get_k_fold_data(K, i, train_dataset)
                if args.causal:
                    # model = CausalGAT(num_features=num_features, num_classes=2, args=args, head=8)
                    # model = model.cuda()
                    # print(train_graphs)
                    # exit()
                    test_auc, f1, accuracy, recall, precision, auc, aupr, one_truth, one_probability = \
                        train_multiple_epochs(train_graphs, vali_graphs, model, args, logger)
                else:
                    # model = GATNet(num_features=num_features, num_classes=2, hidden=args.hidden, head=8)
                    # model = model.cuda()
                    # print(train_graphs)
                    # exit()
                    test_auc, f1, accuracy, recall, precision, auc, aupr, one_truth, one_probability = \
                        train_multiple_epochs_nocausal(train_graphs, vali_graphs, model, args, logger)

                truth.extend(one_truth)
                probability.extend(one_probability)
                f1_s.append(f1)
                accuracy_s.append(accuracy)
                recall_s.append(recall)
                precision_s.append(precision)
                auc_s.append(auc)
                aupr_s.append(aupr)

                if best < auc:
                    best = auc
                    torch.save(model, root_path + 'model.pth')

            logger.info('#' * 10, 'Final k-fold cross validation results', '#' * 10)
            logger.info('The %d-fold CV auc: %.4f +/- %.4f' % (i + 1, np.mean(auc_s), np.std(auc_s)))
            logger.info('The %d-fold CV aupr: %.4f +/- %.4f' % (i + 1, np.mean(aupr_s), np.std(aupr_s)))
            logger.info('The %d-fold CV f1-score: %.4f +/- %.4f' % (i + 1, np.mean(f1_s), np.std(f1_s)))
            logger.info('The %d-fold CV recall: %.4f +/- %.4f' % (i + 1, np.mean(recall_s), np.std(recall_s)))
            logger.info('The %d-fold CV accuracy: %.4f +/- %.4f' % (i + 1, np.mean(accuracy_s), np.std(accuracy_s)))
            logger.info('The %d-fold CV precision: %.4f +/- %.4f' % (i + 1, np.mean(precision_s), np.std(precision_s)))
            all_f1_mean = all_f1_mean + np.mean(f1_s)
            all_f1_std = all_f1_std + np.std(f1_s)

            all_recall_mean = all_recall_mean + np.mean(recall_s)
            all_recall_std = all_recall_std + np.std(recall_s)

            all_accuracy_mean = all_accuracy_mean + np.mean(accuracy_s)
            all_accuracy_std = all_accuracy_std + np.std(accuracy_s)

            all_precision_mean = all_precision_mean + np.mean(precision_s)
            all_precision_std = all_precision_std + np.std(precision_s)

            all_auc_mean = all_auc_mean + np.mean(auc_s)
            all_auc_std = all_auc_std + np.std(auc_s)

            all_aupr_mean = all_aupr_mean + np.mean(aupr_s)
            all_aupr_std = all_aupr_std + np.std(aupr_s)

            truth_predict = [truth, probability]
            all_results.append(truth_predict)

        np.save(root_path + 'truth.npy', np.array(truth))
        np.save(root_path + 'probability.npy', np.array(probability))

        ##################################################################################################
        model = torch.load(root_path + 'model.pth')
        if args.causal:
            [acc_co, acc_c, acc_o], test_auc, one_pred_result = eval_acc_causal(model, test_loader, device, args)
        else:
            [acc_co, acc_c, acc_o], test_auc, one_pred_result = eval_acc(model, test_loader, device, args)
        probability = one_pred_result[0]
        predict = one_pred_result[1]
        truth = one_pred_result[2]

        f1 = metrics.f1_score(truth, predict)
        accuracy = metrics.accuracy_score(truth, predict)
        recall = metrics.recall_score(truth, predict)
        precision = metrics.precision_score(truth, predict)
        fpr, tpr, thresholds1 = metrics.roc_curve(truth, probability, pos_label=1)
        auc_score = metrics.auc(fpr, tpr)
        p, r, thresholds2 = metrics.precision_recall_curve(truth, probability, pos_label=1)
        aupr_score = metrics.auc(r, p)

        logger.info('auc: {:.4f}'.format(auc_score))
        logger.info('aupr: {:.4f}'.format(aupr_score))
        logger.info('test_auc: {:.4f}'.format(test_auc))
        logger.info('recall: {:.4f}'.format(recall))
        logger.info('f1: {:.4f}'.format(f1))
        logger.info('accuracy: {:.4f}'.format(accuracy))
        logger.info('precision: {:.4f}'.format(precision))
        ################################################################################################

    logger.info("All end...")

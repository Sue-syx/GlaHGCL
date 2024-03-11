import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
from sklearn import metrics
from ranger import Ranger
import wandb


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_k_fold_data(k, i, data):
    data_size = int(len(data) / 2)
    data_pos = data[0:data_size]
    data_neg = data[data_size:]

    start = int(i * data_size // k)
    end = int((i + 1) * data_size // k)

    data_valid_pos = data_pos[start:end]
    data_train_pos = data_pos[0:start] + data_pos[end:data_size]
    data_valid_neg = data_neg[start:end]
    data_train_neg = data_neg[0:start] + data_neg[end:data_size]
    data_train = data_train_pos + data_train_neg
    data_valid = data_valid_pos + data_valid_neg
    return data_train, data_valid


def train_multiple_epochs(train_graphs, valid_graphs, model, args, logger):
    logger.info("starting train...")
    LR = 0.01
    batch_size = 4
    epochs = 1
    train_loader = DataLoader(train_graphs, batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_graphs, batch_size, shuffle=True, num_workers=0)
    optimizer = Ranger(model.parameters(), lr=0.001, weight_decay=0)
    start_epoch = 1
    pbar = tqdm(range(start_epoch, epochs + start_epoch))
    # print("!!!!!!!!!!!!!")
    for epoch in pbar:
        model.train()
        total_loss = 0
        total_loss_c = 0
        total_loss_o = 0
        total_loss_co = 0
        correct_o = 0
        for it, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)

            one_hot_target = data.y.view(-1).long()
            c_logs, o_logs, co_logs = model(data, eval_random=args.with_random)
            uniform_target = torch.ones_like(c_logs, dtype=torch.float).to(device) / model.num_classes

            c_loss = F.kl_div(c_logs, uniform_target, reduction='batchmean')
            o_loss = F.nll_loss(o_logs, one_hot_target)
            co_loss = F.nll_loss(co_logs, one_hot_target)
            loss = args.c * c_loss + args.o * o_loss + args.co * co_loss

            pred_o = o_logs.max(1)[1]
            correct_o += pred_o.eq(data.y.view(-1)).sum().item()
            loss.backward()
            total_loss += loss.item() * num_graphs(data)
            total_loss_c += c_loss.item() * num_graphs(data)
            total_loss_o += o_loss.item() * num_graphs(data)
            total_loss_co += co_loss.item() * num_graphs(data)
            optimizer.step()

        train_loss = total_loss / len(train_loader.dataset)
        [acc_co, acc_c, acc_o], train_auc, one_pred_result = eval_acc_causal(model, train_loader, device, args)
        logger.info('\n Epoch: {:03d}, Loss: {:.5f}, Train Auc: {:.5f}, Train Acc_co: {:.5f}, Train Acc_o: {:.5f}'
                    .format(epoch, train_loss, train_auc, acc_co, acc_o))

        if args.wandb_open:
            wandb.log({"train_roc": wandb.plot.roc_curve(one_pred_result[1], one_pred_result[0])})
    
    [acc_co, acc_c, acc_o], test_auc, one_pred_result = eval_acc_causal(model, valid_loader, device, args)
    # print(one_pred_result[0])
    # print(one_pred_result[1])
    # print(one_pred_result[2])
    # print("!!!!")
    # exit()
    # eval_acc_causal
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

    if args.wandb_open:
        wandb.log({"test_roc": wandb.plot.roc_curve(one_pred_result[1], one_pred_result[0])})
    return test_auc, f1, accuracy, recall, precision, auc_score, aupr_score, truth, probability


def eval_acc_causal(model, loader, device, args):
    model.eval()
    eval_random = args.eval_random
    predictions = torch.Tensor()
    probability = torch.Tensor()
    labels = torch.Tensor()
    correct = 0
    correct_c = 0
    correct_o = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            c_logs, o_logs, co_logs = model(data, eval_random=eval_random)
            pred = co_logs.max(1)[1].cpu().detach()
            pred_c = c_logs.max(1)[1].cpu().detach()
            pred_o = o_logs.max(1)[1].cpu().detach()
        probability = torch.cat((probability, torch.exp(co_logs.cpu().detach())), 0)
        predictions = torch.cat((predictions, pred), 0)
        labels = torch.cat((labels, data.cpu().y), 0)
        # print(labels.shape)
        correct += pred.eq(data.y.view(-1)).sum().item()
        correct_c += pred_c.eq(data.y.view(-1)).sum().item()
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()

    acc_co = correct / len(loader.dataset)
    acc_c = correct_c / len(loader.dataset)
    acc_o = correct_o / len(loader.dataset)

    proba = probability[:, 1]
    one_pred_result = np.vstack((proba, predictions, labels))
    fpr, tpr, _ = metrics.roc_curve(labels, proba, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return [acc_co, acc_c, acc_o], auc, one_pred_result


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train_multiple_epochs_nocausal(train_graphs, valid_graphs, model, args, logger):
    logger.info("starting train...")
    LR = 0.01
    batch_size = 64
    epochs = 80
    train_loader = DataLoader(train_graphs, batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_graphs, batch_size, shuffle=True, num_workers=0)
    optimizer = Ranger(model.parameters(), lr=0.001, weight_decay=0)
    start_epoch = 1
    pbar = tqdm(range(start_epoch, epochs + start_epoch))

    for epoch in pbar:
        model.train()
        total_loss = 0
        correct_o = 0
        for it, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)

            one_hot_target = data.y.view(-1).long()
            o_logs = model(data)
            loss = F.nll_loss(o_logs, one_hot_target)

            pred_o = o_logs.max(1)[1]
            correct_o += pred_o.eq(data.y.view(-1)).sum().item()
            loss.backward()
            total_loss += loss.item() * num_graphs(data)
            optimizer.step()

        train_loss = total_loss / len(train_loader.dataset)
        acc, train_auc, one_pred_result = eval_acc(model, train_loader, device, args)
        logger.info('\n Epoch: {:03d}, Loss: {:.5f}, Train Auc: {:.5f}, Train Acc: {:.5f}'
                    .format(epoch, train_loss, train_auc, acc))

        if args.wandb_open:
            wandb.log({"train_roc": wandb.plot.roc_curve(one_pred_result[1], one_pred_result[0])})

    acc, test_auc, one_pred_result = eval_acc(model, valid_loader, device, args)
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

    if args.wandb_open:
        wandb.log({"test_roc": wandb.plot.roc_curve(one_pred_result[1], one_pred_result[0])})
    return test_auc, f1, accuracy, recall, precision, auc_score, aupr_score, truth, probability


def eval_acc(model, loader, device, args):
    model.eval()
    predictions = torch.Tensor()
    probability = torch.Tensor()
    labels = torch.Tensor()
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            o_logs = model(data)
            pred = o_logs.max(1)[1].cpu().detach()
        probability = torch.cat((probability, torch.exp(o_logs.cpu().detach())), 0)
        predictions = torch.cat((predictions, pred), 0)
        labels = torch.cat((labels, data.cpu().y), 0)
        correct += pred.eq(data.y.view(-1)).sum().item()

    acc = correct / len(loader.dataset)

    proba = probability[:, 1]
    one_pred_result = np.vstack((proba, predictions, labels))
    fpr, tpr, _ = metrics.roc_curve(labels, proba, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return acc, auc, one_pred_result

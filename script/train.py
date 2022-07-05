import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Recall, Precision, FBetaScore, CohenKappa, MatthewsCorrCoef, ConfusionMatrix
from torchmetrics.classification.auroc import AUROC
from tqdm import tqdm


class BasicTrainer:
    def __init__(self, loss_func=None, n_classes=5, cuda=False, need_test=False):
        """
        Classification task.
        Evaluation_Metrics: ['Acc', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC', 'Confusion Matrix']

        :param loss_func:
        :param n_classes:
        :param cuda:
        :param need_test:
        """
        self.loss_func = loss_func
        self.cuda = cuda
        self.metrics = {
            'train': {
                'Acc': Accuracy(num_classes=n_classes),
                'AUC': AUROC(num_classes=n_classes),
                'Recall': Recall(num_classes=n_classes, average='macro'),
                'Precision': Precision(num_classes=n_classes, average='macro'),
                'F1': FBetaScore(num_classes=n_classes, average='macro'),
                'Kappa': CohenKappa(num_classes=n_classes),
                'MCC': MatthewsCorrCoef(num_classes=n_classes)
                # 'Confusion Matrix': ConfusionMatrix(num_classes=n_classes)
            },
            'val': {
                'Acc': Accuracy(num_classes=n_classes),
                'AUC': AUROC(num_classes=n_classes),
                'Recall': Recall(num_classes=n_classes, average='macro'),
                'Precision': Precision(num_classes=n_classes, average='macro'),
                'F1': FBetaScore(num_classes=n_classes, average='macro'),
                'Kappa': CohenKappa(num_classes=n_classes),
                'MCC': MatthewsCorrCoef(num_classes=n_classes)
                # 'Confusion Matrix': ConfusionMatrix(num_classes=n_classes)
            },
            'test': {
                'Acc': Accuracy(num_classes=n_classes),
                'AUC': AUROC(num_classes=n_classes),
                'Recall': Recall(num_classes=n_classes, average='macro'),
                'Precision': Precision(num_classes=n_classes, average='macro'),
                'F1': FBetaScore(num_classes=n_classes, average='macro'),
                'Kappa': CohenKappa(num_classes=n_classes),
                'MCC': MatthewsCorrCoef(num_classes=n_classes)
                # 'Confusion Matrix': ConfusionMatrix(num_classes=n_classes)
            } if need_test else None
        }

    def train_per_epoch(self, dataloader: DataLoader, model, opt, loss_func=None, cuda=None, info=''):
        if cuda is not None and isinstance(cuda, bool):
            self.cuda = cuda
        if self.cuda:
            model.cuda()
        else:
            model.cpu()
        if loss_func is not None:
            self.loss_func = loss_func
        self.reset_metrics(mode='train')
        grad_loss = 0.
        model.train()
        num_batch = len(dataloader)
        tbar = tqdm(dataloader)
        for i, (train_x, train_y) in enumerate(tbar):
            if train_y.shape[-1] == 1:
                train_y = train_y.squeeze(-1)
            if self.cuda:
                train_x = train_x.cuda()
                train_y = train_y.cuda()
            pred_y = model(train_x)
            loss = self.loss_func(pred_y, train_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            grad_loss += loss.item()
            for key in self.metrics['train'].keys():
                self.metrics['train'][key](pred_y, train_y)
            tbar.set_description(desc=f'Training {info}  [{i + 1}/{num_batch}]: {grad_loss / (i + 1)}, '
                                      f'{self.metrics["train"]["Acc"].compute().cpu().item()}')
        grad_loss /= num_batch
        return grad_loss, {
            key: self.metrics["train"][key].compute().cpu().item() for key in self.metrics['train'].keys()
        }

    def validation(self, dataloader: DataLoader, model, loss_func=None, cuda=None, val_test='val'):
        """
        :param dataloader:
        :param model:
        :param loss_func:
        :param evaluation:
        :param cuda:
        :param val_test: 'val' or 'test'.
        :return:
        """
        if cuda is not None:
            if cuda:
                model.cuda()
            else:
                model.cpu()
        if self.metrics['test'] is None and val_test == 'test':
            val_test = 'val'
        if loss_func is None:
            loss_func = self.loss_func
        assert loss_func is not None, 'loss_func is None and BasicTrainer.loss_func is None...'
        self.reset_metrics(mode=val_test)
        common_loss = 0.
        model.eval()
        num_batch = len(dataloader)
        tbar = tqdm(dataloader)
        for i, (val_x, val_y) in enumerate(tbar):
            if val_y.shape[-1] == 1:
                val_y = val_y.squeeze(-1)
            if self.cuda:
                val_x = val_x.cuda()
                val_y = val_y.cuda()
            with torch.no_grad():
                pred_y = model(val_x)
            common_loss += loss_func(pred_y, val_y).item()
            for key in self.metrics[val_test].keys():
                self.metrics[val_test][key](pred_y, val_y)
            tbar.set_description(desc=f'Validation [{i + 1}/{num_batch}]: {common_loss / (i + 1)}, '
                                      f'{self.metrics[val_test]["Acc"].compute().cpu().item()}')
        common_loss /= num_batch
        return common_loss, {
            key: self.metrics[val_test][key].compute().cpu().item() for key in self.metrics[val_test].keys()
        }

    def reset_metrics(self, mode='train'):
        """

        :param mode: 'train', 'val' or 'test'
        :return:
        """
        if self.metrics[mode] is not None:
            for key in self.metrics[mode].keys():
                self.metrics[mode][key].reset()

    def cuda(self):
        self.cuda = True
        for mode in ('train', 'val', 'test'):
            if self.metrics[mode] is None:
                continue
            for key in self.metrics[mode].keys():
                self.metrics[mode][key].reset()
        return self

    def get_evaluation_keys(self):
        return list(self.metrics['train'].keys())


def evaluation_models(dataloader: DataLoader, models: list, n_classes=8, loss_func=None, cuda=False, mode='mean'):
    metrics = {
        'Acc': Accuracy(num_classes=n_classes),
        'AUC': AUROC(num_classes=n_classes),
        'Recall': Recall(num_classes=n_classes, average='macro'),
        'Precision': Precision(num_classes=n_classes, average='macro'),
        'F1': FBetaScore(num_classes=n_classes, average='macro'),
        'Kappa': CohenKappa(num_classes=n_classes),
        'MCC': MatthewsCorrCoef(num_classes=n_classes)
        # 'Confusion Matrix': ConfusionMatrix(num_classes=n_classes)
    }
    for key in metrics.keys():
        metrics[key].reset()
    if cuda:
        for model in models:
            model.cuda().eval()
    else:
        for model in models:
            model.cpu().eval()
    num_batch = len(dataloader)
    tbar = tqdm(dataloader)
    common_loss = 0.
    for i, (val_x, val_y) in enumerate(tbar):
        if val_y.shape[-1] == 1:
            val_y = val_y.squeeze(-1)
        if cuda:
            val_x = val_x.cuda()
            val_y = val_y.cuda()
        if mode.lower() == 'mean':
            pred_list = []
            with torch.no_grad():
                for model in models:
                    pred_list.append(model(val_x))
            pred_y = sum(pred_list) / len(pred_list)
        elif mode.lower() == 'max':
            with torch.no_grad():
                pred_y = torch.zeros(val_x.size(0), n_classes)
                for model in models:
                    tmp = model(val_x)
                    tmp = torch.softmax(tmp, dim=1)
                    for batch_id in range(val_x.size(0)):
                        if tmp[batch_id].max() > pred_y[batch_id].max():
                            pred_y[batch_id] = tmp[batch_id]
        else:
            with torch.no_grad():
                preds = []
                for model in models:
                    preds.append(model(val_x))
            votes = []
            pred_y = torch.zeros(val_x.size(0), n_classes)
            for batch_id in range(val_x.size(0)):
                for m_id in range(len(models)):
                    votes.append(preds[m_id][batch_id].argmax())
                votes.sort()
                # print(votes[len(votes)//2], votes)
                pred_y[batch_id][votes[len(votes) // 2]] = 1.
                votes.clear()

        common_loss += loss_func(pred_y, val_y).item()
        for key in metrics.keys():
            metrics[key](pred_y, val_y)
        tbar.set_description(desc=f'Validation [{i + 1}/{num_batch}]: {common_loss / (i + 1)}, '
                                  f'{metrics["Acc"].compute().cpu().item()}')
    common_loss /= num_batch
    return common_loss, {
        key: metrics[key].compute().cpu().item() for key in metrics.keys()
    }

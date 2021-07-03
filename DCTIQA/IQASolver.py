import torch
from scipy import stats
import numpy as np
from torchvision import  models
import data_loader
import torch.nn as nn

DEVICE = torch.device("cuda:7") if torch.cuda.is_available() else "cpu"


class IQASolver(object):
    """Solver for training and testing hyperIQA"""
    def __init__(self, config, path, train_idx, test_idx):

        self.epochs = config.epochs

        self.model_dct = models.resnet50(pretrained=False)
        num_ftrs = self.model_dct.fc.in_features
        self.model_dct.fc = nn.Linear(num_ftrs, 1)
        self.model_dct = self.model_dct.to(DEVICE)

        self.model_dct.train(True)

        self.l1_loss = torch.nn.L1Loss().cuda()

        self.lr = config.lr
        self.weight_decay = config.weight_decay

        train_loader = data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size, batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

        self.solver = torch.optim.Adam(self.model_dct.parameters(), weight_decay=self.weight_decay)

    def train(self):
        """Training"""
        best_srcc = 0.0
        best_plcc = 0.0
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            for img, label in self.train_data:

                img = img.to(DEVICE)
                label = label.to(DEVICE)

                self.solver.zero_grad()

                # Generate weights for target network
                pred = self.model_dct(img)  # 'paras' contains the network weights conveyed to target network

                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                loss = self.l1_loss(pred.squeeze(), label.float().detach())
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

            test_srcc, test_plcc = self.test(self.test_data)
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
            print('%d   \t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc))

            # Update optimizer
            lr = self.lr / pow(10, (t // 6))

            self.solver = torch.optim.Adam(self.model_dct.parameters(), weight_decay=self.weight_decay)

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        self.model_dct.train(False)
        pred_scores = []
        gt_scores = []

        for img, label in data:
            # Data.
            # img = torch.tensor(img.cuda())
            img = img.to(DEVICE)
            # label = torch.tensor(label.cuda())
            label = label.to(DEVICE)

            pred = self.model_dct(img)

            pred_scores.append(float(pred.item()))
            gt_scores = gt_scores + label.cpu().tolist()

        # pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        # gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        self.model_dct.train(True)
        return test_srcc, test_plcc
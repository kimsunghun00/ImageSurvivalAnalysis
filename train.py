import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import tqdm
import copy
import time
from torch.optim.lr_scheduler import StepLR
from utils.utils import lifelines_cindex

class Trainer:
    def __init__(self, net, train_loader, test_loaer, criterion, optimizer,
                 epochs=100, lr = 0.001, l2=0, device = None):
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loaer
        self.criterion = criterion
        self.optimizer = optimizer(self.net.parameters(), lr = lr, weight_decay=l2)
        self.epochs = epochs

        if device is not None:
            self.device = device
        else:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.val_losses = []
        self.train_cindices = []
        self.val_cindices = []

        self.best_model_wts = copy.deepcopy(self.net.state_dict())
        self.best_loss = 10.

    def fit(self):
        lr_scheduler = StepLR(self.optimizer, step_size=5, gamma=0.9)
        since = time.time()
        for epoch in range(self.epochs):
            survtime_true = []
            survtime_preds = []
            censors = []
            running_loss = 0.
            self.net.train()
            for i_batch, sample_batch in tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                image_batch, censor_batch, survtime_batch = sample_batch['image'], sample_batch['censor'], sample_batch['survtime']

                image_batch = image_batch.to(self.device, dtype=torch.float)
                survtime_batch = survtime_batch.to(self.device, dtype=torch.float)
                censor_batch = censor_batch.to(self.device, dtype=torch.float)
                survtime_preds_batch = self.net(image_batch)

                loss = self.criterion(survtime_preds_batch, censor_batch, survtime_batch, self.device)
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.net.parameters(), 5)
                self.optimizer.step()

                running_loss += loss.item()
                survtime_preds.append(survtime_preds_batch)
                survtime_true.append(survtime_batch)
                censors.append(censor_batch)
            # 미니 배치 단위의 예측 결과를 하나로 묶는다
            survtime_true = torch.cat(survtime_true)
            survtime_preds = torch.cat(survtime_preds)
            censors = torch.cat(censors)


            lr_scheduler.step()
            train_cindex = lifelines_cindex(survtime_preds, censors, survtime_true)
            self.train_cindices.append(train_cindex)

            # 검증 데이터의 예측 c-index 및 loss
            val_loss, val_cindex = self.eval_net(self.test_loader, self.device)
            self.val_losses.append(val_loss)
            self.val_cindices.append(val_cindex)

            # epoch 결과 표시
            print('epoch: {}/{} \ntrain_cindex: {:.4f}, val_loss: {:.4f}, val_cindex: {:.4f}'.format(epoch+1, self.epochs,
                                                                                                     self.train_cindices[-1],
                                                                                                     self.val_losses[-1],
                                                                                                     self.val_cindices[-1]))
        time_elapsed = time.time() - since
        print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    def eval_net(self, data_loader, device):
        survtime_true = []
        survtime_preds = []
        censors = []
        running_loss = 0.
        self.net.eval()
        for sample_batch in data_loader:
            image_batch, censor_batch, survtime_batch = sample_batch['image'], sample_batch['censor'], sample_batch['survtime']

            image_batch = image_batch.to(device, dtype=torch.float)
            survtime_batch = survtime_batch.to(device, dtype=torch.float)
            censor_batch = censor_batch.to(device, dtype=torch.float)

            with torch.no_grad():
                survtime_preds_batch = self.net(image_batch)
            survtime_preds.append(survtime_preds_batch)
            survtime_true.append(survtime_batch)
            censors.append(censor_batch)
            loss = self.criterion(survtime_preds_batch, censor_batch, survtime_batch, device)
            running_loss += loss.item()
        survtime_true = torch.cat(survtime_true)
        survtime_preds = torch.cat(survtime_preds)
        censors = torch.cat(censors)

        val_loss = running_loss / len(data_loader)
        val_cindex = lifelines_cindex(survtime_preds, censors, survtime_true)

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(self.net.state_dict())

        return val_loss, val_cindex

    def evaluation(self, data_loader, device):
        model = self.get_best_model()
        running_loss = 0.
        model.eval()
        survtime_true = []
        survtime_preds = []
        censors = []
        for sample_batch in tqdm.tqdm(data_loader, total=len(data_loader)):
            image_batch, censor_batch, survtime_batch = sample_batch['image'], sample_batch['censor'], sample_batch['survtime']

            image_batch = image_batch.to(device, dtype=torch.float)
            survtime_batch = survtime_batch.to(device, dtype=torch.float)
            censor_batch = censor_batch.to(device, dtype=torch.float)

            with torch.no_grad():
                survtime_preds_batch = model(image_batch)
            survtime_preds.append(survtime_preds_batch)
            survtime_true.append(survtime_batch)
            censors.append(censor_batch)
            loss = self.criterion(survtime_preds_batch, censor_batch, survtime_batch, device)
            running_loss += loss.item()
        survtime_true = torch.cat(survtime_true)
        survtime_preds = torch.cat(survtime_preds)
        censors = torch.cat(censors)

        test_loss = running_loss / len(data_loader)
        test_cindex = lifelines_cindex(survtime_preds, censors, survtime_true)

        print(test_cindex)

        return test_loss, test_cindex

    def history(self):
        history = {'val_losses' : self.val_losses,
                   'train_cindices' : self.train_cindices, 'val_cindices' : self.val_cindices}
        return history

    def get_best_model(self):
        self.net.load_state_dict(self.best_model_wts)
        return self.net

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM_Model(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 num_classes):
        super(LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class DeepLog(object):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 num_classes,
                 window_size,
                 batch_size,
                 num_epochs,
                 lr,
                 topN):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.topN = topN

    def fit(self, train_loader):
        print('Running environment : {}'.format(device))
        train_loader = DataLoader(train_loader,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  pin_memory=True)
        model = LSTM_Model(input_size=self.input_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           num_classes=self.num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # 1 train model
        start_time = time.time()
        total_step = len(train_loader)
        for epoch in range(self.num_epochs):
            train_loss = 0
            for step, (seq, label) in enumerate(train_loader):
                seq = seq.clone().detach().\
                    view(-1, self.window_size, self.input_size).to(device)
                output = model(seq)
                loss = criterion(output, label.to(device))
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            print('Epoch [{}/{}], train_loss: {:.4f}'.
                  format(epoch + 1, self.num_epochs, train_loss / total_step))
        elapsed_time = time.time() - start_time
        print('elapsed_time: {:.3f}s'.format(elapsed_time))

        # 2 save model
        model_name = 'Adam_batch_size{}_epoch{}_{}'. \
            format(str(self.batch_size), str(self.num_epochs), device.type)
        model_dir = './result/DeepLog'
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        torch.save(model.state_dict(), model_dir + '/' + model_name + '.pt')
        print(f"Finished training, model saved in: {model_dir}/{model_name}.pt")

        return model_dir + '/' + model_name + '.pt'

    def predict(self, test_loader, model_path):
        # 1 load model
        model = LSTM_Model(self.input_size,
                           self.hidden_size,
                           self.num_layers,
                           self.num_classes).to(device)
        if device.type == 'cpu' and 'cuda' in model_path:
            model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        else:
            model.load_state_dict(torch.load(model_path))

        # 2 test
        y_predicted = []
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            for line in test_loader:
                anomaly_flag = 0
                for i in range(len(line) - self.window_size):
                    seq = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    seq = torch.tensor(seq, dtype=torch.float). \
                        view(-1, self.window_size, self.input_size).to(device)
                    label = torch.tensor(label).view(-1).to(device)
                    output = model(seq)
                    predicted = torch.argsort(output, 1)[0][-self.topN:]
                    if label not in predicted:
                        anomaly_flag = 1
                        break
                y_predicted.append(anomaly_flag)
        elapsed_time = time.time() - start_time
        print('elapsed_time: {:.3f}s'.format(elapsed_time))
        print('Finished Predicting')

        return y_predicted

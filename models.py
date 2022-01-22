import torch
import torch.nn as nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, pretrain_cnn_model, sample_size=9):
        super(Model, self).__init__()
        self.sample_size = sample_size

        self.pretrain_cnn_model = pretrain_cnn_model
        self.pretrain_cnn_model.linear = nn.Identity()

        self.rnn = RecurrentLayer()
        self.linear1 = nn.Linear(32 * self.sample_size, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 128)

        self.out = nn.Linear(128, 1)

    def forward(self, input):  # input : [batch, step, channel, w, h]
        input = input.reshape(-1, self.sample_size, 3, 224, 224)

        conv_outputs = []
        for i in range(self.sample_size):
            image = input[:, i, :, :, :]  # [batch, channel, w, h]
            visual_feature = self.pretrain_cnn_model(image)
            conv_outputs.append(visual_feature)

        visual_features = torch.stack(conv_outputs)  # [seq_len, batch, input_size]
        sequence_feature = self.rnn(visual_features)

        rnn_outputs = []
        for i in range(self.sample_size):
            rnn_outputs.append(sequence_feature[i, :, :])
        aggregation = torch.cat(rnn_outputs, dim = 1)
        linear_output = F.relu(self.linear1(aggregation))
        linear_output = F.relu(self.linear2(linear_output))
        linear_output = F.relu(self.linear3(linear_output))
        prediction = self.out(linear_output)

        return prediction



class ConvFeatureExtractor(nn.Module):
    def __init__(self, input_channel = 3, output_channel = 32):
        super(ConvFeatureExtractor, self).__init__()
        self.output_channel = [4, 8, 16, output_channel]
        
        self.ConvNet = nn.Sequential(nn.Conv2d(input_channel, self.output_channel[0], kernel_size = 3, stride = 2, padding = 1),
                                     nn.ReLU(inplace = True),
                                     nn.MaxPool2d(2, 2), # 4x56x56
                                     
                                     nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1),
                                     nn.ReLU(True),
                                     nn.MaxPool2d(2, 2), # 8x28x28
                                     
                                     nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), # 16x28x28
                                     nn.ReLU(True),
                                     nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1),
                                     nn.MaxPool2d(2, 2), # 16x14x14
                                     
                                     nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias = False),
                                     nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True), # 32x14x14
                                     nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias = False),
                                     nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
                                     nn.MaxPool2d(2, 2), # 32x7x7
                                     nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 0), #32x5x5
                                     nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
                                     nn.AvgPool2d(5, 5) # 32x1x1
                                    )
        
    def forward(self, x):
        x = self.ConvNet(x) # [batch, channel, w, h]
        x = x.view(-1, self.output_channel[3]) # [batch, 32]
        
        return x


class RecurrentLayer(nn.Module):
    def __init__(self, input_size = 32, hidden_size = 16):
        super(RecurrentLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=2, bidirectional=True)
        
    def forward(self, x): # input : [seq_len, batch, input_size]
        rnn_output, _ = self.lstm(x) # rnn_output : [seq_len, batch, hidden_size]
                
        return rnn_output


class PretrainNet(nn.Module):
    def __init__(self, input_channel=3, output_channel=32):
        super(PretrainNet, self).__init__()
        self.output_channel = [4, 8, 16, output_channel]

        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 4x56x56

            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 8x28x28

            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1),  # 16x28x28
            nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1),
            nn.MaxPool2d(2, 2),  # 16x14x14

            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),  # 32x14x14
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 32x7x7
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 0),  # 32x5x5
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.linear = nn.Linear(self.output_channel[3], 1)

    def forward(self, x):
        x = self.ConvNet(x)  # [batch, channel, w, h]
        x = x.view(-1, self.output_channel[3])  # [batch, 32]
        x = self.linear(F.relu(x))
        return x


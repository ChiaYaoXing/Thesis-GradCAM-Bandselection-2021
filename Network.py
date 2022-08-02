import torch.nn as nn
import torch.nn.functional as f


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 3)
        self.conv2 = nn.Conv1d(32, 32, 3)
        self.conv3 = nn.Conv1d(32, 32, 3)
        self.pool = nn.MaxPool1d(2)
        self.conv4 = nn.Conv1d(32, 32, 3)
        self.conv5 = nn.Conv1d(32, 32, 3)
        self.fc = nn.Linear(1194 * 32, 7)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = x.reshape(-1, 1194 * 32)
        x = self.fc(x)
        return x

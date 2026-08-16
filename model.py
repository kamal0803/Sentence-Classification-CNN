import torch

class SentenceClassifierCNNSingleChannel(torch.nn.Module):

    def __init__(self, embedding, out_channels=100, p=0.5, out_features=6):
        super().__init__()

        self.embedding_table = embedding

        self.m3 = torch.nn.Conv1d(in_channels=300,out_channels=out_channels,kernel_size=3)
        self.m4 = torch.nn.Conv1d(in_channels=300,out_channels=out_channels,kernel_size=4)
        self.m5 = torch.nn.Conv1d(in_channels=300,out_channels=out_channels,kernel_size=5)

        self.dropout = torch.nn.Dropout(p=p)
        self.linear = torch.nn.LazyLinear(out_features=out_features)

    def forward(self, X):
        X = self.embedding_table(X).T

        conv3 = torch.relu(self.m3(X))
        conv3 = torch.amax(conv3, 1)
    
        conv4 = torch.relu(self.m4(X))
        conv4 = torch.amax(conv4, 1)
    
        conv5 = torch.relu(self.m5(X))
        conv5 = torch.amax(conv5, 1)
        
        conv = torch.cat((conv3, conv4, conv5), dim=0)

        conv = self.dropout(conv)
        final_op = self.linear(conv)
        
        return final_op


class SentenceClassifierCNNMultiChannel(torch.nn.Module):

    def __init__(self, embedding_static, embedding_non_static, out_channels=100, p=0.5, out_features=6):
        super().__init__()

        self.embedding_static = embedding_static
        self.embedding_non_static = embedding_non_static

        self.m3 = torch.nn.Conv1d(in_channels=300, out_channels=out_channels, kernel_size=3)
        self.m4 = torch.nn.Conv1d(in_channels=300, out_channels=out_channels, kernel_size=4)
        self.m5 = torch.nn.Conv1d(in_channels=300, out_channels=out_channels, kernel_size=5)

        self.dropout = torch.nn.Dropout(p=p)
        self.linear = torch.nn.LazyLinear(out_features=out_features)

    def forward(self, X):

        X_static = self.embedding_static(X).T
        X_non_static = self.embedding_non_static(X).T

        conv3 = self.m3(X_static) + self.m3(X_non_static)
        conv3 = torch.relu(conv3)
        conv3 = torch.amax(conv3, 1)

        conv4 = self.m4(X_static) + self.m4(X_non_static)
        conv4 = torch.relu(conv4)
        conv4 = torch.amax(conv4, 1)

        conv5 = self.m5(X_static) + self.m5(X_non_static)
        conv5 = torch.relu(conv5)
        conv5 = torch.amax(conv5, 1)

        conv = torch.cat((conv3, conv4, conv5), dim=0)

        conv = self.dropout(conv)
        final_op = self.linear(conv)

        return final_op
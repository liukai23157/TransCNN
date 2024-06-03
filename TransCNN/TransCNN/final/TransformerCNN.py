import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader,Dataset
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#textcnn,input = (batch_size,embedding_size,len)#(13,201)
def position_encoding(seq,d):
    """
    Position encoding features introduced in "Attention is all your need",
    the b is changed to 1000 for the short length of sequence.
    """
    b = 1000
    N = len(seq)
    value = []
    for pos in range(N):
        tmp = []
        for i in range(d // 2):
            tmp.append(pos / (b ** (2 * i / d)))
        value.append(tmp)
    value = np.array(value)
    pos_encoding = np.zeros((N, d))
    pos_encoding[:, 0::2] = np.sin(value[:, :])
    pos_encoding[:, 1::2] = np.cos(value[:, :])
    return pos_encoding

class my_dataset(Dataset):
    def __init__(self,data_iter,d):
        self.data = data_iter
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        self.d = d
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        sequence,label = self.data.iloc[index,1],self.data.iloc[index,2]
        encoding_seq = [self.vocab[token] for token in sequence]
        PositionEnconding = position_encoding(sequence,self.d)
        encoding_seq = torch.LongTensor(encoding_seq)
        PositionEnconding = torch.FloatTensor(PositionEnconding)
        label = torch.LongTensor([label])
        return encoding_seq,PositionEnconding,label.squeeze()
class TextCNN(nn.Module):
    def __init__(self,embedding_size,window_size,n_filters,dropout=0.4):
        super(TextCNN, self).__init__()

        self.embed = nn.Embedding(4,embedding_size)
        self.TransformerLayer = nn.TransformerEncoderLayer(embedding_size,nhead=2)
        self.seq2vec = nn.TransformerEncoder(self.TransformerLayer,num_layers=2)

        self.convs = nn.ModuleList([
            nn.Conv2d(1,n_filters,kernel_size=(embedding_size,ws))
            for ws in window_size
        ]) #[n_filters,1,len-k+1]
        self.fc = nn.Linear(len(window_size)*n_filters,2)
        self.dropout = nn.Dropout(dropout)
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0.0)

            # 使用 Xavier 初始化全连接层参数
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
    def forward(self,x,pos_embed):
        x = self.embed(x) + pos_embed
        x = x.permute(1,0,2)
        x = self.seq2vec(x)
        x = x.permute(1,2,0)
        x = x.unsqueeze(1)
        conv_results = [F.relu(conv(x)).squeeze(2) for conv in self.convs]
        pool_results = [F.max_pool1d(conv_result,conv_result.size(2)).squeeze(2) for conv_result in conv_results]#[batch_size,n_fiters]
        result = torch.cat(pool_results,dim=1)
        result = self.fc(result)
        soft_res = F.softmax(result,dim=1)


        return result,soft_res

    def encoder(self,x,pos_embed):
        x = self.embed(x) + pos_embed
        x = x.permute(1, 0, 2)
        x = self.seq2vec(x)
        x = x.permute(1, 2, 0)
        x = x.reshape(-1,x.shape[1]*x.shape[2])
        return x

test_data = pd.read_csv('test.csv', header=None)
data_test = my_dataset(test_data, 16)
test_loader = DataLoader(data_test, batch_size=10, shuffle=False)
tensor = []
model = TextCNN(16,[4,8,12],32)
for a,b,c in test_loader:
    features = model.encoder(a,b)
    tensor.append(features)
    print(features.size())
y_test = list(test_data.iloc[:,2])
res = torch.cat(tensor,dim=0)
tsne = TSNE(n_components=2,random_state=42)
feature_2d = tsne.fit_transform(res.detach().numpy())
plt.figure(figsize=(8, 6))
plt.scatter(feature_2d[:, 0], feature_2d[:, 1], c=y_test, cmap='viridis', alpha=0.7)
plt.colorbar()
plt.title('2D visualization of high-dimensional features (before prediction)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()

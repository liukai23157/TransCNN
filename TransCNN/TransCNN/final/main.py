import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset,Subset
import torch.optim as optim
import pandas as pd
from sklearn.metrics import auc,roc_curve,precision_recall_curve,average_precision_score
from sklearn.model_selection import KFold
import numpy as np
import warnings
import time
from sklearn.manifold import TSNE
import random
warnings.filterwarnings('ignore')
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
        self.seq2vec = nn.TransformerEncoder(self.TransformerLayer,num_layers=3)

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

    def enconder(self,x,pos_embed):
        x = self.embed(x) + pos_embed
        x = x.permute(1, 0, 2)
        x = self.seq2vec(x)
        x = x.permute(1, 2, 0)
        x = x.reshape(-1, x.shape[1] * x.shape[2])
        return x

def evaluation_metric(pred_prob,pred_label,real_label):
    L = len(real_label)
    tp,tn,fp,fn = 0.0,0.0,0.0,0.0
    for i in range(L):
        if real_label[i] == 1:
            if real_label[i] == pred_label[i]:
                tp += 1
            else:
                fn += 1
        else:
            if real_label[i] == pred_label[i]:
                tn += 1
            else:
                fp += 1
    #Accuracy
    ACC = (tp + tn) / L
    #Sensitivity
    if tp + fn == 0.0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = tp / (tp+fn)

    #Specificity
    if tn+fp == 0.0:
        Specificity = 0.0
    else:
        Specificity = tn / (tn+fp)
    #MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = (tp*tn-fp*fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    #Roc and auc,to draw roc curve
    FPR,TPR,thresholds = roc_curve(real_label,pred_prob,pos_label=1)
    AUC = auc(FPR,TPR)

    #prc and ap , constarte on postive samples
    precision,recall,threshold = precision_recall_curve(real_label,pred_prob,pos_label=1)
    AP = average_precision_score(real_label,pred_prob,average='macro',pos_label=1,sample_weight=None)

    performance = [ACC,Sensitivity,Specificity,AUC,MCC]
    roc_data = [FPR,TPR,AUC]
    pr_data =  [recall,precision,AP]
    return performance,roc_data,pr_data


def evaluate(data_iter,model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_prob = []
    label_pred = []
    label_real = []
    features = []
    for embed,pos_embed,label in data_iter:
        embed, pos_embed,label = embed.to(device),pos_embed.to(device), label.to(device)
        output,soft_output= model(embed,pos_embed)
        feature = model.enconder(embed,pos_embed)
        features.append(feature)
        pred_prob += soft_output[:,1].tolist()
        label_pred += soft_output.argmax(axis=1).tolist()
        label_real += label.tolist()
    performance,roc_data,pr_data = evaluation_metric(pred_prob,label_pred,label_real)
    return performance,roc_data,pr_data,label_real,features

class EarlyStopping:
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_acc = None

    def __call__(self, val_acc, model):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        self.best_acc = val_acc
        torch.save(model.state_dict(), 'checkpoint1.pt')

def pre_train(train_loader,test_loader,num_epochs,network):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = network.to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)
    criterion_model = nn.CrossEntropyLoss(reduction='mean')
    early_stopping = EarlyStopping(patience=10, delta=0.001)
    loss_list = []
    best_acc = 0.0
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for embed,pos_embed,label in train_loader:
            embed,pos_embed,label = embed.to(device),pos_embed.to(device),label.to(device)
            output,soft_output = model(embed,pos_embed)
            loss = criterion_model(output,label)
            optimizer.zero_grad()
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()
        # 计算一次epoch的损失
        loss_list.append(running_loss)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            train_performance, train_roc_data, train_pr_data, train_label_real,_ = evaluate(train_loader, model)
            test_performance, test_roc_data, test_pr_data, test_label_real,_ = evaluate(test_loader, model)
        train_acc = train_performance[0]
        test_acc = test_performance[0]
        print('Epoch [{}/{}], Train_metric:acc:{:.4f}, Test_metric:acc:{:.4f}'.format(epoch + 1, num_epochs, train_acc,
                                                                                      test_acc))
        if test_acc > best_acc:
            best_acc = test_acc
            best_performance = test_performance

        print(best_performance)
        best_roc_data = test_roc_data
        best_pr_data = test_pr_data
        early_stopping(test_acc, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    end_time = time.time()
    print('Training completed in {:.2f} seconds'.format(end_time - start_time))
    model.load_state_dict(torch.load('checkpoint1.pt'))
    return best_performance,best_roc_data,best_pr_data
if __name__ == '__main__':
    train_data = pd.read_csv('datasets/trainset.csv',header=None)
    data_train = my_dataset(train_data, 16)
    L = len(train_data)
    all_numbers = list(range(L))
    random.seed(42)
    train_index = random.sample(all_numbers,int(L*0.8))
    val_index = [num for num in all_numbers if num not in train_index]
    train_subset = Subset(data_train,train_index)
    train_loader = DataLoader(train_subset,batch_size=64,shuffle=True)
    val_subset = Subset(data_train, val_index)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    # best hyperparamters combation:embedding_size:16,n_filters:64,[4,8,12]
    net = TextCNN(16, [4,8,12], 64)
    best_performance, best_roc,best_pr = pre_train(train_loader, val_loader, 100, net)
    print(best_performance)

    #predict on testset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net.to(device)
    model.load_state_dict(torch.load('checkpoint1.pt'))
    test_data = pd.read_csv('datasets/testset.csv', header=None)
    data_test = my_dataset(test_data, 16)
    test_loader = DataLoader(data_test, batch_size=64, shuffle=False)
    model.eval()
    with torch.no_grad():
        performance, roc_data, pr_data, label_real,features = evaluate(test_loader, model)
    
    tensor = torch.cat(features,dim=0).cpu()
    res = tensor.detach().numpy()
    np.savetxt('output.csv',res, delimiter=',', fmt='%f')


    np.savetxt('pr1_p', pr_data[0], fmt='%f')
    np.savetxt('pr1_r', pr_data[1], fmt='%f')
    print(performance)
    print(roc_data)
    print('-----------------------------')
    print(pr_data)

    #best hyperparamters combation:embedding_size:16,n_filters:64,[4,8,12]

    # L = len(train_data)
    # kf = KFold(n_splits=5,shuffle=True,random_state=42)
    # best_acc = []
    # for train_index,test_idx in kf.split(range(L)):
    #     def function(params):
    #         embedding_size = params['embedding_size']
    #         window_size = params['window_size']
    #         n_filters = params['n_filters']
    #
    #         data_train = my_dataset(train_data, embedding_size)
    #         train_subset = Subset(data_train, train_index)
    #         test_subset = Subset(data_train, train_index)
    #         train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    #         test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    #         net = TextCNN(embedding_size, window_size, n_filters)
    #         loss, best_performance, best_acc = pre_train(train_loader, test_loader, 100, net)
    #         return -best_acc
    #
    #     #best hyperparamters combation:embedding_size:16,n_filters:64,[4,8,12]
    #     param_space = {'embedding_size': hp.choice('embedding_szie', [8, 10, 12, 14, 16]),
    #                     'window_size':hp.choice('window_size',[[3,4,5],[4,5,6],[3,5,7],[5,7,9],[4,6,8],
    #                                    [3,6,9],[4,7,10],[3,7,10],[4,8,12],[3,7,11]]),
    #                     'n_filters':hp.choice('n_filters',[16,32,48,64])}
    #     trials = Trials()
    #     best = fmin(fn=function, space=param_space, algo=tpe.suggest, max_evals=100, trials=trials)
    #
    #     best_trial_idx = trials.best_trial['tid']
    #     best_trial = trials.trials[best_trial_idx]
    #     best_acc.append(-best_trial['result']['loss'])
    #     print(best_trial['result']['loss'],best_trial['misc']['vals'])
    #
    # print(max(best_acc))
    

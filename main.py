
# coding: utf-8

# In[19]:


import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

dimensions = 150
batchSize = 4


# In[20]:


class LSTMTagger(nn.Module):

    def __init__(self, tagset_size, embedding_dim=dimensions, hidden_dim=10):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self, x=None):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if x==None:
            return (Variable(torch.zeros(1, batchSize, self.hidden_dim)),
                    Variable(torch.zeros(1, batchSize, self.hidden_dim)))
        else:
            return (Variable(x[0].data),Variable(x[1].data))

    def forward(self, sentence):
        lstm_out, self.hidden_out = self.lstm(sentence, self.hidden)
        tag_space = self.hidden2tag(lstm_out[-1])
        self.hidden=self.init_hidden(self.hidden_out)
        return tag_space


# In[21]:


class NTUQuadsDataset(Dataset):

    # Initialize your data, download, etc.
    def __init__(self):
        x = np.loadtxt('TrainData', delimiter=',', dtype=np.str, usecols = [0])
        y = np.loadtxt('TrainData', delimiter=',', dtype=np.float32, usecols = [1])
        self.len = x.shape[0]
        data = []
        for i in range (0, x.shape[0]):
            data.append(np.fromstring(x[i], dtype=float, sep=" "))
        h = np.array(data)
        maxlen = max(len(r) for r in h)
        Z = np.zeros((len(h), maxlen))
        for enu, row in enumerate(h):
            Z[enu, :len(row)] += row
        j = torch.from_numpy(np.stack(Z))
        numFeaturesInFrame = dimensions
        self.x_data = j.view(len(h), -1, numFeaturesInFrame).float()
        self.y_data = torch.from_numpy(y).long()
        

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# In[22]:


class NTUQuadsDatasetTest(Dataset):

    # Initialize your data, download, etc.
    def __init__(self):
        x = np.loadtxt('TestingData', delimiter=',', dtype=np.str, usecols = [0])
        y = np.loadtxt('TestingData', delimiter=',', dtype=np.float32, usecols = [1])
        self.len = x.shape[0]
        data = []
        for i in range (0, x.shape[0]):
            data.append(np.fromstring(x[i], dtype=float, sep=" "))
        h = np.array(data)
        maxlen = max(len(r) for r in h)
        Z = np.zeros((len(h), maxlen))
        for enu, row in enumerate(h):
            Z[enu, :len(row)] += row
        j = torch.from_numpy(np.stack(Z))
        numFeaturesInFrame = dimensions
        self.x_data = j.view(len(h), -1, numFeaturesInFrame).float()
        self.y_data = torch.from_numpy(y).long()
        

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# In[23]:


dataset = NTUQuadsDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batchSize,
                          shuffle=True,
                          num_workers=2)

lstmTagger = LSTMTagger(3, dimensions, 3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lstmTagger.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)


# In[31]:


for epoch in range(100):
    for name, param in lstmTagger.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    print("Epoch ", epoch, "\n-----")
    totalLoss = 0
    
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.view(-1, batchSize, dimensions)
#         inputs = inputs.view(-1, dimensions)
        
        
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # Run your training process
        
#         print(epoch, i, "inputs", inputs.data, "labels", labels.data)
        output = lstmTagger(inputs)
#         print(output)
#         loss = criterion(output[output.shape[0]-1].view(-1, 3), labels)
#         print('Output shape is ', output.shape, 'labels shape is', labels.shape)
        loss = criterion(output, labels)
        totalLoss += loss.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Total Loss in Epoch = ', totalLoss)


# In[25]:


testDataset = NTUQuadsDataset()
test_loader = DataLoader(dataset=testDataset,
                          batch_size=batchSize,
                          shuffle=True,
                          num_workers=2)


# In[26]:


correct = 0
total = 0
for i, data in enumerate(test_loader, 0):
    total = total + 1
 
    # get the inputs
    inputs, labels = data
#     print(inputs)
    
    inputs = inputs.view(-1, batchSize, dimensions)

        
    # wrap them in Variable
    inputs, labels = Variable(inputs), Variable(labels)

    # Run your training process
#     print(epoch, i, "inputs", inputs.data, "labels", labels.data)
    output = lstmTagger(inputs)
#     print(output)
        
    sm = F.softmax(output)
    print(sm)
    val, ind = sm.max(0)
    print("Guessed Index ", ind.data,  " Actual Label ",  labels.data)
    if ind.data.numpy()[0] == labels.data.numpy()[0]:
        correct = correct + 1


# In[27]:


accuracy = correct/total * 100
accuracy


# In[ ]:


a = torch.randn(5, 3, 2)
a


# In[ ]:


a[-1]


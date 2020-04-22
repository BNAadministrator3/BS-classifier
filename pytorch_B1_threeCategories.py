from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from A1c_leaveOneOutData import dataset_operator,  buildTrainValtest
from tqdm import tqdm
from random import randint
import pdb

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)   # try to use this networks
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(2112, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape) [64,32,8,24]
        x = F.relu(x)
        x = self.conv2(x)
        # print(x.shape) [64,64,6,22]
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # print(x.shape) [64,64,3,11]
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # print(x.shape) [64,2112]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
        
def train(model, device, yield_data, optimizer, epoch, iterations_per_epoch, log_interval=None):
    model.train() # boot batch_normalization and dropout, vice versa.
    pbar = tqdm(yield_data)
    iteration = 0
    loss_buffer = []
    for input, labels in pbar:
        input,labels = torch.from_numpy(input),torch.from_numpy(labels)
        input,labels = input.float(),labels.long()
        input,labels = input.to(device),labels.to(device)
        optimizer.zero_grad() #every time a batch comes, the gradient should be set zero.
        output = model(input)
        loss = F.nll_loss(output, labels)
        loss_buffer.append(loss.item())
        loss.backward()
        optimizer.step()
        if iteration == iterations_per_epoch:
            break
        else:
            iteration += 1
    pbar.close()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(
                epoch, loss.item()))
    print('The loss details: \n {}'.format(loss_buffer))
    pdb.set_trace() 
                
def test(model, device, data, type):
    model.eval()
    assert(type in ('train','test'))
    if type == 'test':
        num_data = len(data.testList)  # 获取数据的数量
    else:
        num_data = len(data.trainList)
    ran_num = randint(0, num_data - 1)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i in tqdm(range(num_data)):
            datum, target = data.getNonRepetitiveData((ran_num + i) % num_data,type=type) 
            # pdb.set_trace()
            datum = datum.reshape(1,datum.shape[0],datum.shape[1],datum.shape[2])
            datum, target = torch.from_numpy(datum).float(), torch.from_numpy(target).long()
            datum, target = datum.to(device), target.to(device)
            output = model(datum)
            # print(output)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= num_data
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, num_data,
        100. * correct / num_data))
        
def main():
    #1. hyper-parameter
    batch_size = 64
    epochs = 14
    lr = 2.0
    gamma = 0.7
    torch.manual_seed(1)
    log_interval = 10
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")
    print(device)
    #2. network definition
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma) #tune the lr after each epoch
    #3. data initilization
    opit = dataset_operator()
    data = buildTrainValtest(opit.datatable, opit.labeltable)
    # its temporary
    data.cutintopieces(flag=3)
    data.generatorSetting(batch_size=batch_size)
    num_data = sum(data.setCounting(data.trainPieces))
    for epoch in range(epochs):
        yield_data = data.data_genetator()
        train(model, device, yield_data, optimizer, epoch, data.iteration_per_epoch)
        test(model, device, data, 'test')
        scheduler.step()

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                
                
                
                
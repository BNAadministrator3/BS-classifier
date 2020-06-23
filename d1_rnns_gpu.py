import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pdb

torch.manual_seed(1)    # reproducible
# Hyper Parameters
EPOCH = 5               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.01               # learning rate
DOWNLOAD_MNIST = False   # set to True if haven't download the data


# Mnist digital dataset
train_data = dsets.MNIST(
    root='/home/zhaok14/data/',
    train=True,                         # this is training data
    transform=transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,            # download it if you don't have it
)

# plot one example
# print(train_data.train_data.size())     # (60000, 28, 28)
# print(train_data.train_labels.size())   # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

# Data Loader for easy mini-batch return in training
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# convert test data into Variable, pick 2000 samples to speed up testing
test_data = dsets.MNIST(root='/home/zhaok14/data/', train=False, transform=transforms.ToTensor())
test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255.   # shape (2000, 28, 28) value in range(0,1)
test_y = test_data.test_labels.numpy()[:2000]    # covert to numpy array

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        # pdb.set_trace()
        # choose r_out at the last time step
        # pdb.set_trace()
        out = self.out(r_out[:, -1, :])
        # out = self.out(h_n[0])
        return out


if __name__ == '__main__':
    use_cuda = True
    device = torch.device("cuda:1" if torch.cuda.is_available() and use_cuda else "cpu")
    print(device)
    
    rnn = RNN().to(device)
    print(rnn)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

    # training and testing
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data
            b_x = b_x.view(-1, 28, 28)              # reshape x to (batch, time_step, input_size)
            # pdb.set_trace()
            b_x, b_y = b_x.to(device), b_y.to(device)
            # pdb.set_trace()
            output = rnn(b_x)                               # rnn output
            loss = loss_func(output, b_y)                   # cross entropy loss
            optimizer.zero_grad()                           # clear gradients for this training step
            loss.backward()                                 # backpropagation, compute gradients
            optimizer.step()                                # apply gradients

            if step % 50 == 0:
                test_output = rnn( test_x.to(device) )                   # (samples, time_step, input_size)
                pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
                accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)

    # print 10 predictions from test data
    test_output = rnn(test_x[:10].view(-1, 28, 28).to(device))
    pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
    print(pred_y, 'prediction number')
    print(test_y[:10], 'real number')


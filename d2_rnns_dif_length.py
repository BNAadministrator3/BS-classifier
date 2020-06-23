import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pdb
from c2_pytorch_datasetting import BSdataset, collate_fn
import torch.nn.utils.rnn as rnn_utils

torch.manual_seed(1)    # reproducible
# Hyper Parameters
EPOCH = 5               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 26         # rnn input size / image width
LR = 0.01               # learning rate
DOWNLOAD_MNIST = False   # set to True if haven't download the data



# new 1.
data = BSdataset(train=True)
self_x, self_y = data.packed_separate()
train_loader = torch.utils.data.DataLoader(data, batch_size=3, shuffle=True, 
                             collate_fn=collate_fn)
tests = BSdataset(train=False)
test_x,test_y = tests.packed_separate()
# new 2.
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(64, 2)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        
        # print("r_out's size: ",r_out.size())
        # print("h_n's size: ",h_n.size())
        # print("h_c's size: ",h_c.size())
        # pdb.set_trace()
        # choose r_out at the last time step
        # out = self.out(r_out[:, -1, :])
        out = self.out(h_n[0])
        return out

# new 3.
if __name__ == '__main__':
    use_cuda = True
    device = torch.device("cuda:1" if torch.cuda.is_available() and use_cuda else "cpu")
    # device = "cpu"
    print('Training device: ',device)
    rnn = RNN().to(device)
    # print(rnn)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
    # training and testing
    for epoch in range(EPOCH):
        for step, (b_x, b_y, b_l) in enumerate(train_loader):        # gives batch data
            # pdb.set_trace()
            b_x = rnn_utils.pack_padded_sequence(b_x, b_l, batch_first=True)
            # b_x = b_x.view(-1, 28, 28)              # reshape x to (batch, time_step, input_size)
            b_x, b_y = b_x.to(device), b_y.to(device)
            output = rnn(b_x)                               # rnn output
            # pdb.set_trace()
            loss = loss_func(output, b_y)                   # cross entropy loss
            optimizer.zero_grad()                           # clear gradients for this training step
            loss.backward()                                 # backpropagation, compute gradients
            optimizer.step()                                # apply gradients

            if step % 50 == 0:
                # pdb.set_trace()
                # training accuracy
                self_output = rnn( self_x.to(device) )                   # (samples, time_step, input_size)
                pred_y = torch.max(self_output, 1)[1].data.cpu().numpy()
                self_accu = float((pred_y == self_y).astype(int).sum()) / float(self_y.size)
                # test accuracy
                test_output = rnn( test_x.to(device) )                   # (samples, time_step, input_size)
                pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
                test_accu = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(),'| training accuracy: %.2f' % self_accu, '| test accuracy: %.2f' % test_accu)

    # print 10 predictions from test data
    test_output = rnn(test_x[:10].to(device))
    pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
    print(pred_y, 'prediction number')
    print(test_y[:10], 'real number')


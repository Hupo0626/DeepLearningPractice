import torch
import torch.nn as nn
import torchvision
from MyCNN import MyCNN
import numpy as np
import datetime, json
from tensorboardX import SummaryWriter
class MyModel:
    '''
    Class where learning happends
    '''

    def __init__(self, args):
        self.args = args

        self.network = MyCNN()

        self.data_set = {
            'TRAIN': {
                'X': torchvision.datasets.MNIST(root='~/DataSet', train=True).train_data,
                'Y': torchvision.datasets.MNIST(root='~/DataSet', train=True).train_labels
            },
            'TEST': {
                'X': torchvision.datasets.MNIST(root='~/DataSet', train=False).test_data,
                'Y': torchvision.datasets.MNIST(root='~/DataSet', train=False).test_labels
            }
        }

        self.optimizer = {
            'adam': torch.optim.Adam(self.network.parameters(), lr=args['alpha']),
            'adadelta': torch.optim.Adadelta(self.network.parameters(), lr=args['alpha']),
            'rms': torch.optim.RMSprop(self.network.parameters()),
            'adagrad': torch.optim.Adagrad(self.network.parameters(), lr=args['alpha'])
        }.get(self.args['optimizer_name'], torch.optim.SGD(self.network.parameters(), lr=args['alpha']))

        self.loss_function = nn.CrossEntropyLoss()

        self.logger = SummaryWriter('logs/log_' + self.args['tag'] + '_' +
                                    datetime.datetime.now().strftime('%D-%T').
                                    replace('/', '_') + json.dumps(args))
        self.logger.add_graph(self.network, torch.zeros(self.args['batch_size'], 1, 28, 28), False)


    def fetch_batch(self, phase='TRAIN'):
        '''
        Fetch batch.
        Fetch batch_size numbers of data here.
        :param phase: what dataset we need to batch, default = 'TRAIN'
        :return: batch_size numbers of data with random choice
        '''
        data_size = self.data_set[phase]['X'].shape[0]
        idx = np.random.choice(data_size, self.args['batch_size'], replace=False)

        return self.data_set[phase]['X'][idx], self.data_set[phase]['Y'][idx]


    def data_preprocess(self, data):
        '''
        Data preprocess.
        Jobs done here should be different from what may be done inside customized DataLoader,
        in that some preprocess jobs may need information of input or network,
        which can't be done using just data file.
        '''

        '''
        data: numpy array with the shape of -1,28,28
        return: transformed data (train/test) with shape -1,1,28,28
        '''
        raw_shape = data.shape
        data = data.reshape((raw_shape[0], 1, 28, 28))
        data = data.float().numpy()
        data = (data-np.mean(data, axis=(1, 2, 3), keepdims=True))/np.std(data, axis=(1, 2, 3), keepdims=True)
        return torch.Tensor(data)


    def calculate_loss(self, batch_y_, batch_y):
        '''
        This is where loss is calculated
        '''
        # print(batch_y_.shape,batch_y.shape)
        loss = self.loss_function(batch_y_, batch_y)
        return loss

    def run(self):

        for epoch_id in range(self.args['epoch_num']):

            for phase in ('TRAIN', 'TEST'):
                if phase == 'TRAIN':
                    self.optimizer.zero_grad()
                    train = True
                    self.network.train(True)

                else:
                    train = False
                    self.network.train(False)

                batch_num = self.data_set[phase]['X'].shape[0]//self.args['batch_size']
                for batch_id in range(batch_num):
                    batch_x, batch_y = self.fetch_batch(phase)
                    batch_x = self.data_preprocess(batch_x.float())
                    batch_y_ = self.network(batch_x)
                    loss = self.calculate_loss(batch_y_,batch_y)
                    # print(loss.shape,loss.data,type(loss.data))
                    self.logger.add_scalar(phase + 'loss/', loss.data,
                                           epoch_id * batch_num + batch_id)

                    # print(loss.data[0])
                    if train:
                        loss.backward()

                        for tag, value in self.network.named_parameters():
                            tag = tag.replace('.', '/')
                            self.logger.add_histogram(
                                tag,
                                value.data.cpu().numpy(),
                                epoch_id * batch_num + batch_id)
                            if hasattr(value.grad, 'data'):
                                self.logger.add_histogram(
                                    tag + '/grad',
                                    value.grad.data.cpu().numpy(),
                                    epoch_id * batch_num + batch_id)

                        self.optimizer.step()


    def test(self):
        self.run()
        #
        # batch_x, batch_y = self.fetch_batch('TRAIN')
        # print(batch_x.shape,batch_y.shape)
        # batch_x = self.data_preprocess(batch_x)
        #
        # batch_y_ = self.network(batch_x)
        #
        # print(batch_x,batch_y,batch_y_)
        # data=torch.Tensor(np.array(np.random.randint(0,100,size=(10,28,28))))
        # print(data.shape)
        # _data=self.data_preprocess(data)
        # print(_data.shape,_data[0].mean(),_data[0].var())
        # ape, y.shape, type(x), type(y))


if __name__ == '__main__':
    args = {
        'epoch_num': 10,
        'batch_size': 128,  # (N,...)
        'alpha': 0.00006,
        'optimizer_name': 'sgd',
        'tag': 'test',
        'resume': False
    }

    M = MyModel(args)
    M.test()

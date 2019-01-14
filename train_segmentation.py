from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
# from datasets import PartDataset
from pointnet import PointNetDenseCls
import torch.nn.functional as F
import _pickle as pkl
import pandas as pd
import glob


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')


opt = parser.parse_args()
print (opt)

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = False, class_choice = ['Chair'])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = False, class_choice = ['Chair'], train = False)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

data_path = '/home/sai/data/maplite_data_chunks'

### getting list of the data
data_list = glob.glob(data_path+'/*.pkl')
data_list.sort()

### Number of inputs 
data_length = len(data_list)

### Getting train and test data in random order
indices = np.random.permutation(data_length)

### Dividing training and test set in some ratio
### For 1000 points, 250 points for training and 750 points for testing, then ratio is 1:3
train_part = 1
test_part = 2

### Defining criterion for the neural network
# criterion = criterion

### getting train and test indices(randomly)
train_indices = indices[0:int(train_part*data_length/(train_part + test_part))]
test_indices = indices[int(train_part*data_length/(train_part + test_part)):-1]

df = pd.read_pickle(data_list[0])

def normalize_input(df):

    ### Getting coordinates and features
    x = df.iloc[0]['scan_utm']['x'] 
    y = df.iloc[0]['scan_utm']['y']
    z = df.iloc[0]['scan_utm']['z'] 

    ### getting coordinate values between 0 and 150
    x -= min(x)
    y -= min(y)
    z -= min(z)
    x = 150*x/max(x)
    y = 150*y/max(y) 
    z = 150*z/max(z) 

    ### Normalizing features?? Not decided yet

    ### final train and test data
    coords = torch.randn(len(x), 4)
    coords[:,0] = torch.from_numpy(x.copy())
    coords[:,1] = torch.from_numpy(y.copy())
    coords[:,2] = torch.from_numpy(z.copy())
    coords[:,3] = torch.from_numpy(df.iloc[0]['scan_utm']['intensity'].copy())
    del x, y, z
    train_output = torch.from_numpy(1*(df.iloc[0]['is_road_truth'] == True))
    return coords[0:100,:], train_output[0:100]

print(len(dataset), len(test_dataset))
num_classes = dataset.num_seg_classes
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x:'\033[94m' + x + '\033[0m'


classifier = PointNetDenseCls(k = num_classes)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
classifier.cuda()

num_batch = len(dataset)/opt.batchSize

for epoch in range(opt.nepoch):
    # for i in range(50):
    for i, data in enumerate(dataloader):
        # points, target = data
        points = np.zeros((2,100,4))
        target = np.zeros((2,100))
        df = pd.read_pickle(data_list[i])

                ## Prepare data for training
        points[0,:,:], target[0,:] = normalize_input(df)
        # points = np.expand_dims(points, axis=0)
        # target = np.expand_dims(target, axis=0)
        # df = pd.read_pickle(data_list[i])
        # points[1,:,:], target[1,:] = normalize_input(df)
        print("points.shape, target.shape = ",points.shape, target.shape)
        print(points.shape, target.shape)
        points  = Variable(torch.FloatTensor(points)).cuda()
        target =  Variable(torch.LongTensor(target)).cuda()
        points = points.transpose(2,1) 
        points, target = points.cuda(), target.cuda()   
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, _ = classifier(points)
        pred = pred.view(-1, num_classes)
        target = target.view(-1,1)[:,0] - 1
        #print(pred.size(), target.size())
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' %(epoch, i, num_batch, loss.item(), correct.item()/float(opt.batchSize * 2500)))
        
        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            points, target = Variable(points), Variable(target)
            points = points.transpose(2,1) 
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _ = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1,1)[:,0] - 1

            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize * 2500)))
    
    torch.save(classifier.state_dict(), '%s/seg_model_%d.pth' % (opt.outf, epoch))
import random
from random import shuffle
import numpy as np
# import ipdb
import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import model.meta.learner as Learner
import model.meta.modelfactory as mf
from scipy.stats import pearsonr
import datetime

class BaseNet(torch.nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,           
                 args):
        super(BaseNet, self).__init__()

        self.args = args
        nl, nh = args.n_layers, args.n_hiddens

        config = mf.ModelFactory.get_model(model_type = args.arch, sizes = [n_inputs] + [nh] * nl + [n_outputs],
                                                dataset = args.dataset, args=args)

        self.net = Learner.Learner(config, args)

        # define the lr params
        self.net.define_task_lr_params(alpha_init = args.alpha_init)

        self.opt_wt = torch.optim.SGD(list(self.net.parameters()), lr=args.opt_wt)     
        self.opt_lr = torch.optim.SGD(list(self.net.alpha_lr.parameters()), lr=args.opt_lr) 

        self.epoch = 0
        # allocate buffer
        self.M = []        
        self.M_new = []
        self.age = 0

        # setup losses
        self.loss = torch.nn.CrossEntropyLoss()
        self.is_cifar = ((args.dataset == 'cifar100') or (args.dataset == 'tinyimagenet'))
        self.glances = args.glances
        self.pass_itr = 0
        self.real_epoch = 0

        self.current_task = 0
        self.memories = args.memories
        self.batchSize = int(args.replay_batch_size)

        self.cuda = args.cuda
        if self.cuda:
            self.net = self.net.cuda()

        self.n_outputs = n_outputs

    def push_to_mem(self, batch_x, batch_y, t):
        """
        Reservoir sampling to push subsampled stream
        of data points to replay/memory buffer
        """

        if(self.real_epoch > 0 or self.pass_itr>0):
            return
        batch_x = batch_x.cpu()
        batch_y = batch_y.cpu()              
        t = t.cpu()

        for i in range(batch_x.shape[0]):
            self.age += 1
            if len(self.M_new) < self.memories:
                self.M_new.append([batch_x[i], batch_y[i], t])
            else:
                p = random.randint(0,self.age)  
                if p < self.memories:
                    self.M_new[p] = [batch_x[i], batch_y[i], t]


    def getBatch(self, x, y, t, batch_size=None):
        """
        Given the new data points, create a batch of old + new data, 
        where old data is sampled from the memory buffer
        """
        # numpy-ize the data{x,y,t}
        if x is not None: mxi, myi, mti = np.array(x),              np.array(y),              np.ones(x.shape[0], dtype=int)*t
        # if no data, then create empty numpy array
        else: mxi, myi, mti = np.empty( shape=(0, 0) ), np.empty( shape=(0, 0) ), np.empty( shape=(0, 0) )

        # might store the data into lists
        bxs, bys, bts = [], [], []

        # use from old memory or new memory
        if self.args.use_old_task_memory and t>0: MEM = self.M
        else: MEM = self.M_new

        # use self.batch_size if necessary
        batch_size = self.batchSize if batch_size is None else batch_size

        # if there is anything in memory
        if len(MEM) > 0:
            # order = {0,1,2...,len(MEM)-1}
            order = [i for i in range(len(MEM))]
            # run the loop until minm MEM or batch_siz
            osize = min(batch_size,len(MEM))
            for j in range(osize):
                # randomly shuffle the order list
                shuffle(order)
                # get random tuples of {x,y,t}
                x,y,t = MEM[order[j]]
                
                # numpy-ize the data tuple
                xi, yi, ti = np.array(x), np.array(y), np.array(t)

                # store the data tuple into the lists
                bxs.append(xi)
                bys.append(yi)
                bts.append(ti)

        # b_lists <= add_to <= m_lists
        for j in range(len(myi)):
            bxs.append(mxi[j])
            bys.append(myi[j])
            bts.append(mti[j])

        # b_lists <= torch-ize
        bxs = Variable(torch.from_numpy(np.array(bxs))).float() 
        bys = Variable(torch.from_numpy(np.array(bys))).long().view(-1)
        bts = Variable(torch.from_numpy(np.array(bts))).long().view(-1)
        
        # b_lists <= cuda-ize
        if self.cuda:
            bxs = bxs.cuda()
            bys = bys.cuda()
            bts = bts.cuda()

        return bxs,bys,bts

    def compute_offsets(self, task):
        # mapping from classes [1-100] to their idx within a task
        offset1 = task * self.nc_per_task
        offset2 = (task + 1) * self.nc_per_task
        return int(offset1), int(offset2)

    def zero_grads(self):
        if self.args.learn_lr: self.opt_lr.zero_grad()
        self.opt_wt.zero_grad()
        self.net.zero_grad()
        self.net.alpha_lr.zero_grad()

import random
import numpy as np
# import ipdb
import math

import torch
import torch.nn as nn
from model.lamaml_base import *


# a subset of BaseNet containing the model, parameters, forward, backward, differentiation,etc..
# BaseNet comes from "lamaml_base.py" 
class Net(BaseNet):

    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__(n_inputs, n_outputs, n_tasks, args)

        self.nc_per_task = n_outputs

    # trivial forward
    def forward(self, x, t):
        # push x thru model and get y_pred
        output = self.net.forward(x)
        return output

    # get loss and y_pred normally
    def meta_loss(self, x, fast_weights, y, t):
        """
        differentiate the loss through the network updates wrt alpha
        """
        # simply pushing the x forward thru the net
        # fast_weights doesn't seem to be getting used
        logits = self.net.forward(x, fast_weights)
        # get loss as usual
        loss_q = self.loss(logits.squeeze(1), y)
        # return the loss and the output y_pred
        return loss_q, logits

    # return an update of the fast weights
    def inner_update(self, x, fast_weights, y, t):
        """
        Update the fast weights using the current samples and return the updated fast
        """
        # same usual finding y_pred
        logits = self.net.forward(x, fast_weights)
        # same usual finding loss without squeezing (i wonder why?)
        loss = self.loss(logits, y)   

        # if no fast_weights use the params from the net
        if fast_weights is None: fast_weights = self.net.parameters() 

        # NOTE if we want higher order grads to be allowed, change create_graph=False to True
        graph_required = self.args.second_order
        # get grads of loss w.r.t fast_weights. if graph_req is True, it will retain the graph for future runs as well
        grads = torch.autograd.grad(loss, fast_weights, create_graph=graph_required, retain_graph=graph_required)

        # clipping the grads in-between certain values
        for i in range(len(grads)):
            torch.clamp(grads[i], min = -self.args.grad_clip_norm, max = self.args.grad_clip_norm)

        # update fast_weights manually
        fast_weights = list(
                map(lambda p: p[1][0] - p[0] * nn.functional.relu(p[1][1]), zip(grads, zip(fast_weights, self.net.alpha_lr))))
        return fast_weights

    # idk??
    def observe(self, x, y, t):
        # initialize for training, make use of batch norms, dropouts,etc..
        self.net.train() 

        # glances??
        for pass_itr in range(self.glances):
            # pass_itr is to be used in other funcs(of this class) ig
            self.pass_itr = pass_itr
            
            # Returns a random permutation of integers from 0 to n - 1
            perm = torch.randperm(x.size(0))
            # selecting a random data tuple (x, y)
            x, y = x[perm], y[perm]
            
            # so each it of this loop is an epoch
            self.epoch += 1
            # set {opt_lr, opt_wt, net, net.alpha_lr} (all 4) as zero_grads
            self.zero_grads()

            # current_task=??
            if t != self.current_task:
                # M=??
                self.M = self.M_new
                self.current_task = t

            # get batch_size from the shape of x
            batch_sz = x.shape[0]
            # will want to store batch loss in a list
            meta_losses = [0 for _ in range(batch_sz)] 

            # b_lisst <= {x,y,t} + sample(Memory)
            bx, by, bt = self.getBatch(x.cpu().numpy(), y.cpu().numpy(), t)
            fast_weights = None
            
            # for each tuple in batch
            for i in range(batch_sz):
                # squeeze the tuples
                batch_x, batch_y = x[i].unsqueeze(0), y[i].unsqueeze(0)
                
                # do an inner update
                fast_weights = self.inner_update(batch_x, fast_weights, batch_y, t)

                # if real_epoch is zero, push the tuple to memory
                if(self.real_epoch == 0): self.push_to_mem(batch_x, batch_y, torch.tensor(t))

                # get meta_loss and y_pred
                meta_loss, logits = self.meta_loss(bx, fast_weights, by, t) 
                # collect meta_losses into a list
                meta_losses[i] += meta_loss
    
            # Taking the meta gradient step (will update the learning rates)
            self.zero_grads()

            # get avg of the meta_losses
            meta_loss = sum(meta_losses)/len(meta_losses)

            # do bkwrd
            meta_loss.backward()

            # clip the grads of the theta and alpha
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(self.net.alpha_lr.parameters(), self.args.grad_clip_norm)

            # update alpha learning rate
            if self.args.learn_lr: self.opt_lr.step()

            # update theta learning rate
            if(self.args.sync_update): self.opt_wt.step()
            else:
                # update the parameters in place
                for i,p in enumerate(self.net.parameters()):
                    p.data = p.data - p.grad * nn.functional.relu(self.net.alpha_lr[i])

            # set zero_grad for net and alpha learning rate
            self.net.zero_grad()
            self.net.alpha_lr.zero_grad()

        return meta_loss.item()
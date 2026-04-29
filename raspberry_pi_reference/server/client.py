#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 11:47:25 2023

@author: user
"""

import torch.nn.functional as F
import torch
import time
import paho.mqtt.client as mqtt
import base64
import pickle

class Client(object):
    r"""Implements one clients
    """
    
    def __init__(
            self, model,
            criterion,
            metric,
            device,
            optimizer,
            train_iterator, 
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            lr_scheduler=None,
            is_binary_classification=False
                ):

        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.metric = metric
        self.device = device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.is_binary_classification = is_binary_classification

       

        self.counter=0
        self.logger=logger
        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.test_iterator = test_iterator

        self.train_loader = iter(self.train_iterator)

        self.n_train_samples = len(self.train_iterator)
        self.n_val_samples = len(self.val_iterator)
        self.n_test_samples = len(self.test_iterator)
        self.local_steps = local_steps
        
        self.global_model = None
        # Other initialization code for your FLClient



    def step(self, single_batch_flag=False, *args, **kwargs):
        """
        perform on step for the client

        """
        self.counter+=1
        n_epochs=self.local_steps
        for step in range(n_epochs):
            self.fit_epoch(iterator=self.train_iterator)

        
        
 
    # training 
    def fit_epoch(self, iterator, weights=None):
        """
        perform several optimizer steps on all batches drawn from `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss.detach()
            metric.detach()

        """
        self.model.train()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, indices in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            n_samples += y.size(0)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)

            self.optimizer.zero_grad()

            y_pred = self.model(x)

            loss_vec = self.criterion(y_pred, y)
            if weights is not None:
                weights = weights.to(self.device)
                loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
            else:
                loss = loss_vec.mean()
            loss.backward()

            self.optimizer.step()

            global_loss += loss.detach() * loss_vec.size(0)
            global_metric += self.metric(y_pred, y).detach()

        return global_loss / n_samples, global_metric / n_samples

       

    
    # write logs

    def write_logs(self):

   
        
        train_loss, train_acc = self.evaluate_iterator(self.val_iterator)
        test_loss, test_acc = self.evaluate_iterator(self.test_iterator)

        #self.logger.add_scalar("Train/Loss", train_loss, self.counter)
        #self.logger.add_scalar("Train/Metric", train_acc, self.counter)
        #self.logger.add_scalar("Test/Loss", test_loss, self.counter)
        #self.logger.add_scalar("Test/Metric", test_acc, self.counter)

        return train_loss, train_acc, test_loss, test_acc
        



    def evaluate_iterator(self, iterator):
        """
        Evaluate a ensemble of learners on iterator.

        :param iterator: yields x, y, indices
        :type iterator: torch.utils.data.DataLoader
        :return: global_loss, global_acc

        """

        self.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, _ in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)

            with torch.no_grad():
                y_pred = self.model(x)

                global_loss += self.criterion(y_pred, y).sum().detach()
                global_metric += self.metric(y_pred, y).detach()

            n_samples += y.size(0)

        return global_loss / n_samples, global_metric / n_samples




    

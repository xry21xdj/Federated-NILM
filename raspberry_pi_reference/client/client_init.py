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
            local_iterations,
            client_id, 
            broker_address, 
            broker_port, 
            global_topic,
            client_topic,
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
        self.K = local_iterations
        self.logger=logger
        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.test_iterator = test_iterator

        self.train_loader = iter(self.train_iterator)

        self.n_train_samples = len(self.train_iterator)
        self.n_val_samples = len(self.val_iterator)
        self.n_test_samples = len(self.test_iterator)
        self.local_steps = local_steps
        
        self.client = mqtt.Client(client_id)
        self.broker_address = broker_address
        self.broker_port =broker_port

        self.global_topic = global_topic
        self.client_topic = client_topic
        self.global_model = None
        self.buffer = None
        self.old_global_model = None
        self.timestamp = []


        self.recieved_model_flag=0
        # Other initialization code for your FLClient

    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected with result code {rc}")
        client.subscribe(self.global_topic,qos=2)

    def on_message(self, client, userdata, msg):
        if msg.topic ==self.global_topic:
            global_model_info = pickle.loads(msg.payload)
            #print('global_model info', global_model_info)
            timestamp = global_model_info['timestamp']
            new_global_model=  global_model_info['state_dict']
            buffer = global_model_info['buffer']
            
            if timestamp != self.timestamp[-1]:
                print('*******time stamp=', timestamp)
                self.global_model = new_global_model
                self.timestamp.append(timestamp)
                self.buffer = buffer
                # set the flag to 1 if recieved the model
                self.recieved_model_flag=1


    # send the global_model_dict
    # some algorithm may need to send the optimizer state dict to the server
    def send_local_model(self, local_model_state_dict, local_optimizer_state_dict=None,old_state_dict=None):
        serialized_model = pickle.dumps({'state_dict':local_model_state_dict,
                                        'optimizer_state_dict':local_optimizer_state_dict,
                                        'K':self.K,
                                        'old_state_dict':old_state_dict})
        
        self.client.publish(self.client_topic, payload=serialized_model, qos=2)


    # update the local model using the global information
    def update_local_model(self):


        # load the global model
        self.model.load_state_dict(self.global_model)

        # for fedNAG
        if callable(getattr(self.optimizer, "set_initial_buffer_FedNAG", None)):
            self.optimizer.set_initial_buffer_FedNAG(self.buffer)
        
        # for FastSlowMo
        if callable(getattr(self.optimizer, "set_initial_buffer_FastSlowMo", None)):
            self.optimizer.set_initial_buffer_FastSlowMo(self.buffer)

        # for DOMO
        if callable(getattr(self.optimizer, "set_initial_buffer_DOMO", None)):
            self.optimizer.set_initial_buffer_DOMO(self.buffer)

        # for MIME
        if callable(getattr(self.optimizer, "set_initial_buffer_MIME", None)):
            self.optimizer.set_initial_buffer_MIME(self.buffer)

        # if callable(getattr(self.optimizer, "set_initial_params", None)):
        #     learner.optimizer.set_initial_params(
        #         self.global_learners_ensemble[learner_id].model.parameters(), learner.device
        #     )
            
        # '''for FedAvg-M or FedM, FedMGDA-M, FedMGDA-M-Mom, FedAvg-M-Mom'''
        if callable(getattr(self.optimizer, "set_initial_grad", None)):
            self.optimizer.set_initial_grad(self.buffer)   
            

  


    def start(self):
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(self.broker_address, self.broker_port, 60)
        self.client.loop_start()

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

        self.logger.add_scalar("Train/Loss", train_loss, self.counter)
        self.logger.add_scalar("Train/Metric", train_acc, self.counter)
        self.logger.add_scalar("Test/Loss", test_loss, self.counter)
        self.logger.add_scalar("Test/Metric", test_acc, self.counter)

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


    def compute_full_gradients(self, iterator):
        self.model.train()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0
        self.optimizer.zero_grad()
        for x, y, indices in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            n_samples += y.size(0)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)

           

            y_pred = self.model(x)

            loss_vec = self.criterion(y_pred, y)

            loss = loss_vec.mean()
            loss.backward()

    

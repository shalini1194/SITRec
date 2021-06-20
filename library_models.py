'''
This is a supporting library with the code of the model.
'''

from __future__ import division
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import sys
from collections import defaultdict
import os
import pickle
#import gpustat
from itertools import chain
from tqdm import tqdm, trange, tqdm_notebook, tnrange
import csv

PATH = "./"

total_reinitialization_count = 0
# A NORMALIZATION LAYER
class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)


class SITRec(nn.Module):
    def __init__(self, args, num_features, num_users, num_items):
        super(SITRec,self).__init__()

        #print("*** Initializing the  model ***")
        self.embedding_dim = args.embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.user_static_embedding_size = num_users
        self.item_static_embedding_size = num_items

        #print("Initializing user and item embeddings")
        self.initial_user_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))
        self.initial_item_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))

        rnn_input_size_items = rnn_input_size_users = self.embedding_dim + 1

        #print( "Initializing user and item RNNs")
        self.item_rnn = nn.RNNCell(rnn_input_size_users, self.embedding_dim)
        self.user_rnn = nn.RNNCell(rnn_input_size_items, self.embedding_dim)

        #print ("Initializing linear layers")
        self.linear_layer1 = nn.Linear(self.embedding_dim, 50)
        self.linear_layer2 = nn.Linear(50, 2)
        self.linear_layer3 = nn.Linear(3*self.embedding_dim, self.embedding_dim)
        self.linear_layer4 = nn.Linear(num_features,self.embedding_dim)
        self.linear_layer5 = nn.Linear(4*self.embedding_dim, self.embedding_dim)
        self.prediction_layer = nn.Linear(self.user_static_embedding_size + self.item_static_embedding_size +2* self.embedding_dim , self.item_static_embedding_size + self.embedding_dim)
        self.embedding_layer = NormalLinear(1, self.embedding_dim)
        self.embedding_layer2 = NormalLinear(1, self.embedding_dim)
        self.embedding_layer3 = NormalLinear(1, self.embedding_dim)
        self.embedding_layer4 = NormalLinear(1,  self.embedding_dim)
        self.embedding_layer5 = NormalLinear(1, 1)

        #print( "***  Initialization complete ***\n\n")
        
    def forward(self, user_embeddings, item_embeddings,timediffs=None, features=None, select=None):
        if select == 'item_update':
            input1 = torch.cat([user_embeddings, timediffs], dim=1)

            item_embedding_output = self.item_rnn(input1, item_embeddings)

            return F.normalize(item_embedding_output)

        elif select == 'user_update':
            input2 = torch.cat([item_embeddings, timediffs], dim=1)
            user_embedding_output = self.user_rnn(input2, user_embeddings)
            return F.normalize(user_embedding_output)


    def context_convert(self, embeddings, timediffs, features):

        new_embeddings = embeddings * (1 + self.embedding_layer(timediffs))
        return new_embeddings
    def project_user(self, user_embeddings,feature_syllabus, timediffs=None, features=None):

        new_feature = (self.linear_layer3(torch.cat([user_embeddings, (1 + self.embedding_layer(timediffs)), feature_syllabus], dim=1)))

        return new_feature

    def item_decay(self, item_embeddings, user_embeddings,  posted_on, replied_on, user_item_time_diffs,item_timediff):


        exp_user_time_diffs_emb = torch.exp(-1*user_item_time_diffs.unsqueeze(1))
        exp_item_time_diffs_emb = torch.exp(-0.2*item_timediff.unsqueeze(1))

        factor = (posted_on.unsqueeze(1)+ replied_on.unsqueeze(1)+1)* exp_user_time_diffs_emb* exp_item_time_diffs_emb


        new_embeddings = (item_embeddings+user_embeddings*factor)/(factor+1)
        return new_embeddings


    def predict_label(self, user_embeddings):
        X_out = nn.ReLU()(self.linear_layer1(user_embeddings))
        X_out = self.linear_layer2(X_out)
        return X_out

    def predict_item_embedding(self, user_embeddings):
        X_out = self.prediction_layer(user_embeddings)
        return X_out
    def topic_input(self, current_topics):
        return self.linear_layer4(current_topics)


# INITIALIZE T-BATCH VARIABLES
def reinitialize_tbatches():
    global current_tbatches_interactionids, current_tbatches_user, current_tbatches_item, current_tbatches_timestamp, current_tbatches_feature, current_tbatches_label, current_tbatches_previous_item
    global tbatchid_user, tbatchid_item, current_tbatches_user_timediffs, current_tbatches_item_timediffs, current_tbatches_user_timediffs_next, current_tbatches_negative_sample
    global current_tbatches_posted_on, current_tbatches_replied_on, current_tbatches_user_item_timediff, current_tbatches_item_timediff_sequence
    # list of users of each tbatch up to now
    current_tbatches_interactionids = defaultdict(list)
    current_tbatches_user = defaultdict(list)
    current_tbatches_item = defaultdict(list)
    current_tbatches_timestamp = defaultdict(list)
    current_tbatches_feature = defaultdict(list)
    current_tbatches_label = defaultdict(list)
    current_tbatches_previous_item = defaultdict(list)
    current_tbatches_user_timediffs = defaultdict(list)
    current_tbatches_item_timediffs = defaultdict(list)
    current_tbatches_user_timediffs_next = defaultdict(list)
    current_tbatches_negative_sample= defaultdict(list)
    current_tbatches_posted_on = defaultdict(list)
    current_tbatches_replied_on = defaultdict(list)
    current_tbatches_user_item_timediff = defaultdict(list)
    current_tbatches_item_timediff_sequence = defaultdict(list)
    # the latest tbatch a user is in
    tbatchid_user = defaultdict(lambda: -1)

    # the latest tbatch a item is in
    tbatchid_item = defaultdict(lambda: -1)

    global total_reinitialization_count
    total_reinitialization_count +=1


# CALCULATE LOSS FOR THE PREDICTED USER STATE 
def calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_time_series, y_true, loss_function):
    # PREDCIT THE LABEL FROM THE USER DYNAMIC EMBEDDINGS
    prob = model.predict_label(user_embeddings_time_series[tbatch_interactionids,:])
    y = Variable(torch.LongTensor(y_true).cuda()[tbatch_interactionids])
    
    loss = loss_function(prob, y)

    return loss


# SAVE TRAINED MODEL TO DISK
def save_model(model, optimizer, args, epoch, user_embeddings, item_embeddings, train_end_idx, topic_transaction, topic_transaction2,item_transaction,user_transaction, user_embeddings_time_series=None, item_embeddings_time_series=None, path=PATH):
    print ("*** Saving embeddings and model ***")
    state = {
            'user_embeddings': user_embeddings.data.cpu().numpy(),
            'item_embeddings': item_embeddings.data.cpu().numpy(),
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'train_end_idx': train_end_idx,
            'topic_transaction': topic_transaction.cpu().numpy(),
            'user_transaction': user_transaction.cpu().numpy(),
            'item_transaction' : item_transaction.cpu().numpy(),
            'topic_transaction2':topic_transaction2.cpu().numpy()
            }

    if user_embeddings_time_series is not None:
        state['user_embeddings_time_series'] = user_embeddings_time_series.data.cpu().numpy()
        state['item_embeddings_time_series'] = item_embeddings_time_series.data.cpu().numpy()

    directory = os.path.join(path, 'saved_models/%s' % args.network)
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.join(directory, "checkpoint.ep%d.tp%d.tar" % (epoch, args.embedding_dim))
    torch.save(state, filename)
    #print( "*** Saved embeddings and model to file: %s ***\n\n" % filename)


# LOAD PREVIOUSLY TRAINED AND SAVED MODEL
def load_model(model, optimizer, args, epoch):

    filename = PATH + "saved_models/%s/checkpoint.ep%d.tp%d.tar" % (args.network, epoch, args.embedding_dim)
    checkpoint = torch.load(filename)
    args.start_epoch = checkpoint['epoch']
    user_embeddings = Variable(torch.from_numpy(checkpoint['user_embeddings']).cuda())
    item_embeddings = Variable(torch.from_numpy(checkpoint['item_embeddings']).cuda())

    try:
        train_end_idx = checkpoint['train_end_idx'] 
    except KeyError:
        train_end_idx = None

    try:
        user_embeddings_time_series = Variable(torch.from_numpy(checkpoint['user_embeddings_time_series']).cuda())
        item_embeddings_time_series = Variable(torch.from_numpy(checkpoint['item_embeddings_time_series']).cuda())
    except:
        user_embeddings_time_series = None
        item_embeddings_time_series = None

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return [model, optimizer, user_embeddings, item_embeddings, user_embeddings_time_series, item_embeddings_time_series, train_end_idx]


# SET USER AND ITEM EMBEDDINGS TO THE END OF THE TRAINING PERIOD 
def set_embeddings_training_end(user_embeddings, item_embeddings, user_embeddings_time_series, item_embeddings_time_series, user_data_id, item_data_id, train_end_idx):
    userid2lastidx = {}
    for cnt, userid in enumerate(user_data_id[:train_end_idx]):
        userid2lastidx[userid] = cnt
    itemid2lastidx = {}
    for cnt, itemid in enumerate(item_data_id[:train_end_idx]):
        itemid2lastidx[itemid] = cnt

    try:
        embedding_dim = user_embeddings_time_series.size(1)
    except:
        embedding_dim = user_embeddings_time_series.shape[1]
    for userid in userid2lastidx:
        user_embeddings[userid, :embedding_dim] = user_embeddings_time_series[userid2lastidx[userid]]
    for itemid in itemid2lastidx:
        item_embeddings[itemid, :embedding_dim] = item_embeddings_time_series[itemid2lastidx[itemid]]

    user_embeddings.detach_()
    item_embeddings.detach_()

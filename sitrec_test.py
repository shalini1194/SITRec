'''
This code evaluates the validation and test performance in an epoch of the model trained in sitrec.py.
The task is: interaction prediction, i.e., predicting which item will a user interact with?
To calculate the performance for one epoch:
$ python sitrec_test.py --network ML_network --epoch 49
To calculate the performance for all epochs, use the bash file, evaluate_all_epochs.sh, which calls this file once for every epoch.

'''

from library_data import *
from library_models import *
import argparse

# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--network', default='ML_network', help='Network name')
parser.add_argument('--model', default='jodie', help="Model name")
parser.add_argument('--gpu', default=0, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epoch', default=1, type=int, help='Epoch id to load')
parser.add_argument('--embedding_dim', default=20
                    , type=int, help='Number of dimensions')
parser.add_argument('--week', default=7, type=int, help='Number of weeks in training set')
parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate of the optimizer')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Proportion of training interactions')

args = parser.parse_args()
args.datapath = "data/%s.csv" % args.network
if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')
if args.network == "mooc":
    print("No interaction prediction for %s" % args.network)
    sys.exit(0)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# CHECK IF THE OUTPUT OF THE EPOCH IS ALREADY PROCESSED. IF SO, MOVE ON.
output_fname = "results/interaction_prediction_%s.txt" % args.network

# LOAD NETWORK
def weighted_mse_loss(input,target,weights):
    out = (input-target)**2
    weights = weights.expand_as(out)
    out = out * weights.cuda()
    loss = out.sum() # or sum over whatever dimensions
    return loss
def weighted_pairwise_loss(input,target,weights):
    out = (input-target)**2
    weights = weights.expand_as(out)
    out = out * weights.cuda()
    loss = out.sum(0) # or sum over whatever dimensions
    return loss

[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence, \
 item2id, item_sequence_id, item_timediffs_sequence, \
posted_on, replied_on, posted_on_list, replied_on_list,
 timestamp_sequence, \
 feature_sequence, \
 y_true, item_length, index, feature_syllabus, negative_sample, item_timediff_test_sequence
 , user_item_timediffs] = load_network(args)
num_interactions = len(user_sequence_id)
num_features = len(feature_sequence[0])
feature_syllabus.append([0]*num_features)
num_users = len(user2id)
num_items = len(item2id) + 1
true_labels_ratio = len(y_true)/(sum(y_true)+1)
print("*** Network statistics:\n  %d users\n  %d items\n  %d interactions\n  %d/%d true labels ***\n\n" % (num_users, num_items, num_interactions, sum(y_true), len(y_true)))

# SET TRAIN, VALIDATION, AND TEST BOUNDARIES
train_end_idx = index - 1
test_start_idx = index
test_end_idx = len(user_sequence_id)

# SET BATCHING TIMESPAN
'''
Timespan indicates how frequently the model is run and updated.
All interactions in one timespan are processed simultaneously.
Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
At the end of each timespan, the model is updated as well. So, longer timespan means less frequent model updates.
'''
timespan = timestamp_sequence[-1] - timestamp_sequence[0]

tbatch_timespan = timespan / 5000
item_max_length = max(item_length)
args.state_change = False
# INITIALIZE MODEL PARAMETERS
model = SITRec(args, num_features, num_users, num_items).cuda()
weight = torch.Tensor([1,true_labels_ratio]).cuda()
crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
MSELoss = nn.MSELoss()

# INITIALIZE MODEL
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# LOAD THE MODEL
model, optimizer, user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer, args, args.epoch)
if train_end_idx != train_end_idx_training:
    sys.exit('Training proportion during training and testing are different. Aborting.')

# SET THE USER AND ITEM EMBEDDINGS TO THEIR STATE AT THE END OF THE TRAINING PERIOD
set_embeddings_training_end(user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, user_sequence_id, item_sequence_id, train_end_idx) 

# LOAD THE EMBEDDINGS: DYNAMIC AND STATIC
item_embeddings = item_embeddings_dystat[:, :args.embedding_dim]



item_embeddings = item_embeddings.clone()
item_embeddings_static = item_embeddings_dystat[:, args.embedding_dim:]
item_embeddings_static = item_embeddings_static.clone()

user_embeddings = user_embeddings_dystat[:, :args.embedding_dim]
user_embeddings = user_embeddings.clone()
user_embeddings_static = user_embeddings_dystat[:, args.embedding_dim:]
user_embeddings_static = user_embeddings_static.clone()

# PERFORMANCE METRICS
validation_ranks = []
test_ranks = []

'''
Here we use the trained model to make predictions for the validation and testing interactions.
The model does a forward pass from the start of validation till the end of testing.
For each interaction, the trained model is used to predict the embedding of the item it will interact with.
This is used to calculate the rank of the true item the user actually interacts with.
After this prediction, the errors in the prediction are used to calculate the loss and update the model parameters.
This simulates the real-time feedback about the predictions that the model gets when deployed in-the-wild.
Please note that since each interaction in validation and test is only seen once during the forward pass, there is no data leakage.
'''
tbatch_start_time = None
loss = 0
# FORWARD PASS
user_item={}
user_predicted={}

previous_item_ids={}
#print("*** Making interaction predictions by forward pass (no t-batching) ***")
previous_item_id = -1
with trange(train_end_idx, test_end_idx) as progress_bar:
    for j in progress_bar:
        progress_bar.set_description('%dth interaction for validation and testing' % j)

        # LOAD INTERACTION J
        userid = user_sequence_id[j]
        itemid = item_sequence_id[j]

        negative_sampleid = negative_sample[j]
        feature = feature_sequence[j]
        user_timediff = user_timediffs_sequence[j]
        timestamp = timestamp_sequence[j]
        if not tbatch_start_time:
            tbatch_start_time = timestamp
        itemid_previous = user_previous_itemid_sequence[j]

        # LOAD USER AND ITEM EMBEDDING
        user_embedding_input = user_embeddings[torch.cuda.LongTensor([userid])]
        user_embedding_static_input = user_embeddings_static[torch.cuda.LongTensor([userid])]
        item_embedding_input = item_embeddings[torch.cuda.LongTensor([itemid])]

        user_timediffs_tensor = torch.Tensor([user_timediff]).cuda().unsqueeze(0)
        item_timediffs_tensor = torch.Tensor([item_timediffs_sequence[j]]).cuda().unsqueeze(0)
        posted_on_list_tensor = torch.Tensor([posted_on_list[j]]).cuda()
        user_item_timediff_tensor = torch.Tensor([user_item_timediffs[j]]).cuda()
        #posted_on_list_tensor = posted_on_list_tensor.permute(0,2,1)
        user_item_timediff_tensor = user_item_timediff_tensor.view(num_items,1).squeeze(1)
        posted_on_list_tensor= posted_on_list_tensor.view(num_items,1).squeeze(1)

        replied_on_list_tensor = torch.Tensor([replied_on_list[j]]).cuda()
        replied_on_list_tensor = replied_on_list_tensor.view(num_items,1).squeeze(1)
        item_timediffs_sequence_tensor = torch.Tensor([item_timediff_test_sequence[j]]).cuda()
        item_timediffs_sequence_tensor = item_timediffs_sequence_tensor.view(num_items, 1).squeeze(1)

        #replied_on_list_tensor=replied_on_list_tensor.permute(0,2,1)


        #negative_sample_embedding = model.item_decay(negative_sample_input, user_embedding_input, item_timediffs_tensor)


        item_embedding_static_input = item_embeddings_static[torch.cuda.LongTensor([itemid])]
        negative_sample_static_input = item_embeddings_static[torch.cuda.LongTensor([negative_sampleid[0]])]
        feature_tensor = Variable(torch.Tensor(feature).cuda()).unsqueeze(0)
        feature_syllabus_tensor = torch.Tensor(feature_syllabus).cuda()
        feature_syllabus_project = model.topic_input(feature_syllabus_tensor)

        item_embedding_previous = item_embeddings[torch.cuda.LongTensor([itemid_previous])]
        num_topics = feature_syllabus_tensor.shape[0]

        current_topic = feature_syllabus_tensor[:num_topics].unsqueeze(1).repeat(1,
                                                                                   user_embedding_input.shape[0],
                                                                                   1).permute(1, 0, 2)
        current_topic = current_topic.contiguous().view(
            user_embedding_input.shape[0] * (num_topics), -1)


        current_topic_distance = nn.PairwiseDistance()(feature_tensor.repeat(num_topics, 1),
                                                       current_topic)

        current_topic_distance = torch.argmin(
            current_topic_distance.view(user_embedding_input.shape[0], num_topics), dim=1)

        feature_syllabus_project = feature_syllabus_project[current_topic_distance]
        #
        user_projected_embedding = model.project_user(user_embedding_input, feature_syllabus_project, timediffs=user_timediffs_tensor,
                                                       features=feature_tensor)

        user_item_embedding = torch.cat([user_embedding_input, item_embedding_previous, item_embeddings_static[torch.cuda.LongTensor([itemid_previous])], user_embeddings_static[torch.cuda.LongTensor([userid])]], dim=1)


        # PREDICT ITEM EMBEDDING
        predicted_item_embedding = model.predict_item_embedding(user_item_embedding)
        #print(item_timediffs_sequence_tensor[item_timediffs_sequence_tensor>1])

        embedings = torch.cat([item_embeddings, item_embeddings_static], dim=1)

        item_projected_embedding = model.item_decay(torch.cat([item_embeddings, item_embeddings_static],dim=1),
                                                    predicted_item_embedding.repeat(num_items,1),
                                                    posted_on_list_tensor,
                                                    replied_on_list_tensor,
                                                    user_item_timediff_tensor,
                                                    item_timediffs_sequence_tensor)

        #print(item_embeddings_static[negative_sample[ 0], :].shape)
        # CALCULATE PREDICTION LOSS
        negative_sample_ids = torch.argmax(user_item_timediff_tensor, dim=0)
        weights = torch.cat([torch.ones((num_items)) / num_items, torch.ones(args.embedding_dim) / args.embedding_dim])

        loss += MSELoss(predicted_item_embedding, item_projected_embedding)

        # loss -= 0.001*weighted_mse_loss(prediprintcted_item_embedding,
        #                    torch.cat([item_projected_embedding[torch.cuda.LongTensor([negative_sample_ids])], negative_sample_static_input],
        #                              dim=1).detach(),weights)

        
        # CALCULATE DISTANCE OF PREDICTED ITEM EMBEDDING TO ALL ITEMS 
        euclidean_distances = nn.PairwiseDistance()(predicted_item_embedding.repeat(num_items, 1), item_projected_embedding).squeeze(-1)


        # CALCULATE RANK OF THE TRUE ITEM AMONG ALL ITEMS
        true_item_distance = euclidean_distances[itemid]
        euclidean_distances_smaller = (euclidean_distances < true_item_distance).data.cpu().numpy()
        true_item_rank = np.sum(euclidean_distances_smaller) + 1

        if userid in user_item.keys():
            previous_item_ids[userid].append(previous_item_id)
            user_item[userid].append(itemid)

            user_predicted[userid].append(np.argsort(euclidean_distances.data.cpu().numpy())[:10])
        else:
            previous_item_ids[userid]=[previous_item_id]
            user_item[userid] = [itemid]
            user_predicted[userid] = [np.argsort(euclidean_distances.data.cpu().numpy())[:10]]

        if j < test_start_idx:
            validation_ranks.append(true_item_rank)
        else:
            test_ranks.append(true_item_rank)
        previous_item_id = itemid
        # UPDATE USER AND ITEM EMBEDDING
        user_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor  , select='user_update')
        item_embedding_output = model.forward(user_embedding_input, item_embedding_input,timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update')

        # SAVE EMBEDDINGS
        item_embeddings[itemid,:] = item_embedding_output.squeeze(0) 
        user_embeddings[userid,:] = user_embedding_output.squeeze(0) 
        user_embeddings_timeseries[j, :] = user_embedding_output.squeeze(0)
        item_embeddings_timeseries[j, :] = item_embedding_output.squeeze(0)

        # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
        loss += MSELoss(item_embedding_output, item_embedding_input.detach())
        loss += MSELoss(user_embedding_output, user_embedding_input.detach())

        # CALCULATE STATE CHANGE LOSS
        if args.state_change:
            loss += calculate_state_prediction_loss(model, [j], user_embeddings_timeseries, y_true, crossEntropyLoss) 

        # UPDATE THE MODEL IN REAL-TIME USING ERRORS MADE IN THE PAST PREDICTION
        #if timestamp - tbatch_start_time > tbatch_timespan:
        tbatch_start_time = timestamp
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
            
            # RESET LOSS FOR NEXT T-BATCH
        loss = 0
        item_embeddings.detach_()
        user_embeddings.detach_()
        item_embeddings_timeseries.detach_()
        user_embeddings_timeseries.detach_()
            
# CALCULATE THE PERFORMANCE METRICS
performance_dict = dict()

ranks = test_ranks
mrr = np.mean([1.0 / r for r in ranks])
rec10 = sum(np.array(ranks) <= 10)*1.0 / len(ranks)


# PRINT AND SAVE THE PERFORMANCE METRICS
fw = open(output_fname, "a")
# metrics = ['Mean Reciprocal Rank', 'Recall@10']
#

#print("***MAP***")
AP = []
for u, pred in user_predicted.items():

    #for i in range(len(pred)):
    score = 0.0
    num_hits = 0.0
    prediction = pred[0][:5]

    actual_i = user_item[u]
    #print(len(prediction))
    #print(len(actual_i[0]))
        # print(previous_item_ids[u])
    for j, p in enumerate(prediction):
        if p in actual_i:
            num_hits += 1.0
            score += num_hits / (j + 1.0)
    AP.append(score / min(len(pred), 5))
MAP = np.mean(AP)

# print('\n\n **Test MAP performance of epoch %d at 10' %args.epoch +'is  %f***' %MAP )
# AP=[]
# for u, pred in user_predicted.items():
#
#     for i in range(len(pred)):
#         score = 0.0
#         num_hits = 0.0
#         prediction = pred[i][:5]
#
#         actual_i = user_item[u]
#
#         for j, p in enumerate(prediction):
#             if p == actual_i[i]:
#                 num_hits += 1.0
#                 score += num_hits / (j + 1.0)
#     AP.append(score / min(len(pred), 5))

MAP = np.mean(AP)
performance_dict['test'] = [MAP, rec10]
print('\n\n **Test MAP performance of epoch %d at 5' %args.epoch +'is  %f***' %MAP )
metrics = ['MAP', 'Recall@10']
print( '\n\n*** Test performance of epoch %d ***' % args.epoch)
fw.write('\n\n*** Test performance of epoch %d ***\n' % args.epoch)
for i in range(len(metrics)):
    print(metrics[i] + ': ' + str(performance_dict['test'][i]))
    fw.write("Test: " + metrics[i] + ': ' + str(performance_dict['test'][i]) + "\n")
# AP=[]
# for u, pred in user_predicted.items():
#
#     for i in range(len(pred)):
#         score = 0.0
#         num_hits = 0.0
#         prediction = pred[i][:3]
#
#         actual_i = user_item[u]
#         for j, p in enumerate(prediction):
#             if p == actual_i[i]:
#                 num_hits += 1.0
#                 score += num_hits / (j + 1.0)
#     AP.append(score / min(len(pred), 3))
# MAP = np.mean(AP)
# print('\n\n **Test MAP performance of epoch %d at 3' %args.epoch +'is  %f***' %MAP )


# fw.flush()
# fw.close()

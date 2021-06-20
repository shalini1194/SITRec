'''
This code trains the JODIE model for the given dataset.
The task is: interaction prediction.

How to run:
$ python sitrec.py --network ML_network --epochs 50
'''

from library_data import *
import library_models as lib
from library_models import *
import argparse
# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--network', default = "ML_network", help='Name of the network/dataset')
parser.add_argument('--gpu', default=0, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train the model')
parser.add_argument('--embedding_dim', default=20, type=int, help='Number of dimensions of the dynamic embedding')
parser.add_argument('--week', default=7, type=int, help='Number of weeks in training set')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate of the optimizer')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')
parser.add_argument('--state_change', default=False, type=bool, help='True if training with state change of users along with interaction prediction. False otherwise. By default, set to True.')
args = parser.parse_args()
args.state_change = False
args.datapath = "data/%s.csv" % args.network
if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
# LOAD DATA
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
 item2id, item_sequence_id, item_timediffs_sequence,
 posted_on, replied_on,posted_on_list, replied_on_list,
 timestamp_sequence, feature_sequence, y_true, item_length, index, feature_syllabus,negative_sample, item_timediff_test_sequence,
 user_item_timediffs] = load_network(args)

num_interactions = len(user_sequence_id)
num_users = len(user2id)
num_items = len(item2id) + 1 # one extra item for "none-of-these"
num_features = len(feature_sequence[0])


true_labels_ratio = len(y_true)/(1.0+sum(y_true)) # +1 in denominator in case there are no state change labels, which will throw an error.
print("*** Network statistics:\n  %d users\n  %d items\n  %d interactions\n  %d/%d true labels ***\n\n" % (num_users, num_items, num_interactions, sum(y_true), len(y_true)))

# SET TRAINING, VALIDATION, TESTING, and TBATCH BOUNDARIES
train_end_idx = index-1
test_start_idx = index
test_end_idx = len(user_sequence_id)


timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = timespan / 500

# INITIALIZE MODEL AND PARAMETERS
model = SITRec(args, num_features, num_users, num_items).cuda()
weight = torch.Tensor([1,true_labels_ratio]).cuda()
crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
MSELoss = nn.MSELoss()

# INITIALIZE EMBEDDING
initial_user_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0)) # the initial user and item embeddings are learned during training as well
initial_item_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
model.initial_user_embedding = initial_user_embedding
model.initial_item_embedding = initial_item_embedding

user_embeddings = initial_user_embedding.repeat(num_users, 1) # initialize all users to the same embedding
item_embeddings = initial_item_embedding.repeat(num_items, 1) # initialize all items to the same embedding
item_embedding_static = Variable(torch.eye(num_items).cuda()) # one-hot vectors for static embeddings
user_embedding_static = Variable(torch.eye(num_users).cuda()) # one-hot vectors for static embeddings

# INITIALIZE MODEL
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# RUN THE JODIE MODEL
'''
THE MODEL IS TRAINED FOR SEVERAL EPOCHS. IN EACH EPOCH, JODIES USES THE TRAINING SET OF INTERACTIONS TO UPDATE ITS PARAMETERS.
'''
print("*** Training the JODIE model for %d epochs ***" % args.epochs)
def weighted_mse_loss(input,target,weights):
    out = (input-target)**2
    weights = weights.expand_as(out)
    out = out * weights.cuda()
    loss = out.sum() # or sum over whatever dimensions
    return loss
topic_transaction = torch.zeros([num_interactions], dtype = torch.long).cuda()
topic_transaction2 = torch.zeros([num_interactions, 9],dtype=torch.float).cuda()
user_transaction =torch.zeros([num_interactions], dtype = torch.long).cuda()
item_transaction = torch.zeros([num_interactions],dtype=torch.long).cuda()
with trange(args.epochs) as progress_bar1:
    for ep in progress_bar1:
        progress_bar1.set_description('Epoch %d of %d' % (ep, args.epochs))

        # INITIALIZE EMBEDDING TRAJECTORY STORAGE
        user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())
        item_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())

        feature_syllabus_tensor = Variable(torch.Tensor(feature_syllabus).cuda())

        optimizer.zero_grad()
        reinitialize_tbatches()
        total_loss, loss, total_interaction_count = 0, 0, 0

        tbatch_start_time = None
        tbatch_to_insert = -1
        tbatch_full = False

        # TRAIN TILL THE END OF TRAINING INTERACTION IDX
        with trange(train_end_idx) as progress_bar2:
            for j in progress_bar2:
                progress_bar2.set_description('Processed %dth interactions' % j)

                # READ INTERACTION J
                userid = user_sequence_id[j]
                itemid = item_sequence_id[j]
                feature = feature_sequence[j]
                user_timediff = user_timediffs_sequence[j]
                item_timediff = item_timediffs_sequence[j]
                item_timediff_test = item_timediff_test_sequence[j]


                # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
                tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1
                lib.tbatchid_user[userid] = tbatch_to_insert
                lib.tbatchid_item[itemid] = tbatch_to_insert
                lib.current_tbatches_feature_syllabus = (feature_syllabus)
                lib.current_tbatches_user[tbatch_to_insert].append(userid)
                lib.current_tbatches_item[tbatch_to_insert].append(itemid)
                lib.current_tbatches_feature[tbatch_to_insert].append(feature)
                lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
                lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
                lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
                lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])
                lib.current_tbatches_user_item_timediff[tbatch_to_insert].append(user_item_timediffs[j])
                lib.current_tbatches_item_timediff_sequence[tbatch_to_insert].append(item_timediff_test)

                lib.current_tbatches_posted_on[tbatch_to_insert].append(posted_on_list[j])
                lib.current_tbatches_replied_on[tbatch_to_insert].append(replied_on_list[j])
                timestamp = timestamp_sequence[j]
                if tbatch_start_time is None:
                    tbatch_start_time = timestamp

                # AFTER ALL INTERACTIONS IN THE TIMESPAN ARE CONVERTED TO T-BATCHES, FORWARD PASS TO CREATE EMBEDDING TRAJECTORIES AND CALCULATE PREDICTION LOSS
                if timestamp - tbatch_start_time > tbatch_timespan:
                    tbatch_start_time = timestamp # RESET START TIME FOR THE NEXT TBATCHES

                    # ITERATE OVER ALL T-BATCHES
                    with trange(len(lib.current_tbatches_user)) as progress_bar3:
                        for i in progress_bar3:
                            progress_bar3.set_description('Processed %d of %d T-batches ' % (i, len(lib.current_tbatches_user)))

                            total_interaction_count += len(lib.current_tbatches_interactionids[i])

                            # LOAD THE CURRENT TBATCH
                            tbatch_userids = torch.LongTensor(lib.current_tbatches_user[i]).cuda() # Recall "lib.current_tbatches_user[i]" has unique elements
                            tbatch_itemids = torch.LongTensor(lib.current_tbatches_item[i]).cuda() # Recall "lib.current_tbatches_item[i]" has unique elements
                            tbatch_negative_sample = torch.LongTensor(lib.current_tbatches_negative_sample[i]).cuda()
                            tbatch_interactionids = torch.LongTensor(lib.current_tbatches_interactionids[i]).cuda()
                            feature_tensor = Variable(torch.Tensor(lib.current_tbatches_feature[i]).cuda()) # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                            user_timediffs_tensor = Variable(torch.Tensor(lib.current_tbatches_user_timediffs[i]).cuda()).unsqueeze(1)
                            item_timediffs_tensor = Variable(torch.Tensor(lib.current_tbatches_item_timediffs[i]).cuda()).unsqueeze(1)
                            tbatch_itemids_previous = torch.LongTensor(lib.current_tbatches_previous_item[i]).cuda()
                            posted_on_tensor = torch.Tensor(lib.current_tbatches_posted_on[i]).cuda()
                            replied_on_tensor = torch.Tensor(lib.current_tbatches_replied_on[i]).cuda()
                            item_embedding_previous = item_embeddings[tbatch_itemids_previous,:]
                            user_item_timediff = torch.Tensor(lib.current_tbatches_user_item_timediff[i]).cuda()
                            item_timediff_test_tensor = torch.Tensor(lib.current_tbatches_item_timediff_sequence[i]).cuda()

                            # PROJECT USER EMBEDDING TO CURRENT TIME
                            user_embedding_input = user_embeddings[tbatch_userids,:]
                            item_embedding_input = item_embeddings[tbatch_itemids, :]


                            feature_syllabus_project = model.topic_input(feature_syllabus_tensor)
                            num_topics = feature_syllabus_tensor.shape[0]
                            current_topic = feature_syllabus_tensor[:num_topics].unsqueeze(1).repeat(1,
                                                                                        user_embedding_input.shape[0],
                                                                                        1).permute(1, 0, 2)
                            current_topic = current_topic.contiguous().view(
                                user_embedding_input.shape[0] * (num_topics), -1)

                            current_topic_distance = nn.PairwiseDistance()(feature_tensor.repeat(num_topics, 1),
                                                                          current_topic)
                            topic_transaction2[tbatch_interactionids] = current_topic_distance.view(user_embedding_input.shape[0], num_topics)
                            current_topic_distance = torch.argmin(
                                current_topic_distance.view(user_embedding_input.shape[0], num_topics), dim=1)
                            topic_transaction[tbatch_interactionids]= current_topic_distance
                            user_transaction[tbatch_interactionids] = tbatch_userids
                            item_transaction[tbatch_interactionids] = tbatch_itemids

                            feature_syllabus_project = feature_syllabus_project[current_topic_distance]

                            user_projected_embedding = model.project_user(user_embedding_input,feature_syllabus_project, timediffs=user_timediffs_tensor, features=feature_tensor)
                            #feature_syllabus_project = model.feature_project(feature_tensor,feature_syllabus_project)
                            user_item_embedding = torch.cat([user_embedding_input, item_embedding_previous, item_embedding_static[tbatch_itemids_previous,:], user_embedding_static[tbatch_userids,:]], dim=1)

                            # PREDICT NEXT ITEM EMBEDDING
                            predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

                            negative_sample_ids = torch.argmax(user_item_timediff, dim=1)
                            negative_sample_embedding1 = model.item_decay(item_embeddings[negative_sample_ids],
                                                                          predicted_item_embedding[:,
                                                                          :args.embedding_dim],
                                                                          posted_on_tensor[torch.arange(len(tbatch_userids)),negative_sample_ids],
                                                                          replied_on_tensor[torch.arange(len(tbatch_userids)),negative_sample_ids],
                                                                          user_item_timediff[torch.arange(len(tbatch_userids)),negative_sample_ids],
                                                                          item_timediff_test_tensor[torch.arange(len(tbatch_userids)),negative_sample_ids])
                            item_projected_embedding = model.item_decay(torch.cat([item_embedding_input, item_embedding_static[tbatch_itemids,:]],dim=1),
                                                                        predicted_item_embedding,
                                                                        posted_on_tensor[torch.arange(len(tbatch_userids)),tbatch_itemids],
                                                                        replied_on_tensor[torch.arange(len(tbatch_userids)),tbatch_itemids],
                                                                        user_item_timediff[torch.arange(len(tbatch_userids)),tbatch_itemids],
                                                                        item_timediff_test_tensor[torch.arange(len(tbatch_userids)),tbatch_itemids])

                            # CAfLCULATE PREDICTION LOS
                            weights = torch.cat([torch.ones((num_items))/num_items, torch.ones(args.embedding_dim)/args.embedding_dim])
                            loss += weighted_mse_loss(predicted_item_embedding,item_projected_embedding, weights)

                            #loss += MSELoss(weighted_predicted_item_embedding, weighted_item_embedding)
                            #loss -= 0.001*weighted_mse_loss(predicted_item_embedding, torch.cat([negative_sample_embedding1, item_embedding_static[negative_sample_ids]], dim=1).detach(),weights)
                           # loss -= 0.001 * MSELoss(predicted_item_embedding, torch.cat(
                           #     [negative_sample_embedding2, item_embedding_static[tbatch_negative_sample[:, 1], :]],
                           #     dim=1).detach())
                           #loss -= 0.001*
                            # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                            user_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')
                            item_embedding_output = model.forward(user_embedding_input, item_embedding_input,timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update')

                            item_embeddings[tbatch_itemids,:] = item_embedding_output
                            user_embeddings[tbatch_userids,:] = user_embedding_output

                            user_embeddings_timeseries[tbatch_interactionids,:] = user_embedding_output
                            item_embeddings_timeseries[tbatch_interactionids,:] = item_embedding_output

                            # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                            loss += MSELoss(item_embedding_output, item_embedding_input.detach())
                            loss += MSELoss(user_embedding_output, user_embedding_input.detach())

                            # CALCULATE STATE CHANGE LOSS
                            if args.state_change:
                                loss += calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_timeseries, y_true, crossEntropyLoss)

                    # BACKPROPAGATE ERROR AFTER END OF T-BATCH
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # RESET LOSS FOR NEXT T-BATCH
                    loss = 0
                    item_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                    user_embeddings.detach_()
                    item_embeddings_timeseries.detach_()
                    user_embeddings_timeseries.detach_()

                    # REINITIALIZE
                    reinitialize_tbatches()

                    tbatch_to_insert = -1

        # END OF ONE EPOCH
        print("\n\nTotal loss in this epoch = %f" % (total_loss))
        item_embeddings_dystat = torch.cat([item_embeddings, item_embedding_static], dim=1)
        user_embeddings_dystat = torch.cat([user_embeddings, user_embedding_static], dim=1)
        # SAVE CURRENT MODEL TO DISK TO BE USED IN EVALUATION.
        save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, topic_transaction,topic_transaction2,item_transaction, user_transaction,user_embeddings_timeseries, item_embeddings_timeseries)

        user_embeddings = initial_user_embedding.repeat(num_users, 1)
        item_embeddings = initial_item_embedding.repeat(num_items, 1)

# END OF ALL EPOCHS. SAVE FINAL MODEL DISK TO BE USED IN EVALUATION.
print("\n\n*** Training complete. Saving final model. ***\n\n")
save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)


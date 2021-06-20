'''
This is a supporting library for the loading the data.

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
'''

from __future__ import division

import numpy as np
import random
import copy
from collections import defaultdict

algo_network_start = 1372679530

ML_network_start = 1366638288

comp_network_start = 1411062133
T1_week=604800
T1_day= 86400

from sklearn.preprocessing import scale

# LOAD THE NETWORK
def load_network(args, time_scaling=True):
    '''
    This function loads the input network.

    The network should be in the following format:
    One line per interaction/edge.
    Each line should be: user, item, timestamp, state label, array of features.
    Timestamp should be in cardinal format (not in datetime).
    State label should be 1 whenever the user state changes, 0 otherwise. If there are no state labels, use 0 for all interactions.
    Feature list can be as long as desired. It should be atleast 1 dimensional. If there are no features, use 0 for all interactions. 
    '''

    datapath = args.datapath

    user_sequence = []
    item_sequence = []
    posted_on = []
    replied_on=[]
    replied_on_list= []
    posted_on_list=[]
    feature_sequence = []
    timestamp_sequence = []
    start_timestamp = None
    y_true_labels = []
    arr=[]
    arr_user=[]
    print(args.network)
    f = open(datapath,"r")
    f.readline()
    index= -1
    start_time = algo_network_start
    if args.network == "comp_network":
        start_time = comp_network_start
    if args.network[:10] == "ML_network":
        start_time = ML_network_start

    for cnt, l in enumerate(f):
        # FORMAT: user, item, timestamp, state label, feature list 
        ls = l.strip().split(",")


        if int(ls[1]) not in arr and int(ls[2])>start_time+T1_week*args.week:
            continue
        if int(ls[0]) not in arr_user and int(ls[2])>start_time+T1_week*args.week:
            continue
        if int(float(ls[2])) > start_time+T1_week*args.week+7*T1_day:
            continue
        if int(ls[2])<=start_time+T1_week*args.week:
            arr.append(int(ls[1]))
            arr_user.append(int(ls[0]))
        if index == -1 and int(ls[2])>start_time+T1_week*args.week:
            index=cnt


        user_sequence.append(ls[0])
        item_sequence.append(ls[1])
        if start_timestamp is None:
            start_timestamp = float(ls[2])
        timestamp_sequence.append(float(ls[2]) - start_timestamp)

        posted_on.append(int(float(ls[3])))
        replied_on.append(int(float(ls[4])))
        feature_sequence.append(list(map(float,ls[5:])))
    f.close()
    user_sequence = np.array(user_sequence) 
    item_sequence = np.array(item_sequence)
    timestamp_sequence = np.array(timestamp_sequence)

    nodeid = 0
    item2id = {}
    item_length = []
    item_timedifference_sequence = []
    item_current_timestamp = defaultdict(float)

    for cnt, item in enumerate(item_sequence):
        if item not in item2id:
            item2id[item] = nodeid
            
            nodeid += 1
            item_length.append(1)

        item_length[item2id[item]]+=1
        timestamp = timestamp_sequence[cnt]
        item_timedifference_sequence.append(timestamp - item_current_timestamp[item])
        item_current_timestamp[item] = timestamp

    num_items = len(item2id)
    item_sequence_id = [item2id[item] for item in item_sequence]
    negative_sample=[]
    last_index = random.randrange(10)
    for cnt, item in enumerate(item_sequence_id):
        cnt1 = cnt + 1
        while cnt1 < len(item_sequence):
            if np.argmax(feature_sequence[cnt1]) != np.argmax(feature_sequence[cnt]):
                break
            cnt1 += 1
        if cnt1 < len(item_sequence):
            negative_sample.append([item_sequence_id[cnt1],item_sequence_id[last_index]])
            continue

        cnt1 = cnt - 1
        while cnt1 >= 0:
            if np.argmax(feature_sequence[cnt1]) != np.argmax(feature_sequence[cnt]):
                break
            cnt1 -= 1
        if cnt1 >= 0:
            negative_sample.append([item_sequence_id[cnt1],item_sequence_id[last_index]])
            continue



    item_timediff_test_sequence =[]

    item_current_timestamp = defaultdict(float)

    for i, item in enumerate(item_sequence):
        item_timediff_test = [0] * (num_items+1)

        timestamp = timestamp_sequence[i]
        item_current_timestamp[item] = timestamp

        for item1, id in item2id.items():

            item_timediff_test[id] = (timestamp - item_current_timestamp[item1])

        item_timediff_test_sequence.append(copy.deepcopy(scale(np.array(item_timediff_test))))
        i+=1
    item_timediff_test_sequence=np.array(item_timediff_test_sequence)

    nodeid = 0
    user2id = {}
    user_timedifference_sequence = []
    user_current_timestamp = defaultdict(float)
    user_previous_itemid_sequence = []
    user_latest_itemid = defaultdict(lambda: num_items)
    for cnt, user in enumerate(user_sequence):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        user_timedifference_sequence.append(timestamp - user_current_timestamp[user])
        user_current_timestamp[user] = timestamp
        user_previous_itemid_sequence.append(user_latest_itemid[user])
        user_latest_itemid[user] = item2id[item_sequence[cnt]]
    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]
    user_thread_post=np.zeros((num_users, num_items+1))

    print('creating user item diffs')
    user_thread_reply = np.zeros((num_users,num_items+1))

    user_item_timediff = np.zeros((num_users,num_items+1))
    user_item_timediffs =[]



    for cnt,user in enumerate(user_sequence_id):

        user_thread_post[user][item_sequence_id[cnt]] = posted_on[cnt]

        user_thread_reply[user][item_sequence_id[cnt]] = replied_on[cnt]

        posted_on_list.append(copy.deepcopy(user_thread_post[user]))
        replied_on_list.append(copy.deepcopy(user_thread_reply[user]))
        posted_on[cnt] = (user_thread_post[user][item_sequence_id[cnt]])
        replied_on[cnt] = (user_thread_reply[user][item_sequence_id[cnt]])
        user_thread_post[user][item_sequence_id[cnt]] +=1
        timestamp = timestamp_sequence[cnt]
        user_item_timediffs_element = []
        user_item_timediff[user][item_sequence_id[cnt]] = timestamp
        for item in  user_item_timediff[user]:
           user_item_timediffs_element.append(timestamp- item)
        user_item_timediffs.append(copy.deepcopy(scale(np.array(user_item_timediffs_element))))

    user_item_timediffs = np.array(user_item_timediffs)
    feature_syllabus = []
    syllabus = 'data/algo_syllabus.csv'
    if args.network == 'Doc2vec_ML':
        syllabus = 'data/Doc2vec_syll_ML.csv'
    if args.network == 'Doc2vec_algo':
        syllabus = 'data/Doc2vec_syll.csv'

    if args.network == 'algo_network_9':
            syllabus = 'data/algo_syllabu_9.csv'
    if args.network == 'algo_network_8':
        syllabus = 'data/algo_syllabu_8.csv'
        syllabus = 'data/ML_syllabus.csv'
    if args.network == 'ML_network':
        syllabus = 'data/ML_syllabus.csv'
    if args.network == 'ML_network_6':
        syllabus = 'data/ML_syllabus_6.csv'
    if args.network == 'ML_network_5':
        syllabus = 'data/ML_syllabus_5.csv'
    if args.network == 'ML_network_4':
        syllabus = 'data/ML_syllabus_4.csv'
    if args.network == 'ML_network_3':
        syllabus = 'data/ML_syllabus_3.csv'
    if args.network == 'ML_network_2':
        syllabus = 'data/ML_syllabus_2.csv'
    if args.network == 'comp_network':
        syllabus = 'data/comp_syllabus.csv'
    f = open(syllabus, 'r')
    f.readline()
    cnt = 0
    for l in (f):
        cnt += 1
        ls = l.strip().split(",")
        feature_syllabus.append([float(i) for i in ls])



    AP = []
    user_item = defaultdict(list)

    for cnt, user in enumerate(user_sequence_id):
        if cnt > index:
            pred = user_item[user]

            score = 0.0
            num_hits = 0.0
            prediction = pred[:5]

            actual_i = item_sequence_id[cnt]
            for j, p1 in enumerate(prediction):
                if p1 == actual_i:
                    num_hits += 1.0
                    score += num_hits / (j + 1.0)
            AP.append(score)
        user_item[user].append(item_sequence_id[cnt])
    MAP = np.mean(AP)

    print("MAP :" + str(MAP))

    if time_scaling:
        print("Scaling timestamps")
        user_timedifference_sequence = scale(np.array(user_timedifference_sequence) + 1)
        item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)

    print("*** Network loading completed ***\n\n")
    return [user2id, user_sequence_id, user_timedifference_sequence, user_previous_itemid_sequence, \
        item2id, item_sequence_id, item_timedifference_sequence, \
        posted_on, replied_on, posted_on_list, replied_on_list,
        timestamp_sequence, \
        feature_sequence, \
        y_true_labels,item_length, index, feature_syllabus, negative_sample, item_timediff_test_sequence,\
        user_item_timediffs]


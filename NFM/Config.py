"""
    author : yang yiqing 2018年07月13日15:57:50
"""

# file
train_file = 'data/dataset1.csv'
valid_file = 'data/dataset2.csv'
test_file = 'data/dataset3.csv'

train_save_file = 'data/dataset1.txt'
valid_save_file = 'data/dataset2.txt'
test_save_file = 'data/dataset3.txt'

label_name = 'label'

# features
numeric_features = ['all_launch_count', 'last_launch', 'all_video_count', 'last_video', 'all_video_day',
                    'all_action_count', 'last_action',
                    'all_action_day', 'register_day']
#numeric_features = []
single_features = ['register_type', 'device_type']
multi_features = []

num_embedding = True
single_feature_frequency = 10
multi_feature_frequency = 0

# model

FM_layer = True
DNN_layer = True

use_numerical_embedding = True


embedding_size = 16

dnn_net_size = [128,64,32]

# train
batch_size = 512
epochs = 300
learning_rate = 0.01




# -*- coding: utf-8 -*- 

LENET5_BATCH_SIZE = 32
LENET5_PATCH_SIZE = 5
LENET5_PATCH_DEPTH_1 = 6
LENET5_PATCH_DEPTH_2 = 16
LENET5_NUM_HIDDEN_1 = 120
LENET5_NUM_HIDDEN_2 = 84

import tensorflow as tf
import numpy as np
import input_data

num_labels = 10
batch_size = LENET5_BATCH_SIZE
image_width = 28
image_height = 28
image_depth = 1
#number of iterations and learning rate
num_steps = 10001
display_step = 1000
learning_rate = 0.001

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def one_hot_encode(np_array):
    return (np.arange(10) == np_array[:,None]).astype(np.float32)

def reformat_data(dataset, labels, image_width, image_height, image_depth):
    np_dataset_ = np.array([np.array(image_data).reshape(image_width, image_height, image_depth) for image_data in dataset])
    np_labels_ = one_hot_encode(np.array(labels, dtype=np.float32))
    np_dataset, np_labels = randomize(np_dataset_, np_labels_)
    return np_dataset, np_labels

def flatten_tf_array(array):
    shape = array.get_shape().as_list()
    return tf.reshape(array, [shape[0], shape[1] * shape[2] * shape[3]])

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

# 第一层：卷积层，该卷积层使用 Sigmoid 激活函数，并且在后面带有平均池化层。
# 第二层：卷积层，该卷积层使用 Sigmoid 激活函数，并且在后面带有平均池化层。
# 第三层：全连接层（使用 Sigmoid 激活函数）。
# 第四层：全连接层（使用 Sigmoid 激活函数）。
# 第五层：输出层。

def variables_lenet5(patch_size = LENET5_PATCH_SIZE,patch_depth1 = LENET5_PATCH_DEPTH_1,
                    patch_depth2 = LENET5_PATCH_DEPTH_2,
                    num_hidden1 = LENET5_NUM_HIDDEN_1,num_hidden2 = LENET5_NUM_HIDDEN_2,
                    image_depth = 1,num_labels = 10):
    # 第一层 卷积层  输出28*28*patch_depth1 ，池化输出14*14*patch_depth1 28 = LENET5_BATCH_SIZE - LENET5_PATCH_SIZE +1
    # truncated_normal 从截断的正态分布中输出随机值。 ,卷积核大小，patch_size*patch_size*image_depth 卷积核深度 image_depth 卷积核个数 patch_depth1 
    w1 = tf.Variable(tf.truncated_normal([patch_size,patch_size,image_depth,patch_depth1],stddev=0.1))
    b1 = tf.Variable(tf.zeros([patch_depth1]))
    
    # 第二层 卷积层 输出10*10*patch_depth2 池化输出5*5*patch_depth2
    # 卷积核大小，patch_size*patch_size*patch_depth1 卷积核深度 patch_depth1 卷积核个数 patch_depth2 
    w2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, patch_depth1, patch_depth2], stddev=0.1))
    b2 = tf.Variable(tf.constant(1.0, shape=[patch_depth2]))

    # 第三层 全连接层 输出num_hidden1
    # 卷积核大小，5*5*patch_depth2 卷积核深度 patch_depth2 卷积核个数 num_hidden1
    w3 = tf.Variable(tf.truncated_normal([patch_size,patch_size,patch_depth2, num_hidden1], stddev=0.1))
    b3 = tf.Variable(tf.constant(1.0, shape = [num_hidden1]))

    # 第四层 全连接层 输出num_hidden2
    # 卷积核大小，5*5*patch_depth2 卷积核深度 patch_depth2 卷积核个数 num_hidden1
    w4 = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2], stddev=0.1))
    b4 = tf.Variable(tf.constant(1.0, shape = [num_hidden2]))

    # 第五层 5*5 接第四层的输出 输出num_labels
    # 卷积核大小，5*5*patch_depth2 卷积核深度 patch_depth2 卷积核个数 num_hidden1
    w5 = tf.Variable(tf.truncated_normal([num_hidden2, num_labels], stddev=0.1))
    b5 = tf.Variable(tf.constant(1.0, shape = [num_labels]))

    variables = {
        'w1':w1,'w2':w2,'w3':w3,'w4':w4,'w5':w5,
        'b1':b1,'b2':b2,'b3':b3,'b4':b4,'b5':b5,
    }
    return variables

def model_lenet5(data,variables):
    # [1,1,1,1] 步长
    layer1_conv = tf.nn.conv2d(data, variables['w1'],[1,1,1,1],padding='SAME')
    #layer1_conv 大小 [-1,28,28,6]
    # 可视化特征
    #split是一个list，list数量为n个tensor，每一个tensor的shape=[-1,28,28,6]
    #输出第一个tensor的featuremap,根据第四个维度划分成64份
    split = tf.split(layer1_conv,num_or_size_splits=6,axis=3)
    tf.summary.image("conv1_features",split[0],data.get_shape()[0])

    #print(layer1_conv) #28*28*6
    layer1_actv = tf.nn.relu(layer1_conv + variables['b1'])
    layer1_pool = tf.nn.max_pool(layer1_actv,[1,2,2,1],[1,2,2,1],padding='SAME')
    #print(layer1_pool) #14*14*6
    layer2_conv = tf.nn.conv2d(layer1_pool, variables['w2'], [1, 1, 1, 1], padding='VALID')
    #print(layer2_conv) #10*10*16
    layer2_actv = tf.nn.relu(layer2_conv + variables['b2'])
    layer2_pool = tf.nn.max_pool(layer2_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    #print(layer2_pool) #5*5*16

    layer3_conv = tf.nn.conv2d(layer2_pool,variables['w3'], [1, 1, 1, 1], padding='VALID')
    layer3_actv = tf.nn.relu(layer3_conv)
    #print(layer3_actv) #1*1*120

    flat_layer = tf.reshape(layer3_actv,[-1,120])
    layer4_fccd = tf.matmul(flat_layer, variables['w4']) + variables['b4']
    layer4_actv = tf.nn.relu(layer4_fccd)
    #print(layer4_actv) #84

    logits = tf.matmul(layer4_actv, variables['w5']) + variables['b5']
    #print(logits) #10

    model = {
        "layer1_conv":layer1_conv,"layer1_actv":layer1_actv,"layer1_pool":layer1_pool,
        "layer2_conv":layer2_conv,"layer2_actv":layer2_actv,"layer2_pool":layer2_pool,
        "layer3_conv":layer3_conv,"layer3_actv":layer3_actv,"flat_layer":flat_layer,
        "layer4_fccd":layer4_fccd,"layer4_actv":layer4_actv,
        "logits":logits
        }
    return model

def inference(logits):
    return tf.nn.softmax(logits)
# -*- coding: utf-8 -*- 
import tensorflow as tf
import numpy as np
import input_data
import LeNet
from PIL import Image

num_labels = 10
batch_size = LeNet.LENET5_BATCH_SIZE
image_width = 28
image_height = 28
image_depth = 1
#number of iterations and learning rate
num_steps = 200001
display_step = 1000
learning_rate = 0.001

#mnist = input_data.read_data_sets(FLAGS.train_dir,FLAGS.fake_data)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#parameters determining the model size

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', './log_dir',
                           """Directory where to write event logs """)

#the datasets
train_dataset = mnist.train
test_dataset = mnist.test.images
test_dataset = np.reshape(test_dataset,[-1,image_width,image_height,1])
test_labels = mnist.test.labels
print(train_dataset.num_examples)

# 保存图片
# image_num = train_dataset.num_examples
# for i in range(image_num):
#     #print(train_dataset.images[i])
#     image=np.reshape(np.array(np.multiply(train_dataset.images[i],255),dtype=np.uint8),[image_width,image_height])
#     #print(image)
#     image=Image.fromarray(image)
#     #print(train_dataset.labels[i])
#     labels = train_dataset.labels[i]
#     num = np.where(labels == np.max(labels))[0][0]
#     #print(num)
#     image.save("pic/train/{0}-{1}.png".format(i+1,num))
#     #exit()

graph = tf.Graph()
with graph.as_default():
    #1) First we put the input data in a tensorflow friendly form. 
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size,image_width,image_height,image_depth))
    tf_train_labels = tf.placeholder(tf.float32, shape= (batch_size,num_labels))
    tf_test_dataset = tf.constant(test_dataset,tf.float32)
    tf_inference_data = tf.placeholder(tf.float32, shape=(1,image_width,image_height,image_depth))
    
    #2) Then, the weight matrices and bias vectors are initialized
    variables = LeNet.variables_lenet5(image_depth = image_depth, num_labels = num_labels)

    #3. The model used to calculate the logits (predicted labels)
    model = LeNet.model_lenet5
    logits = model(tf_train_dataset,variables)
    logits = logits["logits"]
    #4. then we compute the softmax cross entropy between the logits and the (actual) labels
    # 先求出样本的取平均值 Computes softmax cross entropy between `logits` and `labels`
    # 第一步 先对网络最后一层的输出做一个softmax，这一步通常是求取输出属于某一类的概率
    # 第二步 第二步是softmax的输出向量和样本的实际标签做一个交叉熵
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = tf_train_labels))

    #5. The optimizer is used to calculate the gradients of the loss function
    # 梯度下降优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    # 计算概率
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_dataset,variables)["logits"])

    # inference = tf.nn.softmax(model(tf_inference_data,variables))

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized with learning_rate',learning_rate)

    summary_writer = tf.summary.FileWriter(FLAGS.log_dir,
                                            graph=session.graph)
    for step in range(num_steps):
        #Since we are using stochastic gradient descent, we are selecting  small batches from the training dataset,
        #and training the convolutional neural network each time with a batch. 
        batch_data,batch_labels =train_dataset.next_batch(batch_size)
        batch_data = np.reshape(batch_data,[-1,28,28,1])
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels :batch_labels}
        _, l, predictions = session.run([optimizer,loss,train_prediction],feed_dict=feed_dict)


        if step % display_step == 0:
            summary_str = session.run(summary_op,feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)

            train_accuracy = LeNet.accuracy(predictions, batch_labels)
            test_accuracy = LeNet.accuracy(test_prediction.eval(),test_labels)
            #print(step,l,train_accuracy)
            print("step {:d} : loss is {:.2f}, accuracy on training set {:.2f} %,accuracy on testing set {:02.2f} %".format(step, l, train_accuracy,test_accuracy))


    
    # 保存训练模型
    #创建saver对象
    saver = tf.train.Saver()
    #使用saver提供的简便方法去调用 save op
    saver.save(session,"./lenet5.ckpt")
# -*- coding: utf-8 -*- 

import tensorflow as tf
import numpy as np
import input_data
import LeNet
import matplotlib.pyplot as plt
import cv2
import random
import tkinter
from PIL import Image,ImageDraw

image_width = 28
image_height = 28
image_depth = 1
num_labels = 10

class Canvas:
    def __init__(self,root):
        self.root=root
        self.canvas=tkinter.Canvas(root,width=256,height=256,bg='black')
        self.canvas.pack()
        self.image1 = Image.new("RGB", (256, 256), "black")
        self.draw = ImageDraw.Draw(self.image1)
        self.canvas.bind('<B1-Motion>',self.Draw)

    def Draw(self,event):
        self.canvas.create_oval(event.x,event.y,event.x,event.y,outline="white",width = 20)
        self.draw.ellipse((event.x-10,event.y-10,event.x+10,event.y+10),fill=(255,255,255))

def getRandomImgNum(image_width,image_height,image_depth):
    # 随机生成一张数字图片
    randnum=random.randint(0,9)
    image = np.zeros([image_width,image_height,image_depth],dtype=float)
    # 将3写到图片上
    cv2.putText(image,"{0}".format(randnum), (5,20), 
        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 
        thickness = 1, lineType = 1)
    image = np.array(np.multiply(image,1/255),dtype=np.float)
    return image

def predict(inference,image,session):
    maxat=-1
    inference_acc = session.run(inference,feed_dict={data:[image]})
    print(inference_acc[0])
    max = 0.3   # 设置概率达到80以上才认为
    index = 0
    for value in inference_acc[0]:
        if value > max:
            max = value
            maxat = index
        index = index + 1
    return maxat

session=tf.Session()
 #2) Then, the weight matrices and bias vectors are initialized
data = tf.placeholder(tf.float32, shape=(1,image_width,image_height,image_depth))
variables = LeNet.variables_lenet5(image_depth = image_depth, num_labels = num_labels)
model = LeNet.model_lenet5(data,variables)
inference = LeNet.inference(model["logits"])

# 单个预测
saver = tf.train.Saver()
saver.restore(session,"./lenet5.ckpt")

def inference_click():
    img = canvas1.image1
    #print(img)
    img = img.convert('L')
    img = img.resize([image_width,image_height],Image.ANTIALIAS)
    # plt.imshow(np.array(img,dtype=np.uint8))
    # plt.show()
    img = np.array(img,dtype=float)/255

    img = np.reshape(img,[image_width,image_height,image_depth])
    #print(img)
    result = predict(inference,img,session)
    result = int(result)
    if result == -1:
        result = "识别失败"
    
    label2["text"] = str(result)
    
    
    canvas1.canvas.delete("all")
    canvas1.image1 = Image.new("RGB", (256, 256), "black")
    canvas1.draw = ImageDraw.Draw(canvas1.image1)
    
    # #----------------------------------特征可视化-------------------------------
    # conv1,pool1,conv2,pool2=session.run([model["layer1_conv"],model["layer1_pool"],model["layer2_conv"],model["layer2_pool"]],feed_dict={data:[img]})

    # conv1_reshape = session.run(tf.split(conv1,num_or_size_splits=6,axis=3))
    # pool1_reshape = session.run(tf.split(pool1,num_or_size_splits=6,axis=3))

    # conv2_reshape = session.run(tf.split(conv2,num_or_size_splits=16,axis=3))
    # pool2_reshape = session.run(tf.split(pool2,num_or_size_splits=16,axis=3))

    # fig1,ax1 = plt.subplots(nrows=2, ncols=6, figsize = (6,1))
    # for i in range(6):
    #     ax1[0][i].imshow(np.reshape(conv1_reshape[i],[image_width,image_width]))                      # tensor的切片[batch, channels, row, column]
    # plt.title('Conv1 6x28x28')

    # for i in range(6):
    #     ax1[1][i].imshow(np.reshape(pool1_reshape[i],[14,14]))                      # tensor的切片[batch, channels, row, column]
    # plt.title('Pool1 6x14x14')
    

    # fig2,ax2 = plt.subplots(nrows=2, ncols=16, figsize = (16,1))
    # for i in range(16):
    #     ax2[0][i].imshow(np.reshape(conv2_reshape[i],[10,10]))                      # tensor的切片[batch, channels, row, column]
    # plt.title('Conv2 16x10x10')

    # for i in range(16):
    #     ax2[1][i].imshow(np.reshape(pool2_reshape[i],[5,5]))                      # tensor的切片[batch, channels, row, column]
    # plt.title('Pool2 16x5x5')
    
    # plt.show()
    #----------------------------------特征可视化 end-------------------------------
    


root=tkinter.Tk()
root.geometry('300x400')
frame=tkinter.Frame(root,width=256,height=256)
frame.pack_propagate(0)
frame.pack(side='top')
canvas1=Canvas(frame)
    

botton_Inference=tkinter.Button(root,
            text="Inference",         
            width=7,           
            height=1,                
            command=inference_click      
            )

botton_Inference.pack()
label1 = tkinter.Label(root, justify="center",text = "Inference result is")
label1.pack()
label2 = tkinter.Label(root, justify="center")
label2["font"] = ("Arial, 48")
label2.pack()
root.mainloop()
session.close()
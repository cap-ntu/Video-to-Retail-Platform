# Author: Wang Yongjie
# Email:  yongjie.wang@ntu.edu.sg import os
import tensorflow as tf
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

N = 10
M = 44

def get_file_list(directory):
    filelist = []
    for i in os.listdir(directory):
        subpath = os.path.join(directory, i)
        sublist = []
        for j in os.listdir(subpath):
            filename = os.path.join(subpath, j)
            sublist.append(filename)

        filelist.append(sublist)
    return filelist

def build_pos_neg(filelist, num):
    test_samples = []
    for i in range(len(filelist)):
        for j in range(num): # select positive pairs
            positive_pair = []
            num1 = np.random.randint(len(filelist[i]))
            num2 = np.random.randint(len(filelist[i]))
            if num1 == num2:
                num2 = (num2 + 1) % len(filelist[i])

            positive_pair.append(filelist[i][num1])
            positive_pair.append(filelist[i][num2])
            test_samples.append(positive_pair)

    for i in range(len(filelist)):
        for j in range(num): # select negative pairs
            negative_pair = []
            num1 = np.random.randint(len(filelist[i]))
            k = np.random.randint(len(filelist))
            if i == k:
                k = (k + 1) % len(filelist)

            num2 = np.random.randint(len(filelist[k]))
            
            negative_pair.append(filelist[i][num1])
            negative_pair.append(filelist[k][num2])
            test_samples.append(negative_pair)


    return test_samples

def compute_cosine_dis(test_samples, pb_file):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.gfile.GFile(pb_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name ='')

        sess = tf.Session(config = config, graph = graph)


    inputs = sess.graph.get_tensor_by_name('img_inputs:0')
    dropout = sess.graph.get_tensor_by_name('dropout_rate:0')
    feature = sess.graph.get_tensor_by_name("resnet_v1_50/E_BN2/Identity:0")

    distances = []
    for i in range(len(test_samples)):
        img1, img2 = test_samples[i][0], test_samples[i][1]
        mat1 = cv2.imread(img1)
        mat2 = cv2.imread(img2)

        if isinstance(mat1, type(None)):
            continue
        if isinstance(mat2, type(None)):
            continue

        mat1 = (mat1 - 127.5) / 128
        mat2 = (mat2 - 127.5) / 128
        feature1 = sess.run(feature, feed_dict = {inputs: [mat1], dropout:1} )
        feature2 = sess.run(feature, feed_dict = {inputs: [mat2], dropout:1} )
        cosine_dist = np.dot(feature1[0], feature2[0]) / (np.linalg.norm(feature1[0]) * np.linalg.norm(feature2[0]))

        distances.append(cosine_dist)

    return distances


def compute_pr(distances):
    minimize = min(distances)
    maximize = max(distances)
    print('max distance', maximize)
    print('min distance', minimize)

    thresholds = np.linspace(minimize, maximize, 20)

    test_results = []

    for i in range(len(thresholds)):
        tmp = []
        threshold = thresholds[i]
        results = distances >= threshold
        results = np.reshape(results, (2, N * M))
        tp = np.sum(results[0])
        fp = np.sum(results[1])
        acc = float(tp) / (tp + fp)
        recall = float(tp) / (N * M)
        print("threshold\t%f\tacc\t %f\trecall\t%f"%(threshold, acc, recall))
        tmp.append(acc)
        tmp.append(recall)
        tmp.append(threshold)
        test_results.append(tmp)

    return test_results

def draw(input_data, save_name):
    x = input_data[:, 0]
    y = input_data[:, 1]
    num = input_data[:, 2]
    plt.plot(x, y, 'o')
    plt.xlabel('precision')
    plt.ylabel('recall')
    zz = list(zip(x, y, num))
    length = len(zz)
    for i in range(length):
        plt.text(zz[i][0], zz[i][1], '%.2f'%num[i])
    plt.legend()
    plt.draw()
    plt.savefig(save_name)


def get_threshold(directory, pb_file, save_name):
    filelist = get_file_list(directory)
    global M 
    M = len(os.listdir(directory))
    test_samples = build_pos_neg(filelist, N)
    distances = compute_cosine_dis(test_samples, pb_file)
    test_results = compute_pr(distances)
    test_results = np.array(test_results)
    print(test_results)
    draw(test_results, save_name)


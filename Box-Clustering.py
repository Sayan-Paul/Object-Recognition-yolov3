#! /usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
from tqdm import tqdm
from PIL import Image


with open('data/OpenImages/oid.json', 'r') as oid:
    oid_data = json.load(oid)

with open('data/OpenImages/label_map.txt', 'r') as label_map_file:
    label_map = eval(label_map_file.read())

wh = list()

for split in oid_data:
    for img_name in tqdm(oid_data[split]['images'], desc="Reading " + split):
        width, height = Image.open(oid_data[split]['images'][img_name]).size
        for obj in oid_data[split]['boxes'][img_name]:
            w = abs(float(obj[3]) - float(obj[1])) / width  # make the width range between [0,GRID_W)
            h = abs(float(obj[4]) - float(obj[2])) / height  # make the width range between [0,GRID_H)
            wh.append([w, h])

wh = np.array(wh)
print("Clustering feature data is ready. shape = (N object, width and height) =  {}".format(wh.shape))

plt.figure(figsize=(10,10))
plt.scatter(wh[:,0],wh[:,1],alpha=0.3)
plt.title("Clusters",fontsize=20)
plt.xlabel("normalized width", fontsize=20)
plt.ylabel("normalized height", fontsize=20)
plt.show()


def iou(box, clusters):
    '''
    :param box:      np.array of shape (2,) containing w and h
    :param clusters: np.array of shape (N cluster, 2) 
    '''
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def kmeans(boxes, k, dist=np.median, seed=1):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))  ## N row x N cluster
    last_clusters = np.zeros((rows,))

    np.random.seed(seed)

    # initialize the cluster centers to be k items
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        # Step 1: allocate each item to the closest cluster centers
        for icluster in range(k):
            distances[:, icluster] = 1 - iou(clusters[icluster], boxes)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        # Step 2: calculate the cluster centers as mean (or median) of all the cases in the clusters.
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters, nearest_clusters, distances


kmax = 10
dist = np.mean
results = {}

for k in range(2, kmax):
    clusters, nearest_clusters, distances = kmeans(wh, k, seed=2, dist=dist)
    WithinClusterMeanDist = np.mean(distances[np.arange(distances.shape[0]), nearest_clusters])
    result = {"clusters": clusters,
              "nearest_clusters": nearest_clusters,
              "distances": distances,
              "WithinClusterMeanDist": WithinClusterMeanDist}
    print("{:2.0f} clusters: mean IoU = {:5.4f}".format(k, 1 - result["WithinClusterMeanDist"]))
    results[k] = result


def plot_cluster_result(plt, clusters, nearest_clusters, WithinClusterSumDist, wh, k):
    for icluster in np.unique(nearest_clusters):
        pick = nearest_clusters == icluster
        c = current_palette[icluster]
        plt.rc('font', size=8)
        plt.plot(wh[pick, 0], wh[pick, 1], "p",
                 color=c,
                 alpha=0.5, label="cluster = {}, N = {:6.0f}".format(icluster, np.sum(pick)))
        plt.text(clusters[icluster, 0],
                 clusters[icluster, 1],
                 "c{}".format(icluster),
                 fontsize=20, color="red")
        plt.title("Clusters=%d" % k)
        plt.xlabel("width")
        plt.ylabel("height")
    plt.legend(title="Mean IoU = {:5.4f}".format(WithinClusterSumDist))


current_palette = list(sns.xkcd_rgb.values())

figsize = (15, 35)
count = 1
fig = plt.figure(figsize=figsize)
for k in range(5, 10):
    result = results[k]
    clusters = result["clusters"]
    nearest_clusters = result["nearest_clusters"]
    WithinClusterSumDist = result["WithinClusterMeanDist"]

    ax = fig.add_subplot(kmax / 2, 2, count)
    plot_cluster_result(plt, clusters, nearest_clusters, 1 - WithinClusterSumDist, wh, k)
    count += 1
plt.show()

anchors = sorted(clusters, key=lambda x: x[0] * x[1])
anchor_string = ""
for anchor in anchors:
    anchor_string += '{} {} '.format(round(anchor[0], 4), round(anchor[1], 4))
anchor_string = anchor_string[:-2]

with open('data/oid_anchors.txt', 'w') as anchor_file:
    anchor_file.write(anchor_string)

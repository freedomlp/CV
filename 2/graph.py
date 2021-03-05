from random import random
from matplotlib import pyplot as plt
import numpy as np
import cv2

# 定义顶点类
class Node:
    def __init__(self, parent, priority = 0, size = 1):
        self.parent = parent
        self.priority = priority
        self.size = size

# 定义森林类
class Forest:
    def __init__(self, num_nodes):
        self.num = num_nodes
        self.nodes = [Node(i) for i in range(num_nodes)]
        self.roots = [i for i in range(num_nodes)]

    def size(self, i):
        return self.nodes[i].size

    def find(self, n):
        temp = n
        while temp != self.nodes[temp].parent:
            temp = self.nodes[temp].parent


        self.nodes[n].parent = temp
        return temp

    def merge(self, a, b):
        # 根据两个节点的优先级决定谁来当母顶点
        if self.nodes[a].priority > self.nodes[b].priority:
            self.roots.remove(self.nodes[b].parent)
            self.nodes[b].parent = a
            self.nodes[a].size = self.nodes[a].size + self.nodes[b].size
        else:
            self.roots.remove(self.nodes[a].parent)
            self.nodes[a].parent = b
            self.nodes[b].size = self.nodes[b].size+ self.nodes[a].size

            if self.nodes[a].priority == self.nodes[b].priority:
                self.nodes[b].priority = self.nodes[b].priority + 1

        # 合并后分割的区域数量减1
        self.num = self.num - 1

def diff(img, x1, y1, x2, y2):
    res = np.sum((img[x1, y1] - img[x2, y2]) ** 2)
    return np.sqrt(res)

def threshold(size, const):
    return (const * 1.0 / size)

# 创建一条边，由两个点的位置和边的权重组成
def create_edge(img, width, x1, y1, x2, y2):
    vertex1 = x1 * width + y1
    vertex2 = x2 * width + y2
    weight = diff(img, x1, y1, x2, y2)
    return (vertex1, vertex2, weight)

# 创建图
def build_graph(img):
    height = img.shape[0]
    width = img.shape[1]
    graph_edges = []
    
    for x in range(height):
        for y in range(width):
            if x > 0:   
                graph_edges.append(create_edge(img, width, x, y, x-1, y))
 
            if y > 0:
                graph_edges.append(create_edge(img, width, x, y, x, y-1))

            if x > 0 and y > 0:
                graph_edges.append(create_edge(img, width, x, y, x-1, y-1))

            if x > 0 and y < (width-1):
                graph_edges.append(create_edge(img, width, x, y, x-1, y+1))
                    
    return graph_edges

# 初次分割后的图像，对于其中定点数均于min_size的两个相邻区域，进行合并
def remove_small_components(forest, graph, min_size):
    for edge in graph:
        a = forest.find(edge[0])
        b = forest.find(edge[1])

        if a != b and (forest.size(a) < min_size or forest.size(b) < min_size):
            forest.merge(a, b)

    return  forest

# 分割函数
def segment_graph(graph_edges, num_nodes, const, min_size):
    # 初始化
    forest = Forest(num_nodes)
    weight = lambda edge: edge[2]
    sorted_graph = sorted(graph_edges, key = weight)
    thresholds = [ threshold(1, const) for _ in range(num_nodes) ]

    # 合并
    for edge in sorted_graph:
        parent_a = forest.find(edge[0])
        parent_b = forest.find(edge[1])
        a_condition = weight(edge) <= thresholds[parent_a]
        b_condition = weight(edge) <= thresholds[parent_b]

        if parent_a != parent_b and a_condition and b_condition:
            forest.merge(parent_a, parent_b)
            a = forest.find(parent_a)
            thresholds[a] = weight(edge) + threshold(forest.nodes[a].size, const)

    return remove_small_components(forest, sorted_graph, min_size)

def gt(forest, id):
    gt_img = cv2.imread('../data/gt/' + str(id) + '.png', cv2.IMREAD_GRAYSCALE)
    height = gt_img.shape[0]
    width = gt_img.shape[1]
    new_img = np.zeros(gt_img.shape, dtype = np.uint8)
    
    gt_root = {}
    for root in forest.roots:
        size = 0
        mask = 0
        
        for i in range(height):
            for j in range(width):
                n_root = forest.find(i * width + j)
                if n_root == root:
                    size += 1
                    if gt_img[i][j] == 255:
                        mask += 1
                        
        # 区域 50%以上的像素在GT中标为255，则将区域认定为前景
        if mask * 2 > size:
            gt_root[root] = 1
            for i in range(height):
                for j in range(width):
                    n_root = forest.find(i * width + j)
                    if n_root == root:
                        new_img[i][j] = 255
        else:
            gt_root[root] = 0
    
    return gt_root
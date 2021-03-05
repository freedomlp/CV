from graph import build_graph, segment_graph
from random import random
from matplotlib import pyplot as plt
import numpy as np
import cv2

def generate_image(forest, height, width):
    img_matrix = np.zeros((height, width), dtype=np.uint8)
    img = cv2.cvtColor(img_matrix, cv2.COLOR_GRAY2BGR)
    random_color = lambda: (int(random()*255), int(random()*255), int(random()*255))
    colors = [random_color() for i in range(height * width)]

    for i in range(height):
        for j in range(width):
            comp = forest.find(i * width + j)
            img[i,j] = colors[comp]
    
    return img

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
            
    # 计算IOU
    intersection = 0
    union = 0
    for i in range(height):
        for j in range(width):
            if gt_img[i][j] == 255 or new_img[i][j] == 255:
                union += 1
            if gt_img[i][j] == 255 and new_img[i][j] == 255:
                intersection += 1
    res = intersection * 1.0 / union
    print("IOU为：", res)
    
    return new_img

if __name__ == '__main__':
    for i in range(1, 1001):
        if i % 100 == 66:
            print("当前处理图片", str(i) + '.png')
            pic_path = '../data/imgs/' + str(i) + '.png'
            image = cv2.imread(pic_path)
            
            # 高斯滤波
            source = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            smooth = cv2.GaussianBlur(source, (3, 3), sigmaX = 1.0, sigmaY = 1.0)
            
            graph_edges = build_graph(smooth)
            forest = segment_graph(graph_edges, image.shape[0] * image.shape[1], 200, 20)
            seg_img = generate_image(forest, image.shape[0], image.shape[1])
            new_img = gt(forest, i)
            
            # 输出结果并保存
            plt.figure()
            plt.subplot(121)
            plt.imshow(seg_img)
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(new_img, cmap=plt.cm.gray)
            plt.axis('off')
            plt.savefig('result/' + str(i) + '.png')
            plt.show()
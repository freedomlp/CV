import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
from math import ceil

for i in range(1, 1001):
    if i % 100 == 66:
        # 读入需要处理的图片
        pic_path = '../data/imgs/' + str(i) + '.png'
        print("当前处理图片", str(i) + '.png')
        img = cv2.imread(pic_path)
        img1 = img.copy()   
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # 计算图片高度和宽度
        height = gray.shape[0]
        width = gray.shape[1]
 
        # 计算特征点和特征向量
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        print("原特征点的个数：", len(kp))
        cv2.drawKeypoints(gray, kp, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(img1)

        # 采用PCA算法将特征降维至 10 维
        print("原特征向量的维数：", des.shape)
        pca = PCA(n_components = 10)
        newdes = pca.fit_transform(des)
        print("PCA降维后特征向量的维数：", newdes.shape)

        # 以每个 sift 特征点为中心截取 16*16 的 patch，删除掉越界的patch及对应的特征值和特征向量
        patchs = []
        final_kp = []
        final_des = []
        for j in range(len(kp)):
            x = int(kp[j].pt[0])
            y = int(kp[j].pt[1])
            if x - 7 >= 0 and x + 8 < width and y - 7 >= 0 and y + 8 < height:
                patchs.append(img[y - 7 : y + 9, x - 7 : x + 9])
                final_kp.append(kp[j])
                final_des.append(newdes[j])
                
        desMat = np.array(final_des)
        print("patchs的数量：", len(patchs))
        print("删减后特征向量的维数：", desMat.shape)
        
        # 计算归一化颜色直方图
        colorHistlist = []
        for j in range(len(patchs)):
            b,g,r = cv2.split(patchs[j])       # 通道拆分
            
            # 对三个通道进行直方图均衡化
            b = cv2.equalizeHist(b)
            g = cv2.equalizeHist(g)
            r = cv2.equalizeHist(r)
            testimg = cv2.merge([b, g, r])     # 通道合并
            hist = cv2.calcHist([testimg], [0, 1 ,2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
            finalhist = hist.reshape(64)
            colorHistlist.append(finalhist)
            
        colorHistMat = np.array(colorHistlist)
        print("patchs的归一化颜色直方图维数:", colorHistMat.shape)
        
        # 将 sift 特征和颜色直方图特征拼接， 每个 sift 特征点表示为 74 维向量
        siftMat = np.concatenate((desMat, colorHistMat), axis = 1)
        print("最终特征向量的维数：", siftMat.shape)
        
        # 采用 k-means 聚类算法将该图像所有 sift 特征点聚为 3 类
        clf = KMeans(n_clusters=3)
        cls = clf.fit(siftMat)
        labels = cls.labels_
        class0 = []
        class1 = []
        class2 = []
        for j in range(len(patchs)):
            if labels[j] == 0:
                class0.append(patchs[j])
            elif labels[j] == 1:
                class1.append(patchs[j])
            elif labels[j] == 2:
                class2.append(patchs[j])
                
        
        print("第0簇的特征点个数", len(class0))
        print("第1簇的特征点个数", len(class1))
        print("第2簇的特征点个数", len(class2))
        #print(class0)
        
        
        
        # 组合形成展示图
        row0 = ceil(len(class0) / 10)
        row1 = ceil(len(class1) / 10)
        row2 = ceil(len(class2) / 10)
        row = max(row0, row1, row2)
        plt.figure(num='image', figsize=(32/2,row/2))
        
        k = 0
        for j in range(len(class0)):
            plt.subplot(row, 32, j + k + 1)
            plt.imshow(class0[j])
            plt.axis('off')
            plt.subplots_adjust(wspace=0,hspace=0)
            if (j + 1) % 10 == 0:
                k += 22
                
        k = 11
        for j in range(len(class1)):
            plt.subplot(row, 32, j + k + 1)
            plt.imshow(class1[j])
            plt.axis('off')
            plt.subplots_adjust(wspace=0,hspace=0)
            if (j + 1) % 10 == 0:
                k += 22
            
        k = 22
        for j in range(len(class2)):
            plt.subplot(row, 32, j + k + 1)
            plt.imshow(class2[j])
            plt.axis('off')
            plt.subplots_adjust(wspace=0,hspace=0)
            if (j + 1) % 10 == 0:
                k += 22

        plt.savefig('result/' + str(i) + '.png')
        plt.show()
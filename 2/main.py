import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
import graph
import cv2

# 计算区域颜色直方图
def local(img, forest, c):
    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros((height, width), dtype="uint8")
    
    for i in range(height):
        for j in range(width):
            if forest.find(i * width + j) == c:
                mask[i][j] = 1
                
    # 只处理mask为1的点，即同一区域的点
    hist = cv2.calcHist([img], [0,1,2], mask, [8,8,8], [0,256,0,256,0,256])
    return hist.ravel()

def train():
    features = []
    roots = []
    
    # 用前10张图作为训练集
    for i in range(1, 21):
        print("当前训练图片", str(i) + '.png')
        pic_path = '../data/imgs/' + str(i) + '.png'
        image = cv2.imread(pic_path, cv2.IMREAD_COLOR)
        
        # 高斯滤波
        source = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.GaussianBlur(source, (3, 3), sigmaX = 1.0, sigmaY = 1.0)
        
        height = image.shape[0]
        width = image.shape[1]
        
        graph_edges = graph.build_graph(image)
        forest = graph.segment_graph(graph_edges, height * width, 200, 20)
        gt_root = graph.gt(forest, i)

        # 计算全图颜色直方图
        hist = cv2.calcHist([image], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
        g_feature = hist.ravel().tolist()
    
        for c in forest.roots:
            l_feature = local(image, forest, c)
            # 拼接区域颜色直方图和全图颜色直方图
            feature = np.concatenate((g_feature, l_feature), axis=0)
            roots.append(gt_root[c])
            features.append(feature)
            
    X = np.array(features)
    Y = np.array(roots)
    
    # PCA降维至50维
    pca = PCA(n_components=50)
    new_X = pca.fit_transform(X)
    
    clf = svm.SVC(gamma = 0.001, C = 100.)
    print("训练开始")
    model = clf.fit(new_X, Y)
    print("训练结束")
    return model

def accuracy(model, X, Y):
    size, _ = X.shape
    result = model.predict(X)
    count = 0
    
    for i in range(size):
        if result[i] == Y[i]:
            count += 1
            
    right = count / size
    print("预测准确率为:", right)

if __name__ == "__main__":
    model = train()
    features = []
    roots = []
    for i in range(1, 1001):
        if i % 100 == 66:
            print("当前测试图片", str(i) + '.png')
            pic_path = '../data/imgs/' + str(i) + '.png'
            image = cv2.imread(pic_path, cv2.IMREAD_COLOR)
            # 高斯滤波
            source = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.GaussianBlur(source, (3, 3), sigmaX = 1.0, sigmaY = 1.0)
            
            height = image.shape[0]
            width = image.shape[1]
            
            graph_edges = graph.build_graph(image)
            forest = graph.segment_graph(graph_edges, height * width, 200, 20)
            gt_root = graph.gt(forest, i)
            
            # 计算全图颜色直方图
            hist = cv2.calcHist([image], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
            g_feature = hist.ravel().tolist()
        
            for c in forest.roots:
                l_feature = local(image, forest, c)
                # 拼接区域颜色直方图和全图颜色直方图
                feature = np.concatenate((g_feature, l_feature), axis=0)
                roots.append(gt_root[c])
                features.append(feature)

    X = np.array(features)
    Y = np.array(roots)

    # PCA降维至50维
    pca = PCA(n_components = 50)
    new_X = pca.fit_transform(X)
    accuracy(model, new_X, Y)
    
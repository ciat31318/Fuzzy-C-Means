import cv2 as cv
import numpy as np
import random
import math

# 取得圖片訊息
img = cv.imread( '17.jpg' )
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()
px = img[:]
#print(px.shape)
new_px = px.reshape((1,500*500,3))
#print(new_px)

# Number of Clusters 
k = 10
# Fuzzy paremeter
m =3
# Number of data point 
n = 500*500
# 遞迴次數限制
Max_iter = 10

# 建立每個點的初始權重
def init_ratioMatrix():
    ratio_matrix = []
    for i in range(n):
        random_num = np.array([ random.random() for j in range(k) ])
        random_num = random_num/np.sum(random_num)
        ratio_matrix.append(random_num)
    return ratio_matrix

# 初始化 Cluster centers
def init_center( new_px, k ):
    cluster_centers = []
    while len(cluster_centers)<k:
        i = random.randint(0,500*500)
        cluster_centers.append(new_px[0,i])
    return cluster_centers
    
    

# 計算 Cluster center
def cal_Clustercenter( ratio_matrix ):
    ratio_matrix = np.array(ratio_matrix)
    cluster_mem_val = list(zip(*ratio_matrix))
    cluster_centers = []
    for j in range(k):
        x = list(cluster_mem_val[j])
        temp = [e**m for e in x]
        denominator = sum(temp)
        temp_num = []
        for i in range(n):
            prod = [ temp[i] * val for val in new_px[0,i] ]
            temp_num.append(prod)
        numerator = map( sum, zip( *temp_num ) )
        center = [ z/denominator for z in numerator ]
        cluster_centers.append(center)
    return cluster_centers

# 權重更新
def update_Ratiomatrix( ratio_matrix, cluster_centers ):
    p = float( 2/(m-1) )
    for i in range(n):
        distances = [ np.linalg.norm( np.array( new_px[0,i] - cluster_centers[j] ))  for j in range(k)  ]
        distances = np.array(distances)
        distances[ distances <=1 ] = 1
        for j in range(k):
            den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(k)])
            ratio_matrix[i][j] = float(1/den)
    return ratio_matrix

if __name__ == '__main__':
    ratio_matrix = init_ratioMatrix()
    current = 0
    cluster_centers = init_center( new_px, k )
    while current < Max_iter : 
        tmp = ratio_matrix.copy() 
        ratio_matrix = update_Ratiomatrix( ratio_matrix, cluster_centers )
        
        cluster_centers = cal_Clustercenter( ratio_matrix )
        current += 1
        #print(current)
        
        #tmp = np.array(tmp)
        ratio_matrix = np.array( ratio_matrix)
        '''
        error = -1
        for i in range(n):
            error = max(error,  np.linalg.norm(ratio_matrix[i]-tmp[i]) ) 
        print(error)
        if error<0.005:
            break
        '''
        


    ratio_matrix = ratio_matrix.tolist()
    for i in range(n):
        a = ratio_matrix[i].index(max( ratio_matrix[i] )) 
        img[i//500][i%500] = cluster_centers[a]
    
    cv.imshow('img',img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    #print(cluster_centers)
















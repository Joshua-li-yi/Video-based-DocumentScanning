import cv2
import numpy as np


class matchers:
    def __init__(self):
        self.surf = cv2.xfeatures2d.SURF_create()
        # self.surf = cv2.xfeatures2d.SIFT_create()
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        # 寻找最近邻近似匹配
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, i1, i2, direction=None):
        imageSet1 = self.getSURFFeatures(i1)
        imageSet2 = self.getSURFFeatures(i2)
        print("Direction : ", direction)
        matches = self.flann.knnMatch(
            imageSet2['des'],
            imageSet1['des'],
            k=2
        )
        # queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
        # trainIdx：样本图像的特征点描述符下标,同时也是描述符对应特征点的下标。
        # distance：代表这怡翠匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
        good = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good.append((m.trainIdx, m.queryIdx))

        if len(good) > 4:
            pointsCurrent = imageSet2['kp']
            pointsPrevious = imageSet1['kp']

            matchedPointsCurrent = np.float32([pointsCurrent[i].pt for (__, i) in good])
            matchedPointsPrev = np.float32([pointsPrevious[i].pt for (i, __) in good])
            #  计算单应矩阵
            #  CV_RANSAC  - 基于RANSAC的鲁棒方法
            #  第四个参数取值范围在1到10，绝一个点对的阈值。原图像的点经过变换后点与目标图像上对应点的误差
            H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
            return H
        return None

    def getSURFFeatures(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kp, des = self.surf.detectAndCompute(gray, None)
        return {'kp': kp, 'des': des}

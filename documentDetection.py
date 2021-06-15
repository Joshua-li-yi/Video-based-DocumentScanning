from util import is_main_object, resize
from similarImg import is_similar_img
import cv2
import numpy as np
import os
import time

VIDEO_PATH = "video/test2.mp4"
FRAME_DIR_PATH = "frame/test2/"
RESULT_PATH = 'result'


def edge_detection(img):
    orig = img.copy()
    try:
        image = img.copy()
        cv2.imshow("orig", orig)
        # 预处理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(blur, threshold1=5, threshold2=140, L2gradient=True)
        cv2.imshow("edge", edged)
        # cv2.waitKey(0)
        # *************  轮廓检测 ****************
        # 轮廓检测
        _, contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        if len(cnts) < 4:
            return
        screenCnt = np.zeros([4, 1, 2])
        # 遍历轮廓
        for c in cnts:
            # 计算轮廓近似
            peri = cv2.arcLength(c, True)
            # c表示输入的点集，epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # 4个点的时候就拿出来
            if len(approx) == 4:
                screenCnt = approx
                break
        # print("===============")
        contours_point_list = np.reshape(screenCnt, [4, 2])
        for point in contours_point_list:
            cv2.circle(edged, tuple(point), 2, (255, 0, 0), 4)
        cv2.imshow("contour point", edged)
        # cv2.waitKey(0)
        return contours_point_list
    except:
        print("except")
        return [0]


def edge_detection2(img_path):
    # 读取输入
    img = cv2.imread(img_path)
    orig = img.copy()
    image = img.copy()
    cv2.imshow("original image", orig)
    # 预处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blur, threshold1=5, threshold2=140, L2gradient=True)
    cv2.imshow("edge", edged)
    # 轮廓检测
    _, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    screen_cnt = np.zeros([4, 1, 2])
    # 遍历轮廓
    for c in cnts:
        # 计算轮廓近似
        peri = cv2.arcLength(c, True)
        # c表示输入的点集，epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # 4个点的时候就拿出来
        if len(approx) == 4:
            screen_cnt = approx
            break
    # contours_point_list = np.reshape(screen_cnt, [4, 2])
    # for point in contours_point_list:
    #     cv2.circle(edged, tuple(point), 2, (255, 0, 0), 4)
    # cv2.imshow("contour point", edged)
    # cv2.waitKey(0)
    # res = cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    # res = cv2.drawContours(image, cnts[0], -1, (0, 255, 0), 2)
    # show(orig)
    return orig, screen_cnt


def order_points(pts):
    # 一共四个坐标点
    rect = np.zeros((4, 2), dtype='float32')
    # 按顺序找到对应的坐标0123 分别是左上，右上，右下，左下
    # 计算左上，由下
    # numpy.argmax(array, axis) 用于返回一个numpy数组中最大值的索引值
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算右上和左
    # np.diff()  沿着指定轴计算第N维的离散差值  后者-前者
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# 透视变换
def four_point_transform(image, pts):
    # 获取输入坐标点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # 计算输入的w和h的值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变化后对应坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]],
        dtype='float32')

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # cv2.imshow("warped", warped)
    # cv2.waitKey(0)

    # 返回变换后的结果
    return warped


class DocumentDetectionRolling:
    def __init__(self, video_path, frame_dir_path, result_path):
        start_time = time.time()
        self.video_path = video_path
        self.frame_dir_path = frame_dir_path
        self.result_path = result_path
        self.process_video()
        self.image_process()
        print(f"time = {time.time() - start_time} s")
        pass

    def process_video(self):
        capture = cv2.VideoCapture(self.video_path)

        success = False
        count = 0
        if capture.isOpened():
            success, frame = capture.read()
            print("Start decoding file %s..." % self.video_path)
            count += 1
        else:
            print("Open %s failure!" % self.video_path)
        self.last_frame_list = []
        self.last_frame_list.append(frame)
        frame = resize(frame, 400)
        contours_point_list = edge_detection(frame)
        threshold, _ = is_main_object(frame, contours_point_list)
        print("threshold", threshold)
        interval = 0
        if success:
            success, frame = capture.read()
            while success:
                frame = resize(frame, 500)
                contours_point_list = edge_detection(frame)
                if contours_point_list is None:
                    print(2)
                else:
                    print(1)
                interval += 1
                print(is_main_object(frame, contours_point_list, threshold=0.4))
                print(is_similar_img(self.last_frame_list[count - 1], frame, threshold=0.20))
                if (is_main_object(frame, contours_point_list, threshold=0.4)[1] and not is_similar_img(
                        self.last_frame_list[count - 1], frame, threshold=0.20) and interval > 43) or count == 1:
                    cv2.imwrite(self.frame_dir_path + '%d.jpg' % (count), frame)
                    count += 1
                    interval = 0
                    self.last_frame_list.append(frame)

                success, frame = capture.read()
        capture.release()
        print("Encoding file %s success!" % self.video_path)
        pass

    def image_process(self):
        img_path_list = [os.path.join(self.frame_dir_path, img_name) for img_name in os.listdir(self.frame_dir_path)]
        for i, img_path in enumerate(img_path_list):
            print(f'processing {img_path}')
            orig, screen_cnt = edge_detection2(img_path)
            print("screenCnt", screen_cnt)
            print("reshape", np.reshape(screen_cnt, [4, 2]))
            # screenCnt 为四个顶点的坐标值，但是我们这里需要将图像还原，即乘以以前的比率
            # 透视变换  这里我们需要将变换后的点还原到原始坐标里面
            warped = four_point_transform(orig, screen_cnt.reshape(4, 2))
            # 二值处理
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 112, 255, cv2.THRESH_BINARY)[1]
            cv2.imshow("thresh", thresh)
            # cv2.waitKey(0)
            cv2.imwrite(os.path.join(self.result_path, f'{i + 1}.jpg'), thresh)
        print('finish images processing')
        # thresh_resize = resize(thresh, height=400)
        # dst = thresh_resize.copy()
        # cv2.Laplacian(thresh_resize, cv2.CV_16S, dst)
        pass


if __name__ == '__main__':
    d = DocumentDetectionRolling(VIDEO_PATH, FRAME_DIR_PATH, RESULT_PATH)

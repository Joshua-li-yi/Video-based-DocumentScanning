import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO


def edge_detection(img_path, kernel_size=3, threshold1=5, threshold2=140, is_L2gradien=True):
    # 读取输入
    img = cv2.imread(img_path)
    orig = img.copy()
    image = img.copy()
    st.header("原图像")
    st.image(orig)
    col1, col2 = st.beta_columns(2)

    # 预处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    edged = cv2.Canny(blur, threshold1=threshold1, threshold2=threshold2, L2gradient=is_L2gradien)
    with col1:
        st.header("canny 图像")
        st.image(edged)
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
    contours_point_list = np.reshape(screen_cnt, [4, 2])
    for point in contours_point_list:
        cv2.circle(edged, tuple(point), 4, (255, 0, 0), 10)
    with col2:
        st.header("四个轮廓点")
        st.image(edged)
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
    # 返回变换后的结果
    return warped


st.title("单文档扫描")

uploaded_file = st.file_uploader("Choose a file")
img_bytes = None
if uploaded_file is not None:
    img_bytes = uploaded_file.getvalue()
    bytes_stream = BytesIO(img_bytes)
    capture_img = Image.open(bytes_stream)
    capture_img = cv2.cvtColor(np.asarray(capture_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite("process_img.jpg", capture_img)

kernel_size = st.sidebar.slider(label='kernel_size', min_value=1, max_value=11, step=2, value=3)
threshold1 = st.sidebar.slider(label='threshold1', min_value=1, max_value=255, step=1, value=5)
threshold2 = st.sidebar.slider(label='threshold2', min_value=1, max_value=255, step=1, value=140)
is_L2gradien = st.sidebar.slider(label='is_L2gradien', min_value=0, max_value=1, step=1, value=1)
binary_threshold = st.sidebar.slider(label='binary_threshold', min_value=1, max_value=255, step=1, value=122)
binary_maxvalue = st.sidebar.slider(label='binary_maxvalue', min_value=1, max_value=255, step=1, value=255)
img_path = 'process_img.jpg'

process_img_bt = st.button("处理图像")
if process_img_bt is not None:
    orig, screen_cnt = edge_detection(img_path, kernel_size, threshold1, threshold2, is_L2gradien)
    # screenCnt 为四个顶点的坐标值，但是我们这里需要将图像还原，即乘以以前的比率
    # 透视变换  这里我们需要将变换后的点还原到原始坐标里面
    warped = four_point_transform(orig, screen_cnt.reshape(4, 2))
    col1, col2 = st.beta_columns(2)

    with col1:
        st.header("透视变换后的图像")
        st.image(warped)
    # 二值处理
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, binary_threshold, binary_maxvalue, cv2.THRESH_BINARY)[1]
    with col2:
        st.header("二值处理后的图像")
        st.image(thresh)

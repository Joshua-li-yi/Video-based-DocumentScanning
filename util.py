import cv2


def triangle_area(A, B, C):
    """
    计算三角形的面积
    公式:abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))/2;
    :param A: [x1, y1]
    :param B: [x2, y2]
    :param C: [x3, y3]
    :return:
    """
    return abs(A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1])) / 2


def quadrilateral_area(point_list):
    """
    计算任意四边形的面积
    :param point_list: shape： 4*2 的list
    :return:
    """
    A = point_list[0]
    B = point_list[1]
    C = point_list[2]
    D = point_list[3]
    S1 = triangle_area(A, B, C)
    S2 = triangle_area(A, C, D)
    return S1 + S2


def is_main_object(img, point_list, threshold=0.5):
    """
    判断是否是主要物体
    :param img: imread 之后的img Mat
    :param point_list: shape： 4*2 的list
    :param threshold: 阈值
    :return:
    """
    try:
        rate = quadrilateral_area(point_list) / (img.shape[0] * img.shape[1])
        if rate > threshold:
            return rate, True
        return rate, False
    except:
        return 0, False


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

import numpy as np
import cv2
from matchers import matchers
import time
import os
import time

class Stitch:
    def __init__(self, images_path):
        self.path = images_path
        filenames = [os.path.join(images_path, each) for each in os.listdir(images_path)]
        print(filenames)
        self.images = [cv2.resize(cv2.imread(each), (480, 320)) for each in filenames]
        self.count = len(self.images)
        self.left_list, self.right_list, self.center_im = [], [], None
        self.matcher_obj = matchers()
        self.prepare_lists()

    def prepare_lists(self):
        print("Number of images : %d" % self.count)
        self.centerIdx = self.count / 2
        print("Center index image : %d" % self.centerIdx)
        self.center_im = self.images[int(self.centerIdx)]
        for i in range(self.count):
            if (i <= self.centerIdx):
                self.left_list.append(self.images[i])
        else:
            self.right_list.append(self.images[i])
        print("Image lists prepared")

    def leftshift(self):
        a = self.left_list[0]
        num = 1
        for b in self.left_list[1:]:
            print(f"==============process left img:{num} ==============")
            num += 1
            H = self.matcher_obj.match(a, b, 'left')
            print("Homography is : ", H)
            # 矩阵求逆
            xh = np.linalg.inv(H)
            # print("Inverse Homography :", xh)
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]));
            ds = ds / ds[-1]
            print("final ds=>", ds)
            f1 = np.dot(xh, np.array([0, 0, 1]))
            f1 = f1 / f1[-1]
            xh[0][-1] += abs(f1[0])
            xh[1][-1] += abs(f1[1])
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
            offsety = abs(int(f1[1]))
            offsetx = abs(int(f1[0]))
            dsize = (abs(int(ds[0]) + offsetx), abs(int(ds[1]) + offsety))
            print("image dsize =>", dsize)
            tmp = cv2.warpPerspective(a, xh, dsize)
            cv2.imshow("warped", tmp)
            # cv2.waitKey()
            # 拼接
            tmp[offsety:b.shape[0] + offsety, offsetx:b.shape[1] + offsetx] = b
            a = tmp
        self.left_image = tmp
        pass

    def rightshift(self):
        num = 0
        for each in self.right_list:
            print(f"==============process right img:{num} ================")
            num += 1
            H = self.matcher_obj.match(self.left_image, each, 'right')
            print("Homography :", H)
            txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
            txyz = txyz / txyz[-1]
            dsize = (abs(int(txyz[0]) + self.left_image.shape[1]), abs(int(txyz[1]) + self.left_image.shape[0]))
            tmp = cv2.warpPerspective(each, H, dsize)
            tmp = self.mix_and_match(self.left_image, tmp)
            print("tmp shape", tmp.shape)
            print("self.leftimage shape=", self.left_image.shape)
        self.right_image = tmp
        pass

    def mix_and_match(self, leftImage, warpedImage):
        i1y, i1x = leftImage.shape[:2]
        i2y, i2x = warpedImage.shape[:2]
        print(leftImage[-1, -1])

        t = time.time()
        black_l = np.where(leftImage == np.array([0, 0, 0]))
        black_wi = np.where(warpedImage == np.array([0, 0, 0]))
        print(time.time() - t)
        print(black_l[-1])

        for i in range(0, i1x):
            for j in range(0, i1y):
                try:
                    if (np.array_equal(leftImage[j, i], np.array([0, 0, 0])) and np.array_equal(warpedImage[j, i],
                                                                                                np.array([0, 0, 0]))):
                        # print("BLACK")
                        # instead of just putting it with black,
                        # take average of all nearby values and avg it.
                        warpedImage[j, i] = [0, 0, 0]
                    else:
                        if np.array_equal(warpedImage[j, i], [0, 0, 0]):
                            # print( "PIXEL")
                            warpedImage[j, i] = leftImage[j, i]
                        else:
                            if not np.array_equal(leftImage[j, i], [0, 0, 0]):
                                bw, gw, rw = warpedImage[j, i]
                                bl, gl, rl = leftImage[j, i]
                                # b = (bl+bw)/2
                                # g = (gl+gw)/2
                                # r = (rl+rw)/2
                                warpedImage[j, i] = [bl, gl, rl]
                except:
                    pass
        return warpedImage

    def showImage(self, string=None):
        if string == 'left':
            cv2.imshow("left image", self.left_image)
        elif string == "right":
            cv2.imshow("right Image", self.right_image)
        # cv2.waitKey(0)


if __name__ == '__main__':
    start_time = time.time()
    s = Stitch("frame\\test")
    s.leftshift()
    s.showImage('left')
    s.rightshift()
    s.showImage('right')
    cv2.imwrite("test12.jpg", s.right_image)
    print("image written")
    print(f"time = {time.time() - start_time} s")
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

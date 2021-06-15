import os
import cv2
from util import resize

video_src_path = './video/test'
frame_des_path = './frame/'


def video2frame(video_src_path, frame_des_path, interval=1):
    videos = os.listdir(video_src_path)

    for video in videos:
        video_name = video[:-4]  # video format including mov/mkv
        os.mkdir(frame_des_path + video_name)

        frame_save_path = frame_des_path + video_name + '/'
        video_cur_path = os.path.join(video_src_path, video)
        capture = cv2.VideoCapture(video_cur_path)

        success = False
        count = 0
        if capture.isOpened():
            success, frame = capture.read()
            print("Start decoding file %s..." % video)
            count += 1
        else:
            print("Open %s failure!" % video)

        while success:
            if count % interval == 0 or count == 1:
                print("Writing the number %d of frame to src file" % (count // interval))
                frame = resize(frame, 500, 600)
                print(frame.shape)
                cv2.imwrite(frame_save_path + '%d.jpg' % (count // interval), frame)
            success, frame = capture.read()
            count += 1
        capture.release()
        print("Encoding file %s success!" % video)


if __name__ == '__main__':
    video2frame(video_src_path, frame_des_path, interval=100)

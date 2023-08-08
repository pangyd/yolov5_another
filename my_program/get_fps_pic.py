import cv2
import os
import shutil

video_path = "img/world_cup.mp4"
save_folder = "video_to_pic"
per_pic_num = 10


def video_to_pic(video_path, save_folder, index):
    # 读取视频
    video = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        _, frame = video.read()
        if frame is None:
            break
        if frame_count % per_pic_num == 0:
            save_path = "{}/{:>03d}.jpg".format(save_folder, index)
            cv2.imwrite(save_path, frame)
            index += 1
        frame_count += 1

    print("当前视频一共有{}帧".format(frame_count))
    print("一共保存了{}张图片".format(index-1))

    # 计算FPS
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")
    if int(major_ver) < 3:
        fps = video.get(cv2.CV_CAP_PROP_FPS)
        print("帧每秒使用的fps:{0}".format(fps))
    else:
        fps = video.get(cv2.cv.CAP_PROP_FPS)
        print("帧每秒使用的fps:{0}".format(fps))

    video.release()


def pic_information(pic_path):
    pic = cv2.imread(pic_path)
    height = pic.shape[0]
    width = pic.shape[1]
    channel = pic.shape[2]
    print(height, width, channel)


if __name__ == "__main__":
    # if os.path.exists(save_folder):
    #     shutil.rmtree(save_folder)
    # os.mkdir(save_folder)
    #
    # video_to_pic(video_path, save_folder, 1)

    # pic_information("video_to_pic/001.jpg")
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    print(fps)



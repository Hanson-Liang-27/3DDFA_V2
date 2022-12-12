# coding: utf-8

__author__ = 'cleardusk'

import numpy as np
import cv2
from math import sqrt
import matplotlib.pyplot as plt
import os
# import subprocess

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


def get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos:]


def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res


def calc_hypotenuse(pts):
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    return llength / 3


def parse_roi_box_from_landmark(pts):
    """calc roi box from landmark"""
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[3] + bbox[1]) / 2

    roi_box = [0] * 4
    roi_box[0] = center_x - llength / 2
    roi_box[1] = center_y - llength / 2
    roi_box[2] = roi_box[0] + llength
    roi_box[3] = roi_box[1] + llength

    return roi_box


def parse_roi_box_from_bbox(bbox):
    left, top, right, bottom = bbox[:4]
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.58)

    roi_box = [0] * 4
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size

    return roi_box


def plot_image(img):
    height, width = img.shape[:2]
    plt.figure(figsize=(12, height / width * 12))

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    plt.imshow(img[..., ::-1])
    plt.show()


def draw_landmarks(img, pts, style='fancy', wfp=None, show_flag=False, **kwargs):
    """Draw landmarks using matplotlib"""
    height, width = img.shape[:2]
    plt.figure(figsize=(12, height / width * 12))
    plt.imshow(img[..., ::-1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    dense_flag = kwargs.get('dense_flag')

    if not type(pts) in [tuple, list]:
        pts = [pts]
    for i in range(len(pts)):
        if dense_flag:   #38365个点
            # plt.plot(pts[i][0, ::6], pts[i][1, ::6], 'o', markersize=0.4, color='c', alpha=0.7)
            plt.plot(pts[i][0, ::6], pts[i][1, ::6], 'o', markersize=2, color='w', alpha=0.7)
        else:    #68个点
            alpha = 0.8
            markersize = 4
            lw = 1.5
            color = kwargs.get('color', 'w')
            markeredgecolor = kwargs.get('markeredgecolor', 'black')

            nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]

            # close eyes and mouths
            plot_close = lambda i1, i2: plt.plot([pts[i][0, i1], pts[i][0, i2]], [pts[i][1, i1], pts[i][1, i2]],
                                                 color=color, lw=lw, alpha=alpha - 0.1)
            plot_close(41, 36)
            plot_close(47, 42)
            plot_close(59, 48)
            plot_close(67, 60)

            for ind in range(len(nums) - 1):
                l, r = nums[ind], nums[ind + 1]
                plt.plot(pts[i][0, l:r], pts[i][1, l:r], color=color, lw=lw, alpha=alpha - 0.1)

                plt.plot(pts[i][0, l:r], pts[i][1, l:r], marker='o', linestyle='None', markersize=markersize,
                         color=color,
                         markeredgecolor=markeredgecolor, alpha=alpha)
    if wfp is not None:
        plt.savefig(wfp, dpi=150)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plt.show()




def cv_draw_landmark(img_ori, pts, box=None, color=BLUE, size=5):
    img = img_ori.copy()
    n = pts.shape[1]
    if n <= 106:
        for i in range(n):
            cv2.circle(img, (int(round(pts[0, i])), int(round(pts[1, i]))), size, color, -1)
    else:
        sep = 1
        for i in range(0, n, sep):
            cv2.circle(img, (int(round(pts[0, i])), int(round(pts[1, i]))), size, color, 1)

    if box is not None:
        left, top, right, bottom = np.round(box).astype(np.int32)
        left_top = (left, top)
        right_top = (right, top)
        right_bottom = (right, bottom)
        left_bottom = (left, bottom)
        # cv2.line(img, left_top, right_top, BLUE, 1, cv2.LINE_AA)
        # cv2.line(img, right_top, right_bottom, BLUE, 1, cv2.LINE_AA)
        # cv2.line(img, right_bottom, left_bottom, BLUE, 1, cv2.LINE_AA)
        # cv2.line(img, left_bottom, left_top, BLUE, 1, cv2.LINE_AA)
        cv2.line(img, left_top, right_top, BLUE, 5, cv2.LINE_AA)
        cv2.line(img, right_top, right_bottom, BLUE, 5, cv2.LINE_AA)
        cv2.line(img, right_bottom, left_bottom, BLUE, 5, cv2.LINE_AA)
        cv2.line(img, left_bottom, left_top, BLUE, 5, cv2.LINE_AA)

    return img

# def rotate90_to_mp4(inpath,outpath):
    # !cp /content/drive/MyDrive/019-white-17/four_video/face121.mov /content/3DDFA_V2/TestSamples
    # 视频逆时针旋转90度
    # !ffmpeg -i /content/drive/MyDrive/019-white-17/four_video/face121.mov -vf "transpose=3" /content/3DDFA_V2/TestSamples/face019_3.mp4
    # !ffmpeg -i $inpath -vf "transpose=3" $outpath
    # !test -f $outpath && echo "rotate success" || echo "rotate fail"

def video2sequence(video_path, sample_step=10):
    videofolder = os.path.splitext(video_path)[0]
    os.makedirs(videofolder, exist_ok=True)
    video_name = os.path.splitext(os.path.split(video_path)[-1])[0]
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        # if count%sample_step == 0:
        imagepath = os.path.join(videofolder, f'{video_name}_frame{count:04d}.jpg')
        cv2.imwrite(imagepath, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list

def plt_face_box(img_fp,boxes):
    import matplotlib.pyplot as plt
    img = plt.imread(img_fp)
    plt.imshow(img)
    plt.plot([boxes[0][0], boxes[0][2]],[boxes[0][3], boxes[0][3]], color='r')
    plt.plot([boxes[0][2], boxes[0][2]],[boxes[0][3], boxes[0][1]], color='r')
    plt.plot([boxes[0][0], boxes[0][2]],[boxes[0][1], boxes[0][1]], color='r')
    plt.plot([boxes[0][0], boxes[0][0]],[boxes[0][3], boxes[0][1]],  color='r')
    plt.show() 

def fit_single_wrinkle(img_fp,boxes,polynum,out_path):
    # 检测皱纹红点label
    # Open the image
    im = cv2.imread(img_fp)
    red_coords = []
    # Loop through all pixels in the image
    for x in range(len(im[1])):
        for y in range(len(im[0])):
            if im[y, x][2] >= 230 and im[y,x][1] <=10:
                red_coord = []
                red_coord.append(x - boxes[0][0])
                red_coord.append(y - boxes[0][1])
                red_coords.append(red_coord)
    print('red_coords locations:',red_coords)
    print('number of red_coords:',len(red_coords))

    # 读取图片并转换为数组
    img_array = np.array(img)
    # 提取图片中的曲线点
    curve_points = red_coords
    x = [point[0] for point in curve_points]
    y = [point[1] for point in curve_points]
    # 拟合曲线
    fit = np.polyfit(x, y, polynum)
    # 绘制拟合后的曲线
    curve = np.poly1d(fit)

    for i in range(int(min(x)),int(max(x))):
        y = curve(i)
        img_array[int(y + boxes[0][1]), int(i + boxes[0][0])] = (255, 0, 0)

    # 保存图片并输出结果
    cv2.imwrite(out_path, img_array)
    return fit
import sys

from PIL import Image
from PIL import ImageSequence
import numpy as np
from tqdm import tqdm
import cv2
# import imageio
# import gif2numpy
# from scipy import ndimage
# import array2gif
# import skvideo.io
# import matplotlib.pyplot as plt

INF = np.iinfo(np.int64).max
video_path = sys.argv[1]
PATCH_SIZE = 5


def lucas_kanade_optic_flow(vid):
    vid_gray = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in vid])
    shape = vid_gray.shape
    It, Ix, Iy = np.gradient(vid_gray)
    velocity = np.zeros(list(shape) + [2])
    for t in tqdm(range(shape[0])):
        for y in range(0, shape[1], PATCH_SIZE):
            for x in range(0, shape[2], PATCH_SIZE):
                v = calc_patch_velocity(
                    Ix[t, y:y + PATCH_SIZE, x:x + PATCH_SIZE],
                    Iy[t, y:y + PATCH_SIZE, x:x + PATCH_SIZE],
                    It[t, y:y + PATCH_SIZE, x:x + PATCH_SIZE])
                velocity[t, y:y + PATCH_SIZE, x:x + PATCH_SIZE, :] = v[None, None, None, :]
    return velocity


def draw_arrows(vid, velocity):
    shape = vid.shape
    for t in range(shape[0]):
        for y in range(0, shape[1], PATCH_SIZE):
            for x in range(0, shape[2], PATCH_SIZE):
                draw_arrow(vid[t, :, :, :], x + PATCH_SIZE // 2, y + PATCH_SIZE // 2,
                           *velocity[t, y + PATCH_SIZE // 2, x + PATCH_SIZE // 2])


def draw_arrow(img, x, y, vx, vy):
    ARROW_COLOR = (0, 0, 255)
    ARROW_WIDTH = 1
    cv2.arrowedLine(img, (x, y), (x + int(vx), y + int(vy)), ARROW_COLOR, ARROW_WIDTH)


def calc_patch_velocity(Ix_patch, Iy_patch, It_patch):
    A = np.vstack([Ix_patch.flatten(), Iy_patch.flatten()]).T
    b = -It_patch.flatten()
    return least_square(A, b)


def least_square(A, b):
    try:
        A = np.matrix(A)
        b = np.matrix(b).T
        return (np.linalg.inv(A.T * A) * (A.T * b)).squeeze()
    except np.linalg.LinAlgError:
        return np.array([0, 0])


def loop_show(vid):
    FPS = 60
    QUIT_KEY = 'q'
    sleep = 1000 // FPS
    cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
    while True:
        for frame in vid:
            cv2.imshow("video", frame)
            key = cv2.waitKey(sleep)
            if key == ord(QUIT_KEY):
                cv2.destroyAllWindows()
                return
            if key == ord('a'):
                sleep *= 2
            if key == ord('s'):
                sleep *= 1 / 2


def load_video(video_path):
    # return gif2numpy.convert(video_path, BGR2RGB=False)[0]
    # return plt.imread(video_path)
    # return ndimage.imread(video_path)
    vid_obj = Image.open(video_path)
    frames = np.array([np.array(frame.copy().convert('RGB').getdata(), dtype=np.uint8).reshape(
        frame.size[1], frame.size[0], 3) for frame in ImageSequence.Iterator(vid_obj)])
    return frames


vid = load_video(video_path)
velocity = lucas_kanade_optic_flow(vid)
draw_arrows(vid, velocity)
loop_show(vid)

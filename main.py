import images2gif
import os
import sys
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2


PATCH_SIZE = 20


def lucas_kanade_optic_flow(vid):
    vid_gray = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in vid]) / 255.0
    shape = vid_gray.shape
    It, Iy, Ix = np.gradient(vid_gray)
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
    ARROW_COLOR = (0, 0, 255)
    ARROW_WIDTH = 1
    shape = vid.shape
    for t in range(shape[0]):
        for y in range(0, shape[1], PATCH_SIZE):
            for x in range(0, shape[2], PATCH_SIZE):
                v = velocity[t, y, x]
                v *= 10
                pos = np.array([x + PATCH_SIZE // 2, y + PATCH_SIZE // 2])
                dst = pos + v.astype(np.int)
                cv2.arrowedLine(vid[t, :, :, :], tuple(pos), tuple(dst), ARROW_COLOR, ARROW_WIDTH)


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
                sleep = sleep // 2


def load_video(video_path):
    return np.array(images2gif.readGif(video_path, True))


def save_video(vid, out_path):
    ims = [Image.fromarray(frame) for frame in vid]
    ims[0].save(out_path, save_all=True, append_images=ims[1:], loop=0, optimize=True)
    # images2gif.writeGif(out_path, [frame for frame in vid], subRectangles=False)


def main():
    video_path = sys.argv[1]
    vid = load_video(video_path)
    velocity = lucas_kanade_optic_flow(vid)
    draw_arrows(vid, velocity)
    base, ext = os.path.splitext(video_path)
    out_path = base + '_out' + ext
    save_video(vid, out_path)
    loop_show(vid)


if __name__ == '__main__':
    main()

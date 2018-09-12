import numpy as np

import cv2

from .utils import convert_to_pixels

try:
    from IPython import get_ipython

    if get_ipython() is not None:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm
except ImportError:
    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)


def make_video(video_filename, centroids, head_position_mean,
               head_orientation_mean, output_video_filename='output.avi',
               cm_to_pixels=1.0, disable_progressbar=False):
    RGB_PINK = (234, 82, 111)
    RGB_YELLOW = (253, 231, 76)
    RGB_WHITE = (255, 255, 255)

    video = cv2.VideoCapture(video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_size = (int(video.get(3)), int(video.get(4)))
    frame_rate = video.get(5)
    n_frames = int(head_orientation_mean.shape[0])

    out = cv2.VideoWriter(output_video_filename, fourcc, frame_rate,
                          frame_size, True)

    centroids = {color: convert_to_pixels(data, frame_size, cm_to_pixels)
                 for color, data in centroids.items()}

    for time_ind in tqdm(range(n_frames - 1), desc='making video',
                         disable=disable_progressbar):
        is_grabbed, frame = video.read()
        if is_grabbed:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            red_centroid = centroids['red'][time_ind]
            green_centroid = centroids['green'][time_ind]

            head_position = head_position_mean[time_ind, [0, 3]]
            head_position = convert_to_pixels(
                head_position, frame_size, cm_to_pixels)
            head_position = tuple(head_position.astype(int))

            head_orientation_red = head_orientation_mean[time_ind, [0, 3]]
            head_orientation_red = convert_to_pixels(
                head_orientation_red, frame_size, cm_to_pixels)
            head_orientation_red = tuple(head_orientation_red.astype(int))

            head_orientation_green = head_orientation_mean[time_ind, [6, 9]]
            head_orientation_green = convert_to_pixels(
                head_orientation_green, frame_size, cm_to_pixels)
            head_orientation_green = tuple(head_orientation_green.astype(int))

            if np.all(~np.isnan(red_centroid)):
                cv2.circle(frame, tuple(red_centroid.astype(int)), 8,
                           RGB_YELLOW, -1, cv2.CV_8U)

            if np.all(~np.isnan(green_centroid)):
                cv2.circle(frame, tuple(green_centroid.astype(int)), 8,
                           RGB_PINK, -1, cv2.CV_8U)

            cv2.arrowedLine(frame, head_orientation_red,
                            head_orientation_green, RGB_WHITE, 4, 8, 0, 0.25)
            cv2.circle(frame, head_position, 8,
                       RGB_WHITE, -1, cv2.CV_8U)

            # cv2.ellipse(frame, tuple(kalman.astype(int)), )
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        else:
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()

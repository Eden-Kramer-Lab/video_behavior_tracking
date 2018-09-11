import numpy as np

import cv2

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

_RED = {'min_color': np.array([2, 150, 168], dtype=np.uint8),
        'max_color': np.array([28, 255, 255], dtype=np.uint8)}

_GREEN = {'min_color': np.array([55, 81, 61], dtype=np.uint8),
          'max_color': np.array([87, 255, 255], dtype=np.uint8)}

_BLUE = {'min_color': np.array([82, 89, 153], dtype=np.uint8),
         'max_color': np.array([119, 255, 255], dtype=np.uint8)}

_BODY = {'min_color': np.array([0, 96, 25], dtype=np.uint8),
         'max_color': np.array([35, 153, 127], dtype=np.uint8),
         'blur_kernel': (15, 15),
         'morph_kernel': (11, 11)}

_COLORS = {'red': _RED, 'green': _GREEN}


def convert_hsv(color):
    '''Convert from Image processing HSV to Open CV

    Image Processing Scale
    H: 0-360, S: 0-100 and V: 0-100

    OpenCV Scale
    H: 0-180, S: 0-255, V: 0-255

    '''
    return (color / np.array([2, 100 / 255, 100 / 255])).astype(np.uint8)


def find_color_centroid(frame, min_color=(0, 0, 0), max_color=(180, 255, 255),
                        blur_kernel=(15, 15), morph_kernel=(5, 5)):
    '''Given a color range, finds the center point of the color in the frame.
    '''
    frame_ = frame.copy()
    frame_ = cv2.GaussianBlur(frame_, blur_kernel, 0)
    frame_ = cv2.cvtColor(frame_, cv2.COLOR_RGB2HSV)
    frame_ = cv2.inRange(frame_, min_color, max_color)
    KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)
    frame_ = cv2.morphologyEx(frame_, cv2.MORPH_CLOSE, KERNEL)
    frame_ = cv2.morphologyEx(frame_, cv2.MORPH_OPEN, KERNEL)

    moments = cv2.moments(frame_)
    try:
        return (
            int(moments['m10'] / moments['m00']),
            int(moments['m01'] / moments['m00'])
        )
    except ZeroDivisionError:
        return (np.nan, np.nan)


def detect_LEDs(video_filename, colors=_COLORS):
    video = cv2.VideoCapture(video_filename)
    frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if n_frames > 0:
        centroids = {color: np.full((n_frames, 2), np.nan) for color in colors}

        for frame_ind in tqdm(np.arange(n_frames - 1), desc='frames'):
            is_grabbed, frame = video.read()
            if is_grabbed:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                for color, kwargs in colors.items():
                    centroids[color][frame_ind] = find_color_centroid(
                        frame, **kwargs)
    else:
        centroids = {color: [] for color in colors}
        n_frames = 0
        pbar = tqdm(total=40000, desc='frames',
                    bar_format='{n_fmt} [{elapsed}<{remaining}]')
        while True:
            is_grabbed, frame = video.read()
            if is_grabbed:
                pbar.update(1)
                n_frames += 1
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for color, kwargs in colors.items():
                    centroids[color].append(find_color_centroid(
                        frame, **kwargs))
            else:
                break
        centroids = {color: np.array(data)
                     for color, data in centroids.items()}

    video.release()
    cv2.destroyAllWindows()

    return centroids, frame_rate, frame_size, n_frames

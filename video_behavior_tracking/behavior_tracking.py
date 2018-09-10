import glob
import json
import os.path
from argparse import ArgumentParser
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.linalg import block_diag

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

STATE_NAMES = ['x_position', 'x_velocity', 'x_acceleration',
               'y_position', 'y_velocity', 'y_acceleration']

Position = namedtuple(
    'Position', ['head_position_mean', 'head_position_covariance',
                 'head_orientation_mean', 'head_orientation_covariance',
                 'centroids', 'frame_rate', 'frame_size', 'n_frames'])


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
    frame_size = (int(video.get(3)), int(video.get(4)))
    frame_rate = video.get(5)
    # n_frames = int(video.get(7))
    n_frames = 100

    centroids = {color: np.full((n_frames, 2), np.nan) for color in colors}

    for frame_ind in tqdm(np.arange(n_frames - 1), desc='centroids'):
        is_grabbed, frame = video.read()
        if is_grabbed:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            for color, kwargs in colors.items():
                centroids[color][frame_ind] = find_color_centroid(
                    frame, **kwargs)

    video.release()
    cv2.destroyAllWindows()

    return centroids, frame_rate, frame_size, n_frames


def kalman_filter(data, state_transition, state_to_observed,
                  state_covariance, measurement_covariance,
                  prior_state, prior_covariance, inverse=np.linalg.inv):
    '''Handles missing observations

    Code modified from https://github.com/rlabbe/filterpy

    Parameters
    ----------
    data : ndarray, shape (n_time, n_observables)
        Observations from sensors
    state_transition : ndarray, shape (n_states, n_states)
        State transition matrix, F
    state_to_observed : ndarray, shape (n_observables, n_states)
        Measurement function/Observation Model, H
    state_covariance : ndarray, shape (n_states, n_states)
        Process covariance, Q
    measurement_covariance : ndarray, shape (n_observables, n_observables)
        Observation covariance, R
    prior_state : ndarray, shape (n_states,)
        Initial state mean
    prior_covariance : ndarray, shape (n_states, n_states)
        Initial state covariance (belief in state)
    inverse : function, optional

    Returns
    -------
    posterior_mean : ndarray (n_time, n_states)
    posterior_covariance : ndarray (n_time, n_states, n_states)

    '''
    n_time, n_states = data.shape[0], state_transition.shape[0]
    posterior_mean = np.zeros((n_time, n_states))
    posterior_covariance = np.zeros((n_time, n_states, n_states))

    posterior_mean[0] = prior_state.copy()
    posterior_covariance[0] = prior_covariance.copy()

    identity = np.eye(n_states)

    for time_ind in tqdm(np.arange(1, n_time), desc='kalman filter'):
        # Predict
        prior_mean = state_transition @ posterior_mean[time_ind - 1]
        prior_covariance = (
            state_transition @ posterior_covariance[time_ind -
                                                    1] @ state_transition.T
            + state_covariance)

        # Update
        system_uncertainty = (
            state_to_observed @ prior_covariance @ state_to_observed.T
            + measurement_covariance)

        # kalman gain (n_states, n_observables)
        # prediction uncertainty vs. measurement uncertainty
        kalman_gain = prior_covariance @ state_to_observed.T @ inverse(
            system_uncertainty)
        prediction_error = data[time_ind] - \
            state_to_observed @ prior_mean  # innovation

        # Handle missing data by not updating the estimate and covariance
        is_missing = np.isnan(data[time_ind])
        prediction_error[is_missing] = 0.0
        kalman_gain[:, is_missing] = 0.0

        # Update mean
        posterior_mean[time_ind] = prior_mean + kalman_gain @ prediction_error

        # Update covariance
        I_KH = identity - kalman_gain @ state_to_observed
        posterior_covariance[time_ind] = (
            I_KH @ prior_covariance @ I_KH.T +
            kalman_gain @ measurement_covariance @ kalman_gain.T)

    return posterior_mean, posterior_covariance


def rts_smoother(posterior_mean, posterior_covariance, state_transition,
                 state_covariance, inverse=np.linalg.inv):
    '''Runs the Rauch-Tung-Striebal Kalman smoother on a set of
    means and covariances computed by a Kalman filter.

    Code modified from https://github.com/rlabbe/filterpy.

    Parameters
    ----------
    posterior_mean : ndarray, shape (n_time, n_states)
    posterior_covariance : ndarray, shape (n_time, n_states, n_states)
    state_transition : ndarray, shape (n_states, n_states)
    state_covariance : ndarray, shape (n_states, n_states)
    inverse : function, optional

    Returns
    -------
    smoothed_mean : ndarray, shape (n_time, n_states)
    smoothed_covariances : ndarray, shape (n_time, n_states, n_states)

    '''
    n_time, n_states = posterior_mean.shape
    smoothed_mean = posterior_mean.copy()
    smoothed_covariances = posterior_covariance.copy()

    for time_ind in tqdm(np.arange(n_time - 2, -1, -1), desc='smoothing'):
        prior_covariance = (state_transition @ posterior_covariance[time_ind] @
                            state_transition.T + state_covariance)
        smoother_gain = (posterior_covariance[time_ind] @ state_transition.T @
                         inverse(prior_covariance))
        smoothed_mean[time_ind] += smoother_gain @ (
            smoothed_mean[time_ind + 1] - state_transition @
            smoothed_mean[time_ind])
        smoothed_covariances[time_ind] += (smoother_gain @ (
            smoothed_covariances[time_ind + 1] - prior_covariance) @
            smoother_gain.T)

    return smoothed_mean, smoothed_covariances


def make_head_position_model(centroids, frame_rate, measurement_variance=1E-4,
                             process_variance=5):
    data = np.concatenate((centroids['red'], centroids['green']), axis=1)
    dt = 1 / frame_rate

    q = np.array([[0.25 * dt**4, 0.5 * dt**3, 0.5 * dt**2],
                  [0.5 * dt**3,        dt**2,          dt],
                  [0.5 * dt**2,           dt,         1.0]])
    state_covariance = block_diag(q, q) * process_variance

    f = np.array([[1.0,  dt, 0.5 * dt**2],
                  [0.0, 1.0,           0],
                  [0.0, 0.0,         1.0]])
    state_transition = block_diag(f, f)

    h = np.array([1, 0, 0])
    state_to_observed = np.concatenate((block_diag(h, h), block_diag(h, h)))

    # Observation covariance
    measurement_covariance = np.eye(4) * measurement_variance

    initial_x = np.nanmean(data[:, [0, 2]], axis=1)
    initial_x = initial_x[np.nonzero(~np.isnan(initial_x))[0][0]]

    initial_y = np.nanmean(data[:, [1, 3]], axis=1)
    initial_y = initial_y[np.nonzero(~np.isnan(initial_y))[0][0]]

    prior_state = np.array([initial_x, 0, 0, initial_y, 0, 0])
    prior_covariance = np.diag([1, 250, 6000, 1, 250, 6000])

    return {'data': data,
            'state_transition': state_transition,
            'state_to_observed': state_to_observed,
            'state_covariance': state_covariance,
            'measurement_covariance': measurement_covariance,
            'prior_state': prior_state,
            'prior_covariance': prior_covariance}


def make_head_orientation_model(centroids, frame_rate,
                                measurement_variance=1E-4, process_variance=5):
    data = np.concatenate((centroids['red'], centroids['green']), axis=1)
    dt = 1 / frame_rate

    q = np.array([[0.25 * dt**4, 0.5 * dt**3, 0.5 * dt**2],
                  [0.5 * dt**3,        dt**2,          dt],
                  [0.5 * dt**2,           dt,         1.0]])
    state_covariance = block_diag(q, q, q, q) * process_variance

    f = np.array([[1.0,  dt, 0.5 * dt**2],
                  [0.0, 1.0,           0],
                  [0.0, 0.0,         1.0]])
    state_transition = block_diag(f, f, f, f)

    h = np.array([1, 0, 0])
    state_to_observed = block_diag(h, h, h, h)

    # Observation covariance
    measurement_covariance = np.eye(4) * measurement_variance

    x1 = data[~np.isnan(data[:, 0]), 0][0]
    y1 = data[~np.isnan(data[:, 1]), 1][0]
    x2 = data[~np.isnan(data[:, 2]), 2][0]
    y2 = data[~np.isnan(data[:, 3]), 3][0]
    prior_state = np.array([x1, 0, 0, y1, 0, 0,
                            x2, 0, 0, y2, 0, 0])
    prior_covariance = np.diag([1, 250, 6000, 1, 250, 6000,
                                1, 250, 6000, 1, 250, 6000])

    return {'data': data,
            'state_transition': state_transition,
            'state_to_observed': state_to_observed,
            'state_covariance': state_covariance,
            'measurement_covariance': measurement_covariance,
            'prior_state': prior_state,
            'prior_covariance': prior_covariance}


def flip_y(data, frame_size):
    '''Flips the y-axis'''
    new_data = data.copy()
    if data.ndim > 1:
        new_data[:, 1] = frame_size[1] - new_data[:, 1]
    else:
        new_data[1] = frame_size[1] - new_data[1]
    return new_data


def convert_to_cm(data, frame_size, cm_to_pixels=1.0):
    '''

    Parameters
    ----------
    data : ndarray, shape (n_time, 2)
    frame_size : array_like, shape (2,)
    cm_to_pixels : float

    Returns
    -------
    data : ndarray, shape (n_time, 2)

    '''
    return flip_y(data, frame_size) * cm_to_pixels


def convert_to_pixels(data, frame_size, cm_to_pixels=1.0):
    return flip_y(data / cm_to_pixels, frame_size)


def make_video(video_filename, centroids, head_position_mean,
               head_orientation_mean, output_video_filename='output.avi',
               cm_to_pixels=1.0):
    RGB_PINK = (234, 82, 111)
    RGB_YELLOW = (253, 231, 76)
    RGB_WHITE = (255, 255, 255)

    video = cv2.VideoCapture(video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    frame_size = (int(video.get(3)), int(video.get(4)))
    frame_rate = video.get(5)
    n_frames = int(head_orientation_mean.shape[0])

    out = cv2.VideoWriter(output_video_filename, fourcc, frame_rate,
                          frame_size, True)

    centroids = {color: convert_to_pixels(data, frame_size, cm_to_pixels)
                 for color, data in centroids.items()}

    for time_ind in tqdm(range(n_frames - 1), desc='making video'):
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


def filter_smooth_data(model):
    posterior_mean, posterior_covariance = kalman_filter(**model)
    posterior_mean, posterior_covariance = rts_smoother(
        posterior_mean, posterior_covariance, model['state_transition'],
        model['state_covariance'])

    return posterior_mean, posterior_covariance


def extract_position_data(video_filename, cm_to_pixels=1.0, colors=_COLORS):
    centroids, frame_rate, frame_size, n_frames = detect_LEDs(
        video_filename, colors=colors)
    centroids = {color: convert_to_cm(data, frame_size, cm_to_pixels)
                 for color, data in centroids.items()}

    head_position_model = make_head_position_model(centroids, frame_rate)
    head_position_mean, head_position_covariance = filter_smooth_data(
        head_position_model)

    head_orientation_model = make_head_orientation_model(centroids, frame_rate)
    head_orientation_mean, head_orientation_covariance = filter_smooth_data(
        head_orientation_model)

    return Position(head_position_mean, head_position_covariance,
                    head_orientation_mean, head_orientation_covariance,
                    centroids, frame_rate, frame_size, n_frames)


def position_dataframe(position):
    head_direction = np.arctan2(
        np.diff(position.head_orientation_mean[:, [3, 9]], axis=1),
        np.diff(position.head_orientation_mean[:, [0, 6]], axis=1))

    position_info = pd.DataFrame(position.head_position_mean,
                                 columns=STATE_NAMES)
    position_info['head_direction'] = head_direction
    position_info['speed'] = np.sqrt(
        position_info.x_velocity ** 2 + position_info.y_velocity ** 2)
    position_info['acceleration'] = np.sqrt(
        position_info.x_acceleration ** 2 + position_info.y_acceleration ** 2)

    return position_info


def convert_to_loren_frank_data_format(position_info, cm_to_pixels=1.0):
    LOREN_FRANK_NAMES = {
        'time': 'time',
        'x_position': 'x',
        'y_position': 'y',
        'head_direction': 'dir',
        'speed': 'vel'
    }

    data = (position_info
            .reset_index()
            .rename(columns=LOREN_FRANK_NAMES)
            .loc[:, LOREN_FRANK_NAMES.values()])
    fields = ' '.join(data.columns)

    return {
        'arg': [],
        'descript': [],
        'fields': fields,
        'data': data.values,
        'cmperpixel': cm_to_pixels,
    }


def video_filename_to_epoch_key(video_filename, date_to_day):
    date, animal, epoch = (video_filename.split('/')[-1]
                           .split('.')[0].split('_'))
    epoch = int(epoch)
    day = date_to_day[date]
    return animal, day, epoch


def save_loren_frank_data(epoch_key, file_type, save_data, n_epochs=None,
                          save_path=None):
    '''Saves data in the Loren Frank file format.

    Parameters
    ----------
    epoch_key : tuple
        (animal, day epoch)
    file_type : str
        The type of data being saved e.g. 'pos'.
    save_data : object
        Data to save.
    n_epochs : int or None, optional
        Total number of epochs per recording day.
    path : str or None, optional

    '''
    animal, day, epoch = epoch_key
    filename = f'{animal}{file_type}{day:02d}.mat'
    if save_path is not None:
        filename = os.path.join(save_path, filename)
    if n_epochs is None:
        n_epochs = epoch

    try:
        file_data = loadmat(filename)
        file_data = file_data[file_type]
    except FileNotFoundError:
        file_data = np.zeros((1, day), dtype=np.object)
        file_data[0, day - 1] = np.zeros((1, n_epochs), dtype=np.object)

    try:
        file_data[0, day - 1][0, epoch - 1] = save_data
    except IndexError:
        old_data = file_data[0, day - 1].copy()
        file_data[0, day - 1] = np.zeros((1, n_epochs), dtype=np.object)
        file_data[0, day - 1][0, :old_data.shape[1]] = old_data
        file_data[0, day - 1][0, epoch - 1] = save_data

    savemat(filename, {file_type: file_data})


def write_config():
    pass


def get_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument('video_filename', type=str, help='Path to file')
    parser.add_argument('config_file', type=str, help='Path to file')
    parser.add_argument('--save_path', type=str,
                        help='Path to save file directory')
    parser.add_argument('--save_video', action='store_true',
                        help='Save video containing extracted position')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_command_line_arguments()
    with open(args.config_file) as data_file:
        config = json.load(data_file)

    for video_filename in glob.glob(args.video_filename):
        print(f'Processing {video_filename}')
        position = extract_position_data(
            video_filename, config['cm_to_pixels'])
        position_info = position_dataframe(position)
        save_data = convert_to_loren_frank_data_format(
            position_info, config['cm_to_pixels'])
        epoch_key = video_filename_to_epoch_key(
            video_filename, config['date_to_day'])
        save_loren_frank_data(epoch_key, 'pos', save_data,
                              save_path=args.save_path)

        if args.save_video:
            animal, day, epoch = epoch_key
            output_video_filename = f'{animal}_{day:02}_{epoch:02}_pos.avi'
            make_video(video_filename, position.centroids,
                       position.head_position_mean,
                       position.head_orientation_mean,
                       output_video_filename=output_video_filename,
                       cm_to_pixels=config['cm_to_pixels'])

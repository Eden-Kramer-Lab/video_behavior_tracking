import json
import os.path

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

STATE_NAMES = ['x_position', 'x_velocity', 'x_acceleration',
               'y_position', 'y_velocity', 'y_acceleration']


def position_dataframe(position, start_time=0.0):
    '''Takes kalman filter data, computes head direction/speed/acceleration and
    puts it in a pandas DataFrame.

    Parameters
    ----------
    position : namedtuple
    start_time : float

    Returns
    -------
    position_info : DataFrame

    '''
    head_direction = np.arctan2(
        np.diff(position.head_orientation_mean[:, [3, 9]], axis=1),
        np.diff(position.head_orientation_mean[:, [0, 6]], axis=1))

    n_time = position.head_position_mean.shape[0]
    time = pd.Index(start_time + (np.arange(n_time) / position.frame_rate),
                    name='time')
    position_info = pd.DataFrame(position.head_position_mean,
                                 columns=STATE_NAMES, index=time)
    position_info['head_direction'] = head_direction
    position_info['speed'] = np.sqrt(
        position_info.x_velocity ** 2 + position_info.y_velocity ** 2)
    position_info['acceleration'] = np.sqrt(
        position_info.x_acceleration ** 2 + position_info.y_acceleration ** 2)

    return position_info


def convert_to_loren_frank_data_format(position_info, cm_to_pixels=1.0):
    '''Converts the pandas DataFrame from the function `position_dataframe` to
    the Loren Frank data format.

    Parameters
    ----------
    position_info : DataFrame
    cm_to_pixels : float

    Returns
    -------
    formatted_data : dict

    '''
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
    date, animal, epoch = (os.path.basename(video_filename)
                           .split('.')[0].split('_')[:3])
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


def flip_y(data, frame_size):
    '''Flips the y-axis

    Parameters
    ----------
    data : ndarray, shape (n_time, 2)
    frame_size : array_like, shape (2,)

    Returns
    -------
    flipped_data : ndarray, shape (n_time, 2)

    '''
    new_data = data.copy()
    if data.ndim > 1:
        new_data[:, 1] = frame_size[1] - new_data[:, 1]
    else:
        new_data[1] = frame_size[1] - new_data[1]
    return new_data


def convert_to_cm(data, frame_size, cm_to_pixels=1.0):
    '''Converts from pixels to cm and flips the y-axis.

    Parameters
    ----------
    data : ndarray, shape (n_time, 2)
    frame_size : array_like, shape (2,)
    cm_to_pixels : float

    Returns
    -------
    converted_data : ndarray, shape (n_time, 2)

    '''
    return flip_y(data, frame_size) * cm_to_pixels


def convert_to_pixels(data, frame_size, cm_to_pixels=1.0):
    '''Converts from cm to pixels and flips the y-axis.

    Parameters
    ----------
    data : ndarray, shape (n_time, 2)
    frame_size : array_like, shape (2,)
    cm_to_pixels : float

    Returns
    -------
    converted_data : ndarray, shape (n_time, 2)

    '''
    return flip_y(data / cm_to_pixels, frame_size)


def write_config(date_to_day, cm_to_pixels):
    '''Writes a json config file.

    Parameters
    ----------
    date_to_day : dict
        Coverts from recording date to day
    cm_to_pixels : float

    '''
    data = {'date_to_day': date_to_day,
            'cm_to_pixels':  cm_to_pixels}
    with open('position_config.json', 'w') as outfile:
        json.dump(data, outfile, sort_keys=True, indent=4,
                  ensure_ascii=False)


def adjust_time(position_filename):
    '''Make sure each successive epoch starts five minutes after the last epoch.

    Parameters
    ----------
    position_filename : str

    '''
    FIVE_MINUTES = 300
    TIME_IND = 0
    animal, day = (position_filename.split(os.sep)[-1]
                   .strip('.mat').split('pos'))
    day = int(day)
    position_file = loadmat(position_filename)
    position_data = position_file['pos'][0, day - 1]

    for epoch_ind, epoch in enumerate(position_data.squeeze()):
        current_epoch_time = epoch['data'][0, 0][:, TIME_IND].copy()
        current_epoch_time -= current_epoch_time[0]
        previous_epoch_time = position_data[
            0, epoch_ind - 1]['data'][0, 0][:, TIME_IND]

        if epoch_ind != 0:
            new_epoch_time_start = previous_epoch_time[-1] + FIVE_MINUTES
            position_data[0, epoch_ind]['data'][0, 0][:, TIME_IND] = (
                current_epoch_time + new_epoch_time_start)

    savemat(position_filename, position_file)

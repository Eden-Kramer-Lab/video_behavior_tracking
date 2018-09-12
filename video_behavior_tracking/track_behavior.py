import glob
import json
from argparse import ArgumentParser

from video_behavior_tracking import (convert_to_loren_frank_data_format,
                                     detect_LEDs, extract_position_data,
                                     make_video, position_dataframe,
                                     save_loren_frank_data,
                                     video_filename_to_epoch_key)


def main(args=None):
    parser = ArgumentParser()
    parser.add_argument('video_filename', type=str, help='Path to file')
    parser.add_argument('config_file', type=str, help='Path to file')
    parser.add_argument('--save_path', type=str,
                        help='Path to save file directory')
    parser.add_argument('--save_video', action='store_true',
                        help='Save video containing extracted position')
    parser.add_argument('--diable_progress_bar', action='store_true',
                        help='Disables the progress bar')

    args = parser.parse_args(args)

    with open(args.config_file) as data_file:
        config = json.load(data_file)

    for video_filename in glob.glob(args.video_filename):
        print(f'\nProcessing {video_filename}')
        centroids, frame_rate, frame_size, n_frames = detect_LEDs(
            video_filename, disable_progressbar=args.disable_progressbar)
        position = extract_position_data(
            centroids, frame_rate, frame_size, n_frames,
            config['cm_to_pixels'],
            disable_progressbar=args.disable_progressbar)
        position_info = position_dataframe(position, start_time=0.0)
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
                       cm_to_pixels=config['cm_to_pixels'],
                       disable_progressbar=args.disable_progressbar)

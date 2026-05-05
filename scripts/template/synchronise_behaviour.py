"""
The pipeline step

  --------------------      -------------------------------------------
  | ephys sync pulse |      | raw behavioural data, incld. sync pulse |
  --------------------      -------------------------------------------
                    \\      //
                     \\    //
                      \\  //
                       \\//
                        ||
          ---------------------------
          | synced behavioural data |
          ---------------------------

This can be very different for different experiments. You need to wrangle your experimental data
into a Tsd frame, by rewriting the load_and_wrangle functions below.

Once it's in the frame, `sync_pulses` does all the synchronisation work.

This is for a VR session. We expect the input to be in the form:

data_folder/
    global_session_type/
        M{mouse:02d}_D{day:02d}_*_{session_type}/
            Record Node 109/                                        <-- ephys output
                ...
            M{mouse:02d}_D{day:02d}_{session_type}_side_capture.csv <-- bonsai output
            M{mouse:02d}_D{day:02d}_{session_type}_blender.csv      <-- blender output

And the output data will be stored in the form

deriv_folder/
    M{mouse:02d}/
        D{day:02d}/
            {session_type}/
                sub-{mouse_string}_day-{day_string}_ses-{session_type}_synced_blender_data.npz
                sub-{mouse_string}_day-{day_string}_ses-{session_type}_synced_bonsai_data.npz
                sync_plots/
                    (some plots showing if the 
                    synchronization was successful).png

The output files are in npz format, and cane be loaded with pynapple. Read more:
    https://pynapple.org/user_guide/02_input_output.html
                    
Based on code originally developed by Wolf du Wulf, from:
    https://github.com/wulfdewolf/mmnav

"""


from pathlib import Path
from argparse import ArgumentParser


import pandas as pd
import numpy as np

import pynapple as nap
import spikeinterface.full as si

from nolanlab_sync.sync import sync_pulses
from nolanlab_ephys.lab_utils import chronologize_paths, get_recording_folders

def get_args():

    parser = ArgumentParser()

    parser.add_argument("mouse", type=int)
    parser.add_argument("day", type=int)
    parser.add_argument("sessions")
    parser.add_argument("--data_folder", default="/home/nolanlab/Work/Wolf_Project/data/")
    parser.add_argument("--deriv_folder", default="/home/nolanlab/Work/Wolf_Project/derivatives/")

    return parser.parse_args()

def main():

    parsed_args = get_args()

    mouse = parsed_args.mouse
    day = parsed_args.day

    mouse_string = f"{mouse:02d}"
    day_string = f"{day:02d}"

    sessions_string = parsed_args.sessions
    sessions = sessions_string.split(",")

    data_folder = Path(parsed_args.data_folder)
    deriv_folder = Path(parsed_args.deriv_folder)

    recording_paths = chronologize_paths(
        get_recording_folders(data_folder=data_folder, mouse=mouse, day=day, sessions=sessions)
    )
    if len(recording_paths) == 0:
        raise FileNotFoundError(f"Cannot find recordings in `data_folder={data_folder}`")

    for ephys_path, session_type in zip(recording_paths, sessions, strict=True):

        mouseday_deriv_folder = deriv_folder / Path(f"M{mouse_string}/D{day_string}/{session_type}")

        ephys_sync_channel = load_and_wrangle_ephys_sync_channel(ephys_path)

        bonsai_filepath = ephys_path / f'M{mouse_string}_D{day_string}_{session_type}_side_capture.csv'
        if not bonsai_filepath.is_file():
            raise FileNotFoundError(f'Cannot find bonsai output at {bonsai_filepath}')

        bonsai_data = load_and_wrangle_bonsai_data(bonsai_filepath)

        synced_bonsai_data, _ = sync_pulses(
            ephys_sync_channel,
            "ephys",
            bonsai_data,
            "bonsai_data",
            mouseday_deriv_folder,
        )

        synced_bonsai_data.save(mouseday_deriv_folder / f"sub-{mouse_string}_day-{day_string}_ses-{session_type}_synced_bonsai_data.npz")

        # blender data only exists for VR sessions
        if 'OF' in session_type:
            continue

        blender_filepath = ephys_path / f'M{mouse_string}_D{day_string}_{session_type}_blender.csv'
        if not blender_filepath.is_file():
            raise FileNotFoundError(f'Cannot find blender output at {blender_filepath}')
        
        blender_data = load_and_wrangle_blender_data(blender_filepath, session_type)

        synced_blender_data, _ = sync_pulses(
            ephys_sync_channel,
            "ephys",
            blender_data,
            "blender_data",
            mouseday_deriv_folder,
        )

        synced_blender_data.save(mouseday_deriv_folder / f"sub-{mouse_string}_day-{day_string}_ses-{session_type}_synced_blender_data.npz")


def load_and_wrangle_ephys_sync_channel(ephys_filepath):

    ephys_sync_channel = si.read_openephys(ephys_filepath, stream_id='Record Node 109#Neuropix-PXI-103.ProbeASYNC')

    # put sync channel in a Tsd
    # Note: we always use the start of the recording as the reference for everything, t=0
    sync_channel = nap.Tsd(
        t=ephys_sync_channel.get_times() - ephys_sync_channel.get_start_time(),
        d=ephys_sync_channel.get_traces().flatten(),
    )

    return sync_channel

def load_and_wrangle_blender_data(blender_filepath, session_type):

    blender_data_raw = pd.read_csv(
        blender_filepath,
        skiprows=4,
        sep=";",
        names=[
            "Time",
            "Position-X",
            "Speed",
            "Speed/gain_mod",
            "Reward_received",
            "Reward_failed",
            "Lick_detected",
            "Tone_played",
            "Position-Y",
            "Tot trial",
            "gain_mod",
            "rz_start",
            "rz_end",
            "sync_pulse",
        ])

    # Preprocess
    blender_data_raw.dropna()
    blender_data_raw["stimulus"] = np.array(
        blender_data_raw["Position-Y"] / 10, dtype=np.int64
    )
    blender_data_raw["stimulus"] -= 1
    blender_data_raw["stimulus"] = blender_data_raw["stimulus"].replace({6: -1})
    blender_data_raw["trial_number"] = (
        (blender_data_raw["stimulus"] != -1)
        & (blender_data_raw["stimulus"].shift(fill_value=-1) == -1)
    ).cumsum()
    blender_data_raw["stimulus_trial_number"] = (
        blender_data_raw["stimulus"].ne(
            blender_data_raw["stimulus"].shift(
                fill_value=blender_data_raw["stimulus"].iloc[0]
            )
        )
    ).cumsum()

    # Tranform position into cm
    blender_data_raw["Position-X"] *= 10.0

    # Unwrap x-position into length of each image
    if session_type == "MMNAV1":
        image_length = int(blender_data_raw["Position-X"][-10000:].max())
        image_length = 5 * np.floor(image_length / 5)
    else:
        image_length = 40.0
    blender_data_raw["Position-X"] = blender_data_raw.groupby("trial_number")[
        "Position-X"
    ].transform(lambda x: np.unwrap(x, period=image_length))

    # Rename columns
    blender_data = pd.DataFrame(
        index=blender_data_raw["Time"],
        data=blender_data_raw[
            [
                "Position-X",
                "Speed",
                "trial_number",
                "Tot trial",
                "stimulus",
                "sync_pulse",
            ]
        ].to_numpy(),
        columns=[
            "P",
            "S",
            "trial_number",
            "stimulus_trial_number",
            "stimulus",
            "sync_pulse",
        ],
    )

    blender_data.infer_objects()
    final_blender_data = nap.TsdFrame(blender_data)

    return final_blender_data

def load_and_wrangle_bonsai_data(bonasi_filepath):

    # Load bonsai data
    bonsai = pd.read_csv(
        bonasi_filepath,
        header=None,
        usecols=[1, 2],
        sep=",",
    )
    bonsai[1] = pd.to_datetime(bonsai[1])
    bonsai.columns = ["time", "sync_pulse"]
  
    bonsai["time"] = pd.to_datetime(bonsai["time"])
    bonsai["time"] = bonsai["time"] - bonsai["time"][0]
    bonsai.set_index("time", inplace=True)
    bonsai.index = bonsai.index.total_seconds()
    return nap.TsdFrame(bonsai)


if __name__ == '__main__':
    main()

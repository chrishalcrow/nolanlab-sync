import numpy as np
import matplotlib.pyplot as plt
import warnings
import pynapple as nap
from scipy.spatial import cKDTree
from pathlib import Path

def normalise_sync_pulse(sync_pulse):
    if len(sync_pulse.unique()) != 2:
        times = sync_pulse.times()
        sync_pulse = np.diff(sync_pulse, prepend=0)
        sync_pulse[sync_pulse < 0] = 0
        sync_pulse = nap.Tsd(d=sync_pulse, t=times)
    return (sync_pulse - sync_pulse.min()) / np.ptp(sync_pulse)


def sync_pulses(ref_sync_pulse, ref_tag, other, other_tag, deriv_folder):
    """
    Synchronises two inputs.

    The `ref_sync_pulse` should be a Pynapple Tsd, containing just the sync pulses.
    This should be a high accuracy reference frame for everything else.
    Usually the sync channel from a ephys recording.

    `other` should be a TsdFrame, one of who's columns is 'sync_pulse'.
    We will synchronise this to the reference frame.
    Usually a behavioural TsdFrame from blender/bonsai output.

    Returns a new copy of `other` with a new column "synced_time", which contains
    the synchronised timestamps.
    """

    # Ref sync times (from e.g. ephys)
    ref_sync_pulse = normalise_sync_pulse(ref_sync_pulse)
    ref_sync_times = ref_sync_pulse.threshold(
        np.median(ref_sync_pulse.values) + 5 * np.std(ref_sync_pulse.values),
        method="above",
    ).time_support.get_intervals_center()

    # Other sync times (from e.g. behaviour)
    other_sync_pulse = normalise_sync_pulse(other["sync_pulse"])
    other_sync_times = other_sync_pulse.threshold(
        np.median(other_sync_pulse.values) + 5 * np.std(other_sync_pulse.values),
        method="above",
    ).time_support.get_intervals_center()

    # if not enough pulses, we cannot do the syncing
    if len(ref_sync_times) < 10:
        raise ValueError("Fewer than ten sync pulses in `ref_sync_times`. Cannot sync.")
    if len(other_sync_times) < 10:
        raise ValueError("Fewer than ten sync pulses in `other['sync_pulse']`. Cannot sync.")

    # Match sync pulses for when they are not perfectly paired
    # This returns pulses which are paired
    ref_sync_times, other_sync_times, cross = match_sync_pulses(
        ref_sync_times, other_sync_times
    )

    # Compute linear transformation
    A = np.vstack(
        [
            other_sync_times.times(),
            np.ones_like(other_sync_times.times()),
        ]
    ).T
    # intercept is the overall time shift between pulses
    # slope would be 1 if the sampling rates were true. In reality, the hardware
    # can slow down, meaning the sampling rate can change as time goes on
    slope, intercept = np.linalg.lstsq(A, ref_sync_times.times(), rcond=None)[0]

    # Plot before sync
    aligned = nap.Ts(other_sync_times.times() * slope + intercept)
    if np.mean(np.abs(aligned.times() - ref_sync_times.times())) > 0.1:
        raise ValueError("Sync pulses are still >0.1s apart after sync.\nSomething is wrong with the syncing.")

    # Add the synced time to the "other" pulses
    unsynced_time = other.times()
    other = other.as_dataframe()
    other["synced_time"] = unsynced_time * slope + intercept
    other = other.reset_index(drop=True).set_index("synced_time")
    other = other.drop(columns=["sync_pulse"])
    other = nap.TsdFrame(other)

    make_sync_plots(ref_sync_times, ref_tag, other_sync_times, other_tag, cross, slope, intercept, deriv_folder)

    return other, (slope, intercept)


def match_sync_pulses(ts1, ts2):
    """Match sync pulse timestamps pairs
    This function results in unshifted but paired timestamps
    which is neccessary because sometimes sync pulses are not paired

    """  
    
    # Initial alignment based on cross-correlation
    cross = nap.compute_crosscorrelogram(
        nap.TsGroup([ts2, ts1]),
        binsize=0.02,
        windowsize=ts2.time_support.end[-1],
    )
    lag = cross.idxmax().values[0]
    ts2_ = nap.Ts(ts2.times() + lag)

    # Format
    ts1_ = np.array(ts1.times())[:, None]
    ts2_ = np.array(ts2_.times())[:, None]

    # Create a KD-tree for fast nearest-neighbor search
    tree = cKDTree(ts2_)

    # Find the closest timestamps in ts2 for each timestamp in ts1
    distances, indices = tree.query(ts1_)

    # Filter out pairs where the distance is too large (optional, to avoid bad matches)
    # You can adjust this threshold as needed
    threshold = np.median(distances) + np.std(distances)
    index_ts1 = distances < threshold
    distances = distances[index_ts1]

    # List to hold the final matches
    final_index_ts1 = []
    final_index_ts2 = []
    final_distances = []

    # Sort the distances and indices by distance and then select the closest ones
    sorted_pairs = sorted(
        zip(
            distances,
            np.where(index_ts1)[0],
            indices[index_ts1],
            strict=True,
        ),
        key=lambda x: x[0],
    )

    # Iterate and pick the closest one for each ts2 (if not already used)
    used_indices = set()  # Set to keep track of already matched ts2 indices
    for distance, idx1, idx2 in sorted_pairs:
        if idx2 not in used_indices:
            final_index_ts1.append(idx1)
            final_index_ts2.append(idx2)
            final_distances.append(distance)
            used_indices.add(idx2)  # Mark this index as used
    final_distances = np.array(final_distances)
    final_index_ts1 = np.array(final_index_ts1)
    final_index_ts2 = np.array(final_index_ts2)

    # Threshold again
    abnormal_threshold = np.mean(final_distances) + 2 * np.std(final_distances)
    final_index_ts1 = final_index_ts1[final_distances < abnormal_threshold]
    final_index_ts2 = final_index_ts2[final_distances < abnormal_threshold]

    # Return the matched timestamps from ts1 and ts2
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="pynapple",
        )
        return ts1[final_index_ts1], ts2[final_index_ts2], cross


def make_sync_plots(
    ref_sync_times,
    ref_tag, 
    other_sync_times,
    other_tag,  
    cross, 
    slope, 
    intercept, 
    deriv_folder: Path
): 

    sync_path = deriv_folder / "sync_plots"
    sync_path.mkdir(exist_ok=True, parents=True)
   
    plt.plot(cross)
    plt.xlabel("lag (s)")
    plt.ylabel("cross-correlation")
    plt.tight_layout()
    plt.savefig(sync_path / f"{ref_tag}_vs_{other_tag}_cross_correlogram.png")
    plt.close()

    x = np.arange(len(other_sync_times))
    plt.plot(x, other_sync_times.times() - ref_sync_times.times())
    plt.scatter(x, other_sync_times.times() - ref_sync_times.times())
    plt.xlabel("pulse count")
    plt.ylabel(f"{other_tag} pulse - {ref_tag} pulse (s)")
    plt.tight_layout()
    plt.savefig(sync_path / f"{other_tag}_before_sync.png")
    plt.close()

    # Plot after sync
    plt.scatter(x, other_sync_times.times()* slope + intercept - ref_sync_times.times())
    plt.plot(x, other_sync_times.times()* slope + intercept - ref_sync_times.times())
    plt.xlabel("pulse count")
    plt.ylabel(f"{other_tag} pulse - {ref_tag} pulse (s)")
    plt.title(f"slope = {slope:.5f}, intercept = {intercept:.5f}")
    plt.tight_layout()
    plt.savefig(sync_path / f"{other_tag}_after_sync.png")
    plt.close()

import os
import numpy as np
import argparse
import pandas as pd
from sklearn import preprocessing


def process_visual_features(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(".csv"):
                input_file = os.path.join(root, file)
                print(input_file)

                relative_path = os.path.relpath(input_file, input_folder)
                output_subfolder = os.path.join(
                    output_folder, os.path.dirname(relative_path)
                )
                os.makedirs(output_subfolder, exist_ok=True)
                output_file = os.path.join(
                    output_subfolder, f"{os.path.splitext(file)[0]}_visual.npy"
                )

                if not os.path.exists(output_file):
                    extract_and_preprocess_single_sequence(input_file, output_file)

    print(f"Features extracted successfully and saved to {output_folder}")


def extract_and_preprocess_single_sequence(input_file, output_file):
    coordinates_columns = (
        ["frame"]
        + [
            "gaze_0_x",
            "gaze_0_y",
            "gaze_0_z",
            "gaze_1_x",
            "gaze_1_y",
            "gaze_1_z",
            "gaze_angle_x",
            "gaze_angle_y",
        ]
        + ["eye_lmk_x_" + str(i) for i in range(56)]
        + ["eye_lmk_y_" + str(i) for i in range(56)]
        + ["eye_lmk_X_" + str(i) for i in range(56)]
        + ["eye_lmk_Y_" + str(i) for i in range(56)]
        + ["eye_lmk_Z_" + str(i) for i in range(56)]
        + ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]
        + ["x_" + str(i) for i in range(68)]
        + ["y_" + str(i) for i in range(68)]
        + [
            "AU01_r",
            "AU02_r",
            "AU04_r",
            "AU05_r",
            "AU06_r",
            "AU07_r",
            "AU09_r",
            "AU10_r",
            "AU12_r",
            "AU14_r",
            "AU15_r",
            "AU17_r",
            "AU20_r",
            "AU23_r",
            "AU25_r",
            "AU26_r",
            "AU45_r",
        ]
        + [
            "AU01_c",
            "AU02_c",
            "AU04_c",
            "AU05_c",
            "AU06_c",
            "AU07_c",
            "AU09_c",
            "AU10_c",
            "AU12_c",
            "AU14_c",
            "AU15_c",
            "AU17_c",
            "AU20_c",
            "AU23_c",
            "AU25_c",
            "AU26_c",
            "AU28_c",
            "AU45_c",
        ]
    )

    data = pd.read_csv(input_file, sep=r",\s*|\s*,\s*", engine="python")
    data = data[coordinates_columns]

    # Downsampling the data
    data = data[(data["frame"] - 1) % 30 == 0][:]

    data = data.values

    # features
    # gaze[0:8]
    # eye_lmk[8:288]
    # head_pose[288:294]
    # landmarks[294:430]
    # AUs[430:465]
    data = data[:, 1:]

    # preprocess
    data_part1 = data[:, 0:447]
    data_part1 = preprocessing.scale(data_part1, axis=0)
    data[:, 0:447] = data_part1

    np.save(output_file, data)


def main():
    parser = argparse.ArgumentParser(
        description="Extract and normalize visual landmark features from CSV files"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="path/to/lmvd/raw_video_features",
        help="Directory containing landmark CSV files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="path/to/save/lmvd/visual_features",
        help="Directory to save extracted visual features",
    )

    args = parser.parse_args()

    process_visual_features(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

import os
import json
import csv
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=str, help="Path to the filtered manifest file", required=True
    )
    parser.add_argument(
        "--output_dir", type=str, help="Path to the outputs dir to save the csv", required=True
    )
    parser.add_argument(
        "--clips_dir", type=str, help="Path to the directory with audio clips", required=True
    )
    parser.add_argument(
        "--mode_specific", type=str, default='None', help="Extra processing according to mode"
    )
    return parser.parse_args()

def load_manifest(path, mode):
    with open(path, "r", encoding="utf8") as f:
        lines = f.readlines()

    data = []
    attributes = list(json.loads(lines[0]).keys())

    if 'recitationId' not in attributes and mode == 'ganjoor':
        attributes.insert(1, 'recitationId')
    
    for line in lines:
        x = json.loads(line)
        x['audio_filepath'] = os.path.basename(x['audio_filepath'])

        if 'recitationId' not in x and mode == 'ganjoor':
            x['recitationId'] = x['audio_filepath'].split("_")[0]

        data.append(x)
    return data, attributes

def save_manifest_csv(data, attributes, path):
    with open(path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=attributes)
        writer.writeheader()
        for x in data:
            writer.writerow(x)

# remove clips in clips_dir that are not in data[idx]['audio_filepath']
def remove_filtered(clips_dir, data):
    clips = os.listdir(clips_dir)
    clips = set(clips)
    data_files = set([x['audio_filepath'] for x in data])
    for clip in clips:
        if clip not in data_files:
            os.remove(os.path.join(clips_dir, clip))

def main():
    args = parse_args()

    data, attributes = load_manifest(args.manifest, args.mode_specific)
    save_manifest_csv(data, attributes, os.path.join(args.output_dir, "metadata.csv"))
    remove_filtered(args.clips_dir, data)



if __name__ == '__main__':
    main()
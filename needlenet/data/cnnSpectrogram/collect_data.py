import argparse
import os
import pandas as pd
import json
from torchaudio import load, save


def load_mono_signal(audio_path, channel):
    signal, sr = load(audio_path)
    signal_mono = signal[channel]

    signal_mono = signal_mono - signal_mono.mean()
    return signal_mono, sr
    
def cut_slices(signal, sr, start_sample, end_sample, slice_duration, stride):
    
    segment = signal[start_sample:end_sample]
    slice_samples = int(slice_duration * sr)
    stride_samples = int(stride * sr)

    slices = []
    for i in range(0, segment.shape[0] - slice_samples + 1, stride_samples):
        start = i
        end = i + slice_samples
        slice_segment = segment[start:end]

        global_start = start_sample + start
        global_end = start_sample + end

        slices.append((slice_segment, global_start / sr, global_end / sr))

    return slices

def extract_class_slices(signal, sr, rec_id, audio_ann, cfg, output_dir_path, dry_run):
    cls_name = cfg['class']
    events_list = cfg['events']
    slice_duration_ms = cfg['slice_ms']
    overlap = float(cfg['overlap'])
    padding = float(cfg['padding'])

    slice_duration = slice_duration_ms / 1000.0
    stride = slice_duration * (1 - overlap)

    if len(events_list) % 2 != 0:
        print(f"Warning: {rec_id} for class {cls_name}: events list length is not even.")
        return []

    event_pairs = list(zip(events_list[0::2], events_list[1::2]))

    class_output_dir_path = os.path.join(output_dir_path, cls_name)

    new_rows = []
    for pair in event_pairs:
        evt_start, evt_end = pair
        e1, e2 = str(evt_start), str(evt_end)
        if e1 not in audio_ann or e2 not in audio_ann:
            print(f"Skipping {rec_id}: missing required events {e1} or {e2}.")
            continue

        start_sample, end_sample = int(audio_ann[e1]['sample']), int(audio_ann[e2]['sample'])

        if padding != 0:
            start_sample = int(start_sample + (padding * sr))
            end_sample = int(end_sample - (padding * sr))
        slices = cut_slices(
            signal, sr, start_sample, end_sample, 
            slice_duration=slice_duration, 
            stride=stride
        )

        for slice in slices:
            slice_signal, slice_start, slice_end = slice
            slice_filename = f"{rec_id}_slice_{slice_start:.3f}-{slice_end:.3f}.wav"
            slice_path = os.path.join(class_output_dir_path, slice_filename)

            if not dry_run:
                save(slice_path, slice_signal.unsqueeze(0), sr)
            
            new_rows.append({
                'path': slice_path,
                'label': cls_name,
                'recording_id': rec_id
            })

    return new_rows

def process_recording(ann_file, annotation_dir_path, audio_dir_path, output_dir_path, collect_config, dry_run, channel=0):
    rec_id = ann_file.replace('.json', '')
    ann_path = os.path.join(annotation_dir_path, ann_file)

    with open(ann_path, "r") as f:
        ann_data = json.load(f)

    try:
        audio_path = os.path.join(audio_dir_path, ann_data['audio_file'])
        audio_ann = ann_data['audio_annotations']
    except Exception:
        print(f'Skipping {rec_id}: missing audio annotations')
        return []
    
    signal_mono, sr = load_mono_signal(audio_path, channel)

    recording_new_rows = []
    
    for cfg in collect_config:
        class_rows = extract_class_slices(
            signal_mono, sr,
            rec_id,
            audio_ann,
            cfg,
            output_dir_path,
            dry_run
        )
        recording_new_rows.extend(class_rows)

    print(f"Processed {rec_id}: extracted {len(recording_new_rows)} slices.")
    return recording_new_rows

def parse_args():
    parser = argparse.ArgumentParser(description="Script for collecting data from datasets and spliting into classes and slices.")
    parser.add_argument('--data-dir', required=True, type=str,
                        help='Path to the directory which contains annotations and audio folders.')
    parser.add_argument('--audio-dir', required=True, type=str,
                        help='Path to the directory which contains audio files.')
    parser.add_argument('--output-dir-name', required=True, type=str,
                        help='Name of the output directory')
    parser.add_argument('--collect-config-path', required=True, type=str,
                        help='Path to the JSON file specifying classes collection configuration.')
    parser.add_argument('--dry-run', action='store_true',
                        help='If set, the script will only print information without performing action')
    return parser.parse_args()
    

def main():
    args = parse_args()

    try:
        with open(args.collect_config_path, 'r') as f:
            collect_config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON string for collect-config") from e

    dataset_path = args.data_dir
    annotation_dir_path = os.path.join(dataset_path, 'annotations')
    audio_dir_path = os.path.join(dataset_path, args.audio_dir)

    if not os.path.exists(annotation_dir_path):
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir_path}")
    
    if not os.path.exists(audio_dir_path):
        raise FileNotFoundError(f"Audio directory not found: {audio_dir_path}")

    output_dir_path = args.output_dir_name
    os.makedirs(output_dir_path, exist_ok=True)

    for cfg in collect_config:
        class_dir = os.path.join(output_dir_path, cfg['class'])
        os.makedirs(class_dir, exist_ok=True)


    files_to_process = [f for f in os.listdir(annotation_dir_path) if f.endswith('.json')]

    print(f"Found {len(files_to_process)} new recordings to process.")

    all_rows = []
    for ann_file in files_to_process:

        new_rows = process_recording(
            ann_file,
            annotation_dir_path,
            audio_dir_path,
            output_dir_path,
            collect_config,
            args.dry_run
        )
        
        all_rows.extend(new_rows)
        
    if all_rows:
        manifest_path = os.path.join(output_dir_path, 'manifest.csv')
        manifest_df = pd.DataFrame(columns=['path', 'label', 'recording_id'], data=all_rows)

        if not args.dry_run:
            manifest_df.to_csv(manifest_path, index=False)
            print(f"Created manifest saved to {manifest_path}.")
        else:
            print("Dry run enabled, no files were written.")
            print(f'Total slices after dry run: {len(manifest_df)}')
    else:
        print("No new slices were extracted.")


if __name__ == "__main__":
    main()
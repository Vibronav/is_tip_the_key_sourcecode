import os
import shutil
import pandas as pd

def merge_datasets(cfg, dataset1, dataset2):

    dataset1_path = os.path.join(cfg['data']['root'], f'{cfg["data"]["name"]}/{dataset1}')
    dataset2_path = os.path.join(cfg['data']['root'], f'{cfg["data"]["name"]}/{dataset2}')
    
    merged_dataset_name = f'{dataset1}_AND_{dataset2}'
    merged_dataset_path = os.path.join(cfg['data']['root'], f'{cfg["data"]["name"]}/{merged_dataset_name}')

    os.makedirs(merged_dataset_path, exist_ok=True)

    dataset1_items = os.listdir(dataset1_path)
    files_number = sum(len(files) for _, _, files in os.walk(dataset1_path))
    print(f'Files in dataset1 before merge: {files_number}')

    for item in dataset1_items:
        if item == 'manifest.csv':
            continue

        src_path = os.path.join(dataset1_path, item)
        dst_path = os.path.join(merged_dataset_path, item)
        shutil.move(src_path, dst_path)

    dataset2_items = os.listdir(dataset2_path)
    files_number = sum(len(files) for _, _, files in os.walk(dataset2_path))
    print(f'Files in dataset2 before merge: {files_number}')

    for item in dataset2_items:
        if item == 'manifest.csv':
            continue

        src_path = os.path.join(dataset2_path, item)
        dst_path = os.path.join(merged_dataset_path, item)

        if os.path.isdir(src_path) and os.path.exists(dst_path):
            for sub_item in os.listdir(src_path):
                shutil.move(
                    os.path.join(src_path, sub_item),
                    os.path.join(dst_path, sub_item)
                )
            os.rmdir(src_path)
        else:
            shutil.move(src_path, dst_path)

    files_number = sum(len(files) for _, _, files in os.walk(merged_dataset_path))
    print(f'Files in merged dataset: {files_number}')


    manifest1 = os.path.join(dataset1_path, 'manifest.csv')
    manifest2 = os.path.join(dataset2_path, 'manifest.csv')
    merged_manifest = os.path.join(merged_dataset_path, 'manifest.csv')

    print(f'Row number in manifest1 before merge: {len(pd.read_csv(manifest1))}')
    print(f'Row number in manifest2 before merge: {len(pd.read_csv(manifest2))}')

    dfs = []
    for manifest in [manifest1, manifest2]:
        df = pd.read_csv(manifest)

        df['path'] = df['path'].apply(
            lambda x: f'{merged_dataset_name}/{x.split("/", 1)[1]}'
        )
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    print(f'Row number in merged manifest: {len(merged_df)}')

    merged_df.to_csv(merged_manifest, index=False)
    return merged_dataset_name
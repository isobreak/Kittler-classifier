import os
import cv2
import pandas as pd
from classic_ML.features import get_features


def get_features_df(directory: str) -> pd.DataFrame:
    """
    :param directory: path to directory (label_1, label_2,...)
    :return: pd.DataFrame with ['name', 'label', features]
    """
    if directory is None:
        print('No such directory')

    labels = os.listdir(directory)
    data = []
    for label in labels:
        files = os.listdir(os.path.join(directory, label))

        # exclude rotated images
        del_files = []
        nod_files = []
        for i, file in enumerate(files):
            if '_' in file:
                del_files.append(file)
            else:
                nod_files.append(file)
        files = nod_files
        print('Following files were considered as rotated:')
        print(del_files)
        print('Extracting features from following files:')
        print(nod_files)

        for i, file in enumerate(files):
            print(i, file)
            img = cv2.imread(os.path.join(directory, label, file))
            row = {'name': file, 'label': label}

            features = get_features(img)
            if features is None:
                print(f'File {file} has been discarded')
                continue
            row.update(features)
            print(row)

            data.append(row.values())

    df = pd.DataFrame(data=data, columns=row.keys())
    df.set_index('name', inplace=True)
    print(df)

    return df

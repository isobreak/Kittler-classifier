import os

import cv2
from joblib import dump, load

from classic_ML.dataframe_creation import get_features_df
from classic_ML.prediction import classic_predict
from classic_ML.train import train
from common.dataset_creation import create_dataset_folder, rotate_images_in_folder
from common.preprocessing import get_cropped_image


def check_prediction(model_name: str, folder_path: str, max_images_number: int):
    """
    A function for CNN prediction check.
    Takes care of creating preprocessed dataset and training results
    (plots, models, csv dataset with models' metrics)\n
    REQUIRES (raw) dataset (class1, class2) at "/data/datasets/raw" for correct work\n
    :param model_name: name of the model to be used for prediction.\n
    Should be in /data/training_results/Models.csv or will be added after calculation
    :param folder_path: path to a folder of the following structure (class1, class2) for prediction check
    :param max_images_number: number of images to be checked in each folder
    """
    # setting paths
    res_path = os.path.join('data', 'training_results')
    model_folder = os.path.join(res_path, 'models')
    model_path = os.path.join(model_folder, model_name + '.joblib')
    # loading/training clf
    if os.path.exists(model_path):
        clf = load(model_path)
    else:
        print(f'Model with name {model_name} has not been fount at "{model_folder}"', end='\n\n')

        # creating preprocessed dataset (if it is necessary)
        preprocessed_dataset_path = os.path.join('data', 'datasets', 'preprocessed')
        raw_dataset_path = os.path.join('data', 'datasets', 'raw')
        if os.path.exists(raw_dataset_path):
            create_dataset_folder(raw_dataset_path, preprocessed_dataset_path)
            rotate_images_in_folder(preprocessed_dataset_path, aug_numbers=(2, 4))
        else:
            print(f'Could not create preprocessed dataset.\nThere is no dataset at "{raw_dataset_path}"')

        # creating dataframe dataset
        df_dataset_path = os.path.join('data', 'datasets', 'features.csv')
        if not os.path.exists(df_dataset_path):
            df = get_features_df(directory=preprocessed_dataset_path)
            df.to_csv(df_dataset_path)
            print(f'Dataframe dataset has been created at {df_dataset_path}')

        # creating folders
        if not os.path.exists(res_path):
            os.mkdir(res_path)
            print(f'Results folder created at {res_path}')
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
            print(f'Models folder created at {model_folder}')
        print()

        # training model
        clf = train(df_path=df_dataset_path, seed=17)

        # saving model
        dump(clf, model_path)

    # prediction
    print(f'"{model_name}" is using for prediction:', end='\n')

    folders = os.listdir(folder_path)
    for folder in folders:
        print('\nFolder:', folder)
        pictures = os.listdir(os.path.join(folder_path, folder))
        for i, picture in enumerate(pictures):
            if i == max_images_number:
                break

            picture_path = os.path.join(folder_path, folder, picture)
            img = cv2.imread(picture_path)
            pred_class = classic_predict(img, clf)
            cv2.imshow('window', cv2.resize(get_cropped_image(img), (750, 750)))
            window_name = f'{folder}: {picture} (pred: {pred_class})'
            cv2.setWindowTitle('window', window_name)
            cv2.waitKey(1000)
            print(f'{picture}: {pred_class}')

if __name__ == "__main__":
    check_prediction('Random_forest_clf', os.path.join('data', 'datasets', 'raw'), 20)
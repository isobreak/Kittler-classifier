import cv2

from CNN.models import StructurelessClf_1 as Best_model_arch
from CNN.prediction import str_less_predict_class, str_less_predict
from CNN.train import *
from common.dataset_creation import create_dataset_folder, rotate_images_in_folder
from common.preprocessing import get_cropped_image


def check_prediction(model_name: str, folder_path: str, max_images_number: int):
    """
    A function for classic_ML prediction check.
    Takes care of creating preprocessed dataset, features (df) dataset and
    saving new model at /data/training_results/models\n
    REQUIRES (raw) dataset (class1, class2) at "/data/datasets/raw" for correct work\n
    :param model_name: name of the model to be used for prediction.\n
    Should be in /data/training_results/models to be used. Will be calculated and saved there otherwise
    :param folder_path: path to a folder of the following structure (class1, class2) for prediction check
    :param max_images_number: number of images to be checked in each folder
    """
    # setting paths
    res_path = os.path.join('data', 'training_results')
    csv_path = os.path.join(res_path, 'Models.csv')
    model_folder = os.path.join(res_path, 'models')
    model_path = os.path.join(model_folder, model_name + '.pt')
    plots_folder = os.path.join(res_path, 'plots')

    if os.path.exists(model_path):
        model = torch.load(model_path)
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

        # creating folders
        if not os.path.exists(res_path):
            os.mkdir(res_path)
            print(f'Results folder created at {res_path}')
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
            print(f'Models folder created at {model_folder}')
        if not os.path.exists(plots_folder):
            os.mkdir(plots_folder)
            print(f'Plots folder created at {plots_folder}')
        print()

        # training model
        seed = 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

        model = Best_model_arch()
        learning_params = {'epochs': 23, 'optimizer': torch.optim.Adam, 'lr': 0.001}

        # creating results path
        if not os.path.exists(res_path):
            os.mkdir(res_path)

        model = train_cnn(dataset_path=preprocessed_dataset_path, model=model, save_csv_path=csv_path,
                          save_model_folder=model_folder, model_name=model_name, save_plots_folder=plots_folder,
                          learning_params=learning_params)

    # printing model metrics
    print()
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(df.drop(['Unnamed: 0'], axis=1))
        print(f'"{model_name}" is using for prediction:', end='\n')

    # prediction
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    folders = os.listdir(folder_path)
    for folder in folders:
        print('\nFolder:', folder)
        pictures = os.listdir(os.path.join(folder_path, folder))
        for i, picture in enumerate(pictures):
            if i == max_images_number:
                break

            picture_path = os.path.join(folder_path, folder, picture)
            img = cv2.imread(picture_path)
            pred_class = str_less_predict_class(img, model)

            cv2.imshow('window', cv2.resize(get_cropped_image(img), (750, 750)))
            window_name = f'{folder}: {picture} (pred: {pred_class})'
            cv2.setWindowTitle('window', window_name)
            cv2.waitKey(1000)
            print(f'{picture}: {pred_class}')

if __name__ == "__main__":
    check_prediction('CNN_clf', os.path.join('data', 'datasets', 'raw'), 20)
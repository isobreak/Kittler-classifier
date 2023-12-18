import os
import cv2
from common.preprocessing import get_cropped_image


def create_dataset_folder(src: str, dst: str, size_step: int = 250, problems_path: str = None,
                          tumor_only: bool = False, solve_hair: bool = True):
    """
    :param src: path to src folder with 2 subfolders (of 2 classes)
    :param dst: path for saving dataset
    :param size_step: resolution step of images in preprocessed dataset
    :param problems_path: path to a .txt file with names of images to be skipped (if exists)
    :param tumor_only: apply mask to a tumor
    :param solve_hair: allow to solve problems with tumor detection on images with hair, but decreases quality of selection
    """
    updating = False

    problem_list = []
    if problems_path is not None:
        problems_d = open(problems_path)
        problems = problems_d.readlines()
        for problem in problems:
            problem = problem[:-1]
            problem_list.append(problem)

        print(f'Following images will be discarded due to their presence in the problems list ({problems_path}')
        print(problem_list)
    else:
        print('The list of problematic images was not found, all images are considering')

    discarded_list = []

    if os.path.exists(src):
        if not os.path.exists(dst):
            os.mkdir(dst)
            print(dst, 'created')

        class_folders = os.listdir(os.path.join(src))

        for class_folder in class_folders:
            dst_class_folder = os.path.join(dst, class_folder)
            if not os.path.exists(dst_class_folder):
                os.mkdir(dst_class_folder)
                print(dst_class_folder, 'created')

            files = os.listdir(os.path.join(src, class_folder))
            for file in files:
                if file not in problem_list:
                    if updating is True:
                        print(file, '...', end='')
                    src_file = os.path.join(src, class_folder, file)
                    dst_file = os.path.join(dst, class_folder, file)
                    if not os.path.exists(dst_file):
                        img = cv2.imread(src_file)
                        img = get_cropped_image(img, size_step=size_step,
                                                solve_hair=solve_hair, tumor_only=tumor_only)
                        cv2.imwrite(dst_file, img)
                        print(dst_file, 'is saved', end='\n')
                        updating = True
                    else:
                        if updating is True:
                            print(dst_file, 'already exists')

                else:
                    discarded_list.append(file)
                    print(file, 'discarded')
    if len(discarded_list) != 0:
        print('Following files were discarded:\n', discarded_list)
    print(f'Dataset is located at {dst}', end='\n\n')


def rotate_images_in_folder(dataset_path: str, aug_numbers: tuple[int, int]):
    """
    Saves rotated instances of image based on aug_numbers
    :param dataset_path: path to a dataset (class1, class2, ...)
    :param aug_numbers: numbers of rotated images (2, 3 or 4)
    """
    folders = os.listdir(dataset_path)
    for i, folder in enumerate(folders):
        n = aug_numbers[i]
        folder_path = os.path.join(dataset_path, folder)
        if n == 2 or n == 3 or n == 4:
            images = os.listdir(folder_path)
            for image in images:
                if 'rotated' in image:
                    continue

                image_path = os.path.join(folder_path, image)
                orig = cv2.imread(image_path)
                img_90 = cv2.rotate(orig, cv2.ROTATE_90_CLOCKWISE)
                if not os.path.exists(image_path.replace('.jpg', '_rotated_1.jpg')):
                    cv2.imwrite(image_path.replace('.jpg', '_rotated_1.jpg'), img_90)
                if n > 2:
                    img_180 = cv2.rotate(orig, cv2.ROTATE_180)
                    if not os.path.exists(image_path.replace('.jpg', '_rotated_2.jpg')):
                        cv2.imwrite(image_path.replace('.jpg', '_rotated_2.jpg'), img_180)
                if n > 3:
                    img_m90 = cv2.rotate(img_90, cv2.ROTATE_180)
                    if not os.path.exists(image_path.replace('.jpg', '_rotated_3.jpg')):
                        cv2.imwrite(image_path.replace('.jpg', '_rotated_3.jpg'), img_m90)
        else:
            print('Incorrect value (n should be in range 2, 3, 4)')

    print('Augmentation complete')

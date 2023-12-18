import cv2
import numpy as np
import torch
import torchvision

from common.preprocessing import get_cropped_image


def str_less_predict(img: np.ndarray, model) -> int:
    """
    :param img: image to be classified
    :param model: CNN model used for classification
    :return: 0 - 'monochrome' or 1 - 'multicolor'
    """
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    img = get_cropped_image(img)
    img = cv2.resize(img, dsize=(250, 250))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))

    img = torch.tensor(img, device=device, dtype=torch.float)
    img /= 255

    transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    img = transform(img)

    with torch.no_grad():
        img = img * 2 - 1
        y = model(img)
        y = y.to('cpu')
        s = torch.sigmoid(y)
        res = torch.round(s)

    return res


def str_less_predict_class(img: np.ndarray, model) -> str:
    """
    :param img: image to be classified
    :param model: CNN model used for classification
    :return: class the image belongs to
    """
    res = str_less_predict(img, model)
    if res == 1:
        label = 'multicolor'
    elif res == 0:
        label = 'monochrome'
    else:
        label = 'STRANGE CLASS'

    return label

import cv2
import numpy as np
from pandas import DataFrame

from classic_ML.features import get_features
from common.preprocessing import get_cropped_image


def classic_predict(img: np.ndarray, model) -> int:
    """
    :param img: image to be classified
    :param model: classic_ML model used for classification
    :return: 'monochrome' or 'multicolor'
    """
    img = get_cropped_image(img)
    img = cv2.resize(img, dsize=(250, 250))
    features = get_features(img)

    df = DataFrame([features])
    label = model.predict(df).item()

    return label

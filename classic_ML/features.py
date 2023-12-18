import cv2
import numpy as np
from common.preprocessing import get_tumor_contour


def get_features(img: np.ndarray) -> dict[str, int]:
    """
    :param img: image from which features should be extracted
    :return: dict of features ('black, 'dark', 'light', 'gray', 'blue', 'orange', 'yellow', 'white', 'red', 'purple')
    """
    color_ranges = get_color_ranges_rgb()
    tumor_contour = get_tumor_contour(img)
    tumor_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(tumor_mask, [tumor_contour], -1, color=[255], thickness=-1)

    features = {}
    for key, color_range in color_ranges.items():
        area = get_area_of_color(img, tumor_mask=tumor_mask, color_range=color_range, color_space='bgr')
        if area is None:
            print('Features could not be found')
            return None
        features[key] = area

    return features


def get_area_of_color(img: np.ndarray, tumor_mask: np.ndarray,
                      color_range: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
                      color_space: str = 'hsv') -> int:
    """
    :param img: image to be investigated
    :param tumor_mask: tumor mask on image
    :param color_range: color range in which the pixel should fit
    :param color_space: color space of color_range ('hsv', 'bgr')
    :return: number of pixels fitted into color_range
    """
    ch_1, ch_2, ch_3 = cv2.split(img)
    if color_space == 'hsv':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif color_space == 'bgr':
        img = img
    else:
        print('Wrong color space')
        return None

    _, ch_1_low = cv2.threshold(ch_1, color_range[0][0], 255, cv2.THRESH_BINARY)
    _, ch_2_low = cv2.threshold(ch_2, color_range[1][0], 255, cv2.THRESH_BINARY)
    _, ch_3_low = cv2.threshold(ch_3, color_range[2][0], 255, cv2.THRESH_BINARY)
    _, ch_1_high = cv2.threshold(ch_1, color_range[0][1], 255, cv2.THRESH_BINARY_INV)
    _, ch_2_high = cv2.threshold(ch_2, color_range[1][1], 255, cv2.THRESH_BINARY_INV)
    _, ch_3_high = cv2.threshold(ch_3, color_range[2][1], 255, cv2.THRESH_BINARY_INV)

    ch_1 = cv2.bitwise_and(ch_1_low, ch_1_high)
    ch_2 = cv2.bitwise_and(ch_2_low, ch_2_high)
    ch_3 = cv2.bitwise_and(ch_3_low, ch_3_high)

    bg = cv2.bitwise_and(ch_1, ch_2)
    mask = cv2.bitwise_and(bg, ch_3)
    mask = cv2.bitwise_and(mask, tumor_mask)

    return cv2.countNonZero(mask)


def get_color_ranges_rgb_from_book() -> dict[str, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    return {'black': ((0, 55), (0, 55), (0, 55)),
            'dark': ((0, 34), (2, 41), (33, 98)),
            'light': ((25, 81), (53, 118), (94, 171)),
            'gray': ((88, 181), (85, 181), (87, 181)),
            'blue': ((100, 182), (4, 67), (0, 68)),
            'orange': ((0, 52), (61, 163), (160, 255)),
            'yellow': ((40, 128), (205, 244), (229, 255)),
            'white': ((218, 255), (226, 255), (231, 255)),
            'red': ((0, 109), (24, 126), (209, 255)),
            'purple': ((87, 155), (8, 47), (41, 88))}


def get_color_ranges_rgb() -> dict[str, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    return {'black': ((0, 55), (0, 55), (0, 55)),
            'dark': ((0, 50), (2, 60), (33, 100)),
            'light': ((31, 59), (67, 116), (120, 181)),
            'gray': ((56, 165), (56, 165), (56, 165)),
            'blue': ((60, 255), (0, 175), (0, 175)),
            'orange': ((0, 52), (61, 163), (160, 255)),
            'yellow': ((40, 128), (205, 244), (229, 255)),
            'white': ((200, 255), (195, 255), (190, 255)),
            'red': ((0, 109), (24, 126), (209, 255)),
            'purple': ((87, 155), (8, 47), (41, 88))}


def get_color_ranges_hsv() -> dict[str, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    return {'black': ((0, 0), (0, 0), (3, 55)),
            'dark': ((2, 6), (126, 248), (35, 100)),
            'bright': ((12, 14), (137, 187), (123, 200)),
            'gray': ((0, 0), (0, 0), (59, 165)),
            'blue': ((120, 120), (36, 222), (69, 213)),
            'orange': ((13, 17), (130, 232), (192, 238)),
            'yellow': ((21, 25), (59, 184), (242, 255)),
            'white': ((0, 120), (0, 13), (201, 255)),
            'red': ((0, 0), (49, 243), (142, 255)),
            'purple': ((131, 133), (57, 215), (120, 255))}
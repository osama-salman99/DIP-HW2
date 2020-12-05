import cv2
import numpy as np
from PIL import Image


def read_GIF(path: str):
    if path.split('.')[-1].lower() != 'gif':
        return cv2.imread(path)[:, :, 0].astype('float64')
    gif = cv2.VideoCapture(path)
    if gif is None:
        return None
    ret, frame = gif.read()
    img = Image.fromarray(frame)
    open_cv_image = np.array(img, dtype='float64')
    open_cv_image = open_cv_image[:, :, ::-1]
    return open_cv_image[:, :, 0]


def mean(images: np.ndarray):
    return images.mean(axis=0)


def median(images: np.ndarray):
    return np.median(images, axis=0)


def percentile(images: np.ndarray, percent: int):
    images.sort(axis=0)
    index1 = int(np.floor(percent * (images.shape[0] + 1) / 100))
    index2 = int(np.ceil(percent * (images.shape[0] + 1) / 100))
    return (images[index1 - 1] + images[index2 - 1]) / 2


def PSNR(image1: np.ndarray, image2: np.ndarray):
    height, width = image1.shape
    differenceImage = image2 - image1
    differenceImage **= 2
    difference = differenceImage.sum().sum()
    PSNR_ = 10 * np.log10(((255 ** 2) * height * width) / difference)
    return PSNR_


def c_log(image: np.ndarray):
    # noinspection PyArgumentList
    mx = image.max()
    return 255 / np.log10(1 + mx)


def c_inverse_log(image: np.ndarray):
    # noinspection PyArgumentList
    mx = image.max()
    return np.log10(1 + 255) / mx


def c_linear(image: np.ndarray):
    # noinspection PyArgumentList
    return 255 / image.max()

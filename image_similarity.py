import cv2
import numpy as np
# For Sasha from Dima with love


_DEBUG = False


def _scale_image(image, w, h):
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)


def _to_float(image):
    res = np.float32(image)
    res = 1.0 - res / 255.0
    return res


def is_sample_in_image(image, sample, kernel_size=3):
    h, w = sample.shape[:2]
    scaled_image = _scale_image(image, w, h)

    if _DEBUG:
        cv2.imshow("Scaled", scaled_image)

    kernel = np.ones((kernel_size, kernel_size), np.float32)

    pattern = _to_float(scaled_image)
    candidate = _to_float(sample)
    dilated_candidate = cv2.dilate(candidate, kernel)

    overlap = np.multiply(pattern, dilated_candidate)

    if _DEBUG:
        cv2.imshow("Pattern", pattern)
        cv2.imshow("Candidate", dilated_candidate)
        cv2.imshow("Overlap", overlap)

        print("overlap sum = {:.2f}".format(np.sum(overlap)))
        print("pattern sum = {:.2f}".format(np.sum(pattern)))

    return np.sum(overlap) / np.sum(pattern)
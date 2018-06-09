import cv2
import numpy as np
# For Sasha from Dima with love


_DEBUG = False


def _phase_correlation(a, b):
    G_a = np.fft.fft2(a)
    G_b = np.fft.fft2(b)
    conj_b = np.ma.conjugate(G_b)
    R = G_a*conj_b
    R /= np.absolute(R)
    r = np.fft.ifft2(R).real
    return r


def _analyze_correlation(correlogram, range=5):
    rolled_correlogram = np.roll(np.roll(correlogram, range, axis=0), range, axis=1)
    if _DEBUG:
        cv2.imshow("Rolled", 5 *rolled_correlogram)
    return np.max(rolled_correlogram[:2*range, :2*range])


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
    # dilated_pattern = cv2.dilate(pattern, kernel)

    candidate = _to_float(sample)
    dilated_candidate = cv2.dilate(candidate, kernel)

    overlap = np.multiply(pattern, dilated_candidate)

    if _DEBUG:
        cv2.imshow("Pattern", pattern)
        cv2.imshow("Candidate", dilated_candidate)
        cv2.imshow("Overlap", overlap)

        print("overlap sum = {:.2f}".format(np.sum(overlap)))
        print("pattern sum = {:.2f}".format(np.sum(pattern)))

    corr = _phase_correlation(pattern, candidate)

    if _DEBUG:
        cv2.imshow("Corr", 5 * corr)
        print("Max correleation = {}".format(np.max(corr)))

    # max_correlation_around = _analyze_correlation(corr, range=3)

    # return np.sum(overlap) / np.sum(pattern), max_correlation_around
    return np.sum(overlap) / np.sum(pattern), 0
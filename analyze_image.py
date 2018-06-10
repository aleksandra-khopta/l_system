#!/bin/python

import cv2
import numpy as np
import math


_VISUALIZATION_STEP = 10
_counter = 0


def line_to_angle(line):
    x1, y1, x2, y2 = line
    angle = np.rad2deg(np.arctan2(x1 - x2, y1 - y2))
    while angle < -90:
        angle += 180
    while angle > 90:
        angle -= 180
    return angle


def line_len(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def calculate_major_angle(image_src):
    lines, width, prec, nfa = cv2.createLineSegmentDetector().detect(image_src)
    # line_len_threshold = max(image_src.shape[:2]) // 30
    line_len_threshold = 8

    filtered_lines = filter(lambda l: line_len(l) > line_len_threshold, map(lambda l: l[0], lines))

    angles = [line_to_angle(line) for line in filtered_lines]
    hist = np.histogram(angles, bins=range(-90, 90))

    sorted_angles = sorted(zip(hist[1], hist[0]), key=lambda x: -x[1])
    sorted_angles = list(filter(lambda pair: pair[1] > 2, sorted_angles))
    sorted_angles = sorted(sorted_angles[:10], key=lambda pair: pair[0])

    min_angle = 5
    negative_angles = list(filter(lambda x: x < -min_angle, [a[0] for a in sorted_angles]))
    positive_angles = list(filter(lambda x: x > min_angle, [a[0] for a in sorted_angles]))
    negative_candidate_angle = max(negative_angles or [None])
    positive_candidate_angle = min(positive_angles or [None])

    # print(negative_candidate_angle)
    # print(positive_candidate_angle)

    if negative_candidate_angle is None:
        major_angle = positive_candidate_angle
    elif positive_candidate_angle is None:
        major_angle = negative_candidate_angle
    elif abs(negative_candidate_angle + positive_candidate_angle) < min_angle:
        major_angle = (positive_candidate_angle - negative_candidate_angle) / 2
    else:
        major_angle = min([positive_candidate_angle, -negative_candidate_angle])
    return major_angle


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape[:2]
    mask = image == 255
    top = 0
    bottom = h-1
    left = 0
    right = w-1

    while np.sum(mask[top, :]) == 0:
        top += 1
    while np.sum(mask[bottom, :]) == 0:
        bottom -= 1
    while np.sum(mask[:, left]) == 0:
        left += 1
    while np.sum(mask[:, right]) == 0:
        right -= 1

    return image[top:bottom, left:right]


def create_debug_image(input_image, points_mask, point):
    res_image = 255 * np.stack([input_image, input_image, input_image], axis=-1)
    res_image[points_mask == 1] = (0, 0, 200)
    cv2.circle(res_image, point, 3, (200, 0, 0), thickness=2)
    return res_image


def show_debug_image(window_name, input_image, points_mask, point, wait_time=-1):
    global _counter
    _counter += 1
    if _counter % _VISUALIZATION_STEP == 0:
        res_image = create_debug_image(input_image, points_mask, point)
        cv2.imwrite("debug/match_map_.png", res_image)
        cv2.imshow(window_name, res_image)
        cv2.waitKey(wait_time)


def subimage(image, center, theta, width, height):
    if 45 < theta <= 90:
        theta = theta - 90
        width, height = height, width

    theta *= math.pi / 180 # convert to rad
    v_x = (math.cos(theta), math.sin(theta))
    v_y = (-math.sin(theta), math.cos(theta))
    s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
    s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)
    mapping = np.array([[v_x[0],v_y[0], s_x], [v_x[1],v_y[1], s_y]])
    return cv2.warpAffine(image, mapping, (width, height), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def subimage_best(image, bottom_center, theta, scale):
    h, w = image.shape[:2]
    width = int(w * scale)
    height = int(h * scale)

    if 45 < theta <= 90:
        theta = theta - 90
        width, height = height, width
    theta *= math.pi / 180 # convert to rad

    bottom_x, bottom_y = bottom_center
    center_x = bottom_x + (height // 2) * math.sin(theta)
    center_y = bottom_y - (height // 2) * math.cos(theta)
    center = (int(center_x), int(center_y))

    v_x = (math.cos(theta), math.sin(theta))
    v_y = (-math.sin(theta), math.cos(theta))
    s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
    s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)
    mapping = np.array([[v_x[0],v_y[0], s_x], [v_x[1],v_y[1], s_y]])
    return cv2.warpAffine(image, mapping, (width, height), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

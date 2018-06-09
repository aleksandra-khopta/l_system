import cv2
import numpy as np
import math
import image_similarity
import sys

sys.setrecursionlimit(50000)


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


_counter = 0


def _show_debug_image(window_name, input_image, points_mask, point, wait_time=-1):
    global _counter
    _counter += 1
    if _counter % 20 == 0:
    # if True:
        res_image = np.stack([input_image, input_image, input_image], axis=-1)
        res_image[points_mask == 1] = (0, 0, 200)
        cv2.circle(res_image, point, 3, (200, 0, 0), thickness=2)
        cv2.imwrite("debug.png", res_image)
        cv2.imshow(window_name, res_image)

        # cv2.imshow("Matches", points_mask)

        cv2.waitKey(wait_time)


# image_src = cv2.imread("fractal.jpg")
image_src = load_image("fractal.png")
# visited = np.zeros_like(image_src)
# matches = np.zeros_like(image_src)
visited = np.zeros(image_src.shape[:2])
matches = np.zeros(image_src.shape[:2])
h, w = image_src.shape[:2]


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

################### TO DO #################
def subimage_best(image, bottom_center, theta, scale):
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


class color(object):
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

def equal(image, subimage):
    similarity, corr = image_similarity.is_sample_in_image(image, subimage)
    return similarity > 0.8

def find_elem(x_start, y_start):
    bottom_center = (x_start, y_start)
    theta_range = [-25, 0, 25]
    scale_range = [0.5]
    for theta in theta_range:
        for scale in scale_range:
            sub_img = subimage_best(image_src, bottom_center, theta, scale)
            if equal(image_src, sub_img):
                return True, theta, scale
    return False, 0, 0


l_system1 = []
_POSSIBLE_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def bypass(back_color, x, y):
    if x < 0 or y < 0 or x >= w or y >= h:
        return
    if visited[y, x] == 1:
        return
    if image_src[y, x] != back_color:
        visited[y, x] = 1
        _show_debug_image("Visited", visited, matches, (x, y), wait_time=1)
        is_sub_img, theta, scale = find_elem(x, y)
        if is_sub_img:
            matches[y, x] = 1
            if theta > 0:
                l_system1.append("+")
            elif theta < 0:
                l_system1.append("-")
            l_system1.append("X")
            ######## TO DO : search in the frame ##########
        for (dy, dx) in _POSSIBLE_MOVES:
            bypass(back_color, x + dx, y + dy)

print(image_src.shape)
print(visited.shape)

last_row = image_src[-1, :]
x = np.argmin(last_row)

bypass(255, x, h - 1)
print(l_system1)

cv2.imshow("Input", image_src)
cv2.imshow("Visited", visited)
cv2.waitKey()

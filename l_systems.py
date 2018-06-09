import cv2
import numpy as np
import sys


from analyze_image import load_image, show_debug_image, subimage_best, equal

sys.setrecursionlimit(50000)

image_src = load_image("images/fractal.png")
visited = np.zeros(image_src.shape[:2])
matches = np.zeros(image_src.shape[:2])
h, w = image_src.shape[:2]


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
        show_debug_image("Visited", visited, matches, (x, y), wait_time=1)
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
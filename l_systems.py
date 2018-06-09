import cv2
import numpy as np
import argparse
from collections import deque
import pickle

from analyze_image import load_image, show_debug_image, subimage_best, equal, create_debug_image


_POSSIBLE_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_POSSIBLE_THETA = [-25, 0, 25]
_POSSIBLE_SCALE = [0.5]


def find_elem(image_src, x_start, y_start):
    bottom_center = (x_start, y_start)
    for theta in _POSSIBLE_THETA:
        for scale in _POSSIBLE_SCALE:
            sub_img = subimage_best(image_src, bottom_center, theta, scale)
            if equal(image_src, sub_img):
                return True, theta, scale
    return False, 0, 0


# def bypass(image_src, visited, matches, back_color, x, y):
#     h, w = image_src.shape[:2]
#     if x < 0 or y < 0 or x >= w or y >= h:
#         return
#     if visited[y, x] == 1:
#         return
#     if image_src[y, x] != back_color:
#         visited[y, x] = 1
#         show_debug_image("Visited", visited, matches, (x, y), wait_time=1)
#         is_sub_img, theta, scale = find_elem(image_src, x, y)
#         if is_sub_img:
#             matches[y, x] = 1
#             if theta > 0:
#                 l_system1.append("+")
#             elif theta < 0:
#                 l_system1.append("-")
#             l_system1.append("X")
#             ######## TO DO : search in the frame ##########
#         for (dy, dx) in _POSSIBLE_MOVES:
#             bypass(image_src, visited, matches, back_color, x + dx, y + dy)


def valid_position(p, image, background_color, visited):
    y, x = p
    h, w = image.shape[:2]
    return 0 <= y < h and 0 <= x < w and visited[p] == 0 and image[p] != background_color


def search_matches(image_src, start, back_color):
    visited = np.zeros(image_src.shape[:2])
    match_mask = np.zeros(image_src.shape[:2])
    distances = -np.ones(image_src.shape[:2])
    matches = {}

    if not valid_position(start, image_src, back_color, visited):
        return

    q = deque()
    q.append(start)
    visited[start] = 1
    distances[start] = 0

    while q:
        y, x = q.popleft()
        show_debug_image("Visited", visited, match_mask, (x, y), wait_time=1)
        is_sub_img, theta, scale = find_elem(image_src, x, y)
        if is_sub_img:
            match_mask[y, x] = 1
            matches[(y,x)] = theta, scale

        neighbours = [(y + dy, x + dx) for dy, dx in _POSSIBLE_MOVES if valid_position((y + dy, x + dx), image_src, back_color, visited)]
        for p in neighbours:
            q.append(p)
            visited[p] = 1
            distances[p] = distances[y, x] + 1

    dbg_image = create_debug_image(visited, match_mask, (start[1], start[0]))
    return visited, match_mask, matches, dbg_image, distances


def find_start(image_src):
    h, w = image_src.shape[:2]
    last_row = image_src[-1, :]
    x = np.argmin(last_row)
    return h-1, x


def main():
    parser = argparse.ArgumentParser(description='Calculates L-system by fractal image.')
    parser.add_argument('input_image', type=str, help='Input image with fractal')
    args = parser.parse_args()

    image_src = load_image(args.input_image)
    start = find_start(image_src)
    visited, match_mask, matches, dbg_image, distances = search_matches(image_src, start, 255)

    with open("debug/visited.pickle", "wb") as f:
        pickle.dump(visited, f)
    with open("debug/match_mask.pickle", "wb") as f:
        pickle.dump(match_mask, f)
    with open("debug/matches.pickle", "wb") as f:
        pickle.dump(matches, f)
    with open("debug/distances.pickle", "wb") as f:
        pickle.dump(distances, f)

    cv2.imshow("Distances", np.float32(distances) / np.max(distances))
    cv2.imshow("Input", image_src)
    cv2.imshow("Visited", visited)
    cv2.imshow("Debug", dbg_image)
    cv2.imwrite("debug/match_map.png", dbg_image)
    cv2.waitKey()


if __name__ == "__main__":
    main()
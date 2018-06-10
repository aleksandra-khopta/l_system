#!/bin/python

import argparse

from collections import deque
from analyze_image import *
from match_structures import *
from image_similarity import is_sample_in_image

_POSSIBLE_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_RANGE_SCALE = [0.33, 0.5]


_FOUND_SCALE = []
_FOUND_D_THETA = []


def find_matches_at_point(image_src, x_start, y_start, threshold, angle_range):
    global _FOUND_SCALE
    global _FOUND_D_THETA
    matches = []
    bottom_center = x_start, y_start
    for scale in _FOUND_SCALE or _RANGE_SCALE:
        for theta in angle_range:
            for d_theta in _FOUND_D_THETA or np.arange(-2.0, 2.0, 1.0):
                sub_img = subimage_best(image_src, bottom_center, theta + d_theta, scale)
                score = is_sample_in_image(image_src, sub_img)
                if score > threshold:
                    matches.append(Match(theta + d_theta, scale, score))
                    if not _FOUND_SCALE:
                        _FOUND_SCALE = [scale]
                        _FOUND_D_THETA = [d_theta]
                        print("Scale = {}".format(scale))
                        print("Theta = {}".format(theta))
                        break
    return matches


def valid_position(p, image, background_color, visited):
    y, x = p
    h, w = image.shape[:2]
    return 0 <= y < h and 0 <= x < w and visited[p] == 0 and image[p] != background_color


def search_matches(image_src, start, back_color, similarity_threshold, angle_range, visual=False):
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
        if visual:
            show_debug_image("Visited", visited, match_mask, (x, y), wait_time=1)
        point_matches = find_matches_at_point(image_src, x, y, similarity_threshold, angle_range)
        if point_matches:
            match_mask[y, x] = 1
            matches[(y,x)] = point_matches

        neighbours = [(y + dy, x + dx) for dy, dx in _POSSIBLE_MOVES
                      if valid_position((y + dy, x + dx), image_src, back_color, visited)]
        for p in neighbours:
            q.append(p)
            visited[p] = 1
            distances[p] = distances[y, x] + 1

    dbg_image = create_debug_image(visited, match_mask, (start[1], start[0]))
    return MatchStructures(visited, match_mask, matches, distances), dbg_image


def find_start(image_src):
    h, w = image_src.shape[:2]
    last_row = image_src[-1, :]
    x = np.argmin(last_row)
    return h-1, x


def generate_angle_range(angle):
    angle_range = [0]
    current_angle = angle
    while current_angle < 90:
        angle_range.insert(0, -current_angle)
        angle_range.append(current_angle)
        current_angle += angle
    return angle_range


def process_image(input_image, background_color=255, similarity_threshold=0.7, dump_folder=None, visual=False):
    image_src = load_image(input_image)
    start = find_start(image_src)
    major_theta = calculate_major_angle(image_src)
    angle_range = generate_angle_range(major_theta)
    print(angle_range)
    match_structures, dbg_image = search_matches(image_src, start, background_color, similarity_threshold, angle_range, visual=visual)

    if dump_folder:
        dump_structures(dump_folder, match_structures)
    if visual:
        show_structures(match_structures)
        cv2.imshow("Debug", dbg_image)
        cv2.imwrite("debug/match_map.png", dbg_image)
        cv2.waitKey()

    return match_structures


def main():
    parser = argparse.ArgumentParser(description='Calculates L-system by fractal image.')
    parser.add_argument('input_image', type=str, help='Input image with fractal')
    parser.add_argument('--dump_folder', type=str, default="debug", help='Output folder for dumping matching structures')
    parser.add_argument('--background_color', type=int, default=255,
                        help='Background color of fractal image (1-channel value)')
    parser.add_argument('--similarity_threshold', type=float, default=0.8,
                        help='Minimum score for pattern to be matched')
    parser.add_argument('--visual', type=bool, default=False, help='Enable visualization')

    args = parser.parse_args()
    print(args)
    process_image(args.input_image, args.background_color, args.similarity_threshold, args.dump_folder, visual=args.visual)


if __name__ == "__main__":
    main()
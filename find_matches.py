import argparse

from collections import deque
from analyze_image import load_image, show_debug_image, subimage_best, equal, create_debug_image
from match_structures import *


_POSSIBLE_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_RANGE_THETA = [-25, 0, 25]
_RANGE_SCALE = [0.5]


def find_matches_at_point(image_src, x_start, y_start):
    matches = []
    bottom_center = x_start, y_start
    for theta in _RANGE_THETA:
        for scale in _RANGE_SCALE:
            sub_img = subimage_best(image_src, bottom_center, theta, scale)
            if equal(image_src, sub_img):
                matches.append(Match(theta, scale))
    return matches


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
        # is_sub_img, theta, scale = find_matches_at_point(image_src, x, y)
        point_matches = find_matches_at_point(image_src, x, y)
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


def main():
    parser = argparse.ArgumentParser(description='Calculates L-system by fractal image.')
    parser.add_argument('input_image', type=str, help='Input image with fractal')
    parser.add_argument('--dump_folder', type=str, default="debug", help='Output folder for dumping matching structures')
    args = parser.parse_args()

    image_src = load_image(args.input_image)
    start = find_start(image_src)
    # visited, match_mask, matches, dbg_image, distances = search_matches(image_src, start, 255)
    match_structures, dbg_image = search_matches(image_src, start, 255)

    dump_structures(args.dump_folder, match_structures)
    show_structures(match_structures)

    cv2.imshow("Debug", dbg_image)
    cv2.imwrite("debug/match_map.png", dbg_image)
    cv2.waitKey()


if __name__ == "__main__":
    main()
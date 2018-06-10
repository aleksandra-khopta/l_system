#!/bin/python

import argparse

from match_structures import *
from analyze_image import create_debug_image

from scipy.ndimage import label


def rule2_from_scale(scale):
    rule2_str = "F="
    for i in range(int(1 / scale)):
        rule2_str += "F"
    return rule2_str


def process_structures(match_structures):
    visited, match_mask, matches, distances = match_structures

    # show_structures(MatchStructures(visited, match_mask, matches, distances))
    # cv2.imshow("Debug", create_debug_image(visited, match_mask, (0, 0)))
    # cv2.waitKey()

    labaled_array, features_num = label(match_mask)

    filtered_matches = {}
    for label_index in range(1, features_num + 1):
        label_mask = labaled_array == label_index
        label_coordinates = np.asarray(np.column_stack(np.ma.where(label_mask)))
        scores = np.zeros_like(match_mask)
        for y, x in label_coordinates:
            current_matches = matches[(y, x)]
            scores[y, x] = np.sum([match.score for match in current_matches])
        max_score_position = np.unravel_index(np.argmax(scores), scores.shape)
        filtered_matches[max_score_position] = [matches[max_score_position], distances[max_score_position]]

    # print(filtered_matches)
    angle = np.max([match.theta for match in current_matches for current_matches in filtered_matches])
    scale = np.max([match.scale for match in current_matches for current_matches in filtered_matches])

    angle_str = str(angle)
    axiom_str = "X"
    rule1_str = "rule1"
    rule2_str = rule2_from_scale(scale)

    l_system_str = "{}\n{}\n{}\n{}".format(angle_str, axiom_str, rule1_str, rule2_str)
    return l_system_str


def main():
    parser = argparse.ArgumentParser(description='Calculates L-system by fractal image.')
    parser.add_argument('input_folder', type=str, help='Folder with dumped structures after searching matches')
    args = parser.parse_args()

    match_structures = load_structures(args.input_folder)
    l_system_str = process_structures(match_structures)
    print(l_system_str)


if __name__ == "__main__":
    main()
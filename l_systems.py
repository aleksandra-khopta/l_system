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


def put_in_braces(str):
    return "[" + str + "]"


def lsystem_to_str(lsystem):
    angle_str, axiom_str, rule1_str, rule2_str = lsystem
    l_system_str = "{}{}{}{}".format(angle_str, axiom_str, rule1_str, rule2_str)
    return l_system_str


def process_structures(match_structures, visual=False):
    visited, match_mask, matches, distances = match_structures

    if visual:
        show_structures(match_structures)
        cv2.imshow("Debug", create_debug_image(visited, match_mask, (0,0)))
        cv2.waitKey()

    dilation = cv2.dilate(match_mask, np.ones((5,5),np.uint8), iterations=1)
    labeled_array, features_num = label(dilation, structure=np.ones((3, 3)))
    labeled_array = labeled_array * match_mask
    if not features_num:
        return ""

    filtered_matches = {}
    for label_index in range(1, features_num + 1):
        label_mask = labeled_array == label_index
        label_coordinates = np.asarray(np.column_stack(np.ma.where(label_mask)))
        scores = np.zeros_like(match_mask)

        for y, x in label_coordinates:
            current_matches = matches[(y, x)]
            scores[y, x] = np.sum([match.score for match in current_matches])
        max_score_position = np.unravel_index(np.argmax(scores), scores.shape)
        filtered_matches[max_score_position] = [matches[max_score_position], distances[max_score_position]]

    sorted_match_list = [x[1] for x in sorted(filtered_matches.items(), key=lambda x: x[1][1])]
    # print(sorted_match_list)
    current_distance = 0

    all_detected_angles = []
    rule1_str = "X="
    for matches, distance in sorted_match_list:
        if distance > current_distance + 5:
            rule1_str += "F"
            current_distance = distance
        sorted_by_angle = sorted(matches, key=lambda m: -m.theta)
        for match in sorted_by_angle:
            theta = match.theta
            all_detected_angles.append(theta)
            d_theta = 2
            if theta < -d_theta:
                match_str = "+X"
            elif theta > d_theta:
                match_str = "-X"
            else:
                match_str = "X"
            rule1_str += put_in_braces(match_str)

    print(filtered_matches)
    print(all_detected_angles)

    angle = np.max(all_detected_angles)
    scale = np.max([match.scale for match in current_matches for current_matches in filtered_matches])

    angle_str = str(angle) + '\n'
    axiom_str = "X\n"
    rule1_str = rule1_str + '\n'
    rule2_str = rule2_from_scale(scale) + '\n'

    return [angle_str, axiom_str, rule1_str, rule2_str]


def main():
    parser = argparse.ArgumentParser(description='Calculates L-system by fractal image.')
    parser.add_argument('input_folder', type=str, help='Folder with dumped structures after searching matches')
    parser.add_argument('--visual', type=bool, default=False, help='Enable visualization')
    args = parser.parse_args()

    match_structures = load_structures(args.input_folder)
    angle_str, axiom_str, rule1_str, rule2_str = process_structures(match_structures, visual=args.visual)
    print(lsystem_to_str([angle_str, axiom_str, rule1_str, rule2_str]))


if __name__ == "__main__":
    main()
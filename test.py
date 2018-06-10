#!/bin/python

import find_matches
import l_systems


_TEST_IMAGES = ["images/fractal-1.png"]
_TEST_LSYSTEMS = ["gt/l-system-1.txt"]


def main():
    for image_file, gt_file in zip(_TEST_IMAGES, _TEST_LSYSTEMS):
        print(">>>>> Evaluate {} <<<<<".format(image_file))
        match_structures = find_matches.process_image(image_file)
        l_system_str = l_systems.process_structures(match_structures)
        gt_str = open(gt_file).read()
        print("-------------------------")
        print("Ground truth L-system:")
        print(gt_str)
        print("-------------------------")
        print("Calculated L-system:")
        print(l_system_str)
        print("-------------------------")
        print("Success = {}".format(gt_str == l_system_str))
        print("-------------------------")


if __name__ == "__main__":
    main()
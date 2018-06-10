#!/bin/python

import find_matches
import l_systems


_TEST_CASES = range(5)
_TEST_IMAGES = ["images/fractal{}.png".format(i) for i in _TEST_CASES]
_TEST_LSYSTEMS = ["gt/l-system-{}.txt".format(i) for i in _TEST_CASES]


def main():
    success_cases = 0
    total_cases = 0
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
        if gt_str == l_system_str:
            success_cases += 1
            print("*** Success ***")
        else:
            print("!!! Failure !!!")
        total_cases += 1
        print("-------------------------")

    print("============ Results =============")
    print("{} of {} test cases pass.".format(success_cases, total_cases))


if __name__ == "__main__":
    main()
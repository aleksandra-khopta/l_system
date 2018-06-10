#!/bin/python

import find_matches
import l_systems


_TEST_CASES = range(3)
# _TEST_CASES = [2]
_TEST_IMAGES = ["images/fractal{}.png".format(i) for i in _TEST_CASES]
_TEST_LSYSTEMS = ["gt/l-system-{}.txt".format(i) for i in _TEST_CASES]


def compare_systems(lsys1, lsys2):
    print(lsys1)
    print(lsys2)

    angle1 = float(lsys1[0])
    angle2 = float(lsys2[0])
    if abs(angle1 - angle2) > 2.0:
        return False
    return lsys1[1:] == lsys2[1:]


def load_lsystem(file):
    with open(file) as f:
        angle = f.readline()
        axiom = f.readline()
        rule1 = f.readline()
        rule2 = f.readline()
        return [angle, axiom, rule1, rule2]


def main():
    success_cases = 0
    total_cases = 0
    for tc, image_file, gt_file in zip(_TEST_CASES, _TEST_IMAGES, _TEST_LSYSTEMS):
        print(">>>>> Evaluate {} <<<<<".format(image_file))

        match_structures = find_matches.process_image(image_file, dump_folder="debug/t{}".format(tc))
        l_system = l_systems.process_structures(match_structures)
        gt_lsystem = load_lsystem(gt_file)

        print("-------------------------")
        print("Ground truth L-system:")
        print(l_systems.lsystem_to_str(gt_lsystem))
        print("-------------------------")
        print("Calculated L-system:")
        if l_system:
            print(l_systems.lsystem_to_str(l_system))
        print("-------------------------")

        if compare_systems(gt_lsystem, l_system):
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
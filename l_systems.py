import argparse

from match_structures import *
from analyze_image import create_debug_image


def main():
    parser = argparse.ArgumentParser(description='Calculates L-system by fractal image.')
    parser.add_argument('input_folder', type=str, help='Folder with dumped structures after searching matches')
    args = parser.parse_args()

    visited, match_mask, matches, distances = load_structures(args.input_folder)

    show_structures(MatchStructures(visited, match_mask, matches, distances))
    cv2.imshow("Debug", create_debug_image(visited, match_mask, (0, 0)))
    cv2.waitKey()


if __name__ == "__main__":
    main()
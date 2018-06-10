import argparse

from match_structures import *
from analyze_image import create_debug_image

from scipy.ndimage import label

def main():
    parser = argparse.ArgumentParser(description='Calculates L-system by fractal image.')
    parser.add_argument('input_folder', type=str, help='Folder with dumped structures after searching matches')
    args = parser.parse_args()

    visited, match_mask, matches, distances = load_structures(args.input_folder)

    # show_structures(MatchStructures(visited, match_mask, matches, distances))
    # cv2.imshow("Debug", create_debug_image(visited, match_mask, (0, 0)))
    # cv2.waitKey()

    labaled_array, features_num = label(match_mask)
    print(features_num)

    for label_index in range(1, features_num + 1):
        # find_label_maximum
        label_mask = labaled_array == label_index
        print(np.sum(label_mask))



if __name__ == "__main__":
    main()
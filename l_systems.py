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

    filtered_matches = {}
    for label_index in range(1, features_num + 1):
        label_mask = labaled_array == label_index
        print(np.sum(label_mask))

        label_coordinates = np.asarray(np.column_stack(np.ma.where(label_mask)))
        scores = np.zeros_like(match_mask)
        for y, x in label_coordinates:
            current_matches = matches[(y, x)]
            scores[y, x] = np.sum([match.score for match in current_matches])
        max_score_position = np.unravel_index(np.argmax(scores), scores.shape)
        print(max_score_position)
        filtered_matches[max_score_position] = [matches[max_score_position], distances[max_score_position]]

    print(filtered_matches)


if __name__ == "__main__":
    main()
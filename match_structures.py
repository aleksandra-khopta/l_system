import os
import cv2
import pickle
import numpy as np
from collections import namedtuple


Match = namedtuple("Match",["theta", "scale"])
MatchStructures = namedtuple("MatchStructures", ["visited", "match_mask", "matches", "distances"])


def dump_structures(folder, match_structures):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "visited.pickle"), "wb") as f:
        pickle.dump(match_structures.visited, f)
    with open(os.path.join(folder, "match_mask.pickle"), "wb") as f:
        pickle.dump(match_structures.match_mask, f)
    with open(os.path.join(folder, "matches.pickle"), "wb") as f:
        pickle.dump(match_structures.matches, f)
    with open(os.path.join(folder, "distances.pickle"), "wb") as f:
        pickle.dump(match_structures.distances, f)


def show_structures(match_structures):
    cv2.imshow("Distances", np.float32(match_structures.distances) / np.max(match_structures.distances))
    cv2.imshow("Visited", match_structures.visited)
    cv2.imshow("Match_mask", match_structures.match_mask)


def load_structures(folder):
    with open(os.path.join(folder, "visited.pickle"), "rb") as f:
        visited = pickle.load(f)
    with open(os.path.join(folder, "match_mask.pickle"), "rb") as f:
        match_mask = pickle.load(f)
    with open(os.path.join(folder, "matches.pickle"), "rb") as f:
        matches = pickle.load(f)
    with open(os.path.join(folder, "distances.pickle"), "rb") as f:
        distances = pickle.load(f)
    return MatchStructures(visited, match_mask, matches, distances)

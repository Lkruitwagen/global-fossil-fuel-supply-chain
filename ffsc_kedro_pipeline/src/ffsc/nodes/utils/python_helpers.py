from typing import List
import itertools
from shapely import geometry


def flatten_list_of_lists(lol: List) -> List:
    """
    Flattens a python list of lists by one level.
    [[1,2], [3,4]] becomes [1, 2, 3, 4]
    [[[1,2], [3,4]], [[5, 6], [7, 8]]] becomes [[1, 2], [3, 4], [5, 6], [7, 8]]
    """
    flattened = []
    for l in lol:
        flattened.extend(l)
    return flattened


def unique_nodes_from_segments(segment_geometries: List[geometry.LineString]) -> List:
    """Extracts the unique node coordinates from a list of line segments"""
    coordinates = [list(line.coords) for line in segment_geometries]
    flattened_coordinates = flatten_list_of_lists(coordinates)
    flattened_coordinates.sort()
    return list(k for k, _ in itertools.groupby(flattened_coordinates))


def convert_segments_to_lines(segments: List) -> List:
    """Converts a list of segments into a list of individual lines,
    by assuming the items next to each other in the segment are connected"""
    line_strings = []
    for segment in segments:
        line_strings.extend(
            [([segment[i], segment[i + 1]]) for i in range(len(segment) - 1)]
        )

    return line_strings

import numpy as np


def select(heatmaps: np.ndarray, n: int) -> "selected bands":

    """

    :param heatmaps: numpy.array class-wise heatmaps
    :param n: number of bands to be selected
    :return: numpy.array of selected bands
    """

    sortedIndex = {}
    numOfClasses = len(heatmaps)
    for i in range(numOfClasses):
        sortedIndex[i] = sorted(range(len(heatmaps[i])), key=lambda x: heatmaps[i][x], reverse=True)
    selected = []
    # bandsPerHeatmaps = [round(len(heatmaps) / n) * (i + 1) for i in range(numOfClasses)]
    bandsPerHeatmaps = np.array([round(n / numOfClasses * (i + 1)) for i in range(numOfClasses)])
    padded = np.roll(bandsPerHeatmaps, 1)
    padded[0] = 0
    bandsPerHeatmaps = bandsPerHeatmaps - padded

    for i in range(numOfClasses):

        heatmapIndex = sortedIndex[i]
        numSelected = 0
        index = 0
        while numSelected < bandsPerHeatmaps[i]:

            if heatmapIndex[index] not in selected:
                selected.append(heatmapIndex[index])
                numSelected += 1

            index += 1

    return selected

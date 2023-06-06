import numpy as np
from scipy import stats


def CPC18_getDist(H, pH, L, lot_shape, lot_num):
    # Extract true full distributions of an option in CPC18
    #   input is high outcome (H: int), its probability (pH: double), low outcome
    #   (L: int), the shape of the lottery ('-'/'Symm'/'L-skew'/'R-skew' only), and
    #   the number of outcomes in the lottery (lot_num: int)
    #   output is a matrix (numpy matrix) with first column a list of outcomes (sorted
    #   ascending) and the second column their respective probabilities.

    if lot_shape == '-':
        if pH == 1:
            dist = np.array([H, pH])
            dist.shape = (1, 2)
        else:
            dist = np.array([[L, 1-pH], [H, pH]])

    else:  # H is multi outcome
        # compute H distribution
        high_dist = np.zeros(shape=(lot_num, 2))
        if lot_shape == 'Symm':
            k = lot_num - 1
            for i in range(0, lot_num):
                high_dist[i, 0] = H - k / 2 + i
                high_dist[i, 1] = pH * stats.binom.pmf(i, k, 0.5)

        elif (lot_shape == 'R-skew') or (lot_shape == 'L-skew'):
            if lot_shape == 'R-skew':
                c = -1 - lot_num
                dist_sign = 1
            else:
                c = 1 + lot_num
                dist_sign = -1
            for i in range(1, lot_num+1):
                high_dist[i - 1, 0] = H + c + dist_sign * pow(2, i)
                high_dist[i - 1, 1] = pH / pow(2, i)

            high_dist[lot_num - 1, 1] = high_dist[lot_num - 1, 1] * 2

        # incorporate L into the distribution
        dist = high_dist
        locb = np.where(high_dist[:, 0] == L)
        if all(locb):
            dist[locb, 1] = dist[locb, 1] + (1-pH)
        elif pH < 1:
            dist = np.vstack((dist, [L, 1-pH]))

        dist = dist[np.argsort(dist[:, 0])]

    return dist

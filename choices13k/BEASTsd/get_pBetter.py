import numpy as np
from distSample import distSample


def get_pBetter(DistX, DistY, corr, accuracy=10000):
    # Return probability that a value drawn from DistX is strictly larger than one drawn from DistY
    # Input: 2 discrete distributions which are set as matrices of 1st column
    # as outcome and 2nd its probability. DistX and DistY are numpy matrices; correlation between the distributions;
    # level of accuracy in terms of number of samples to take from distributions
    # Output: a list with the estimated probability that X generates value strictly larger than Y, and
    # the probability that Y generates value strictly larger than X

    nXbetter = 0
    nYbetter = 0

    for j in range(1, accuracy+1):
        rndNum = np.random.uniform(size=2)
        sampleX = distSample(DistX[:, 0], DistX[:, 1], rndNum[0])
        if corr == 1:
            sampleY = distSample(DistY[:, 0], DistY[:, 1], rndNum[0])
        elif corr == -1:
            sampleY = distSample(DistY[:, 0], DistY[:, 1], 1-rndNum[0])
        else:
            sampleY = distSample(DistY[:, 0], DistY[:, 1], rndNum[1])
        nXbetter = nXbetter + int(sampleX > sampleY)
        nYbetter = nYbetter + int(sampleY > sampleX)

    pXbetter = nXbetter / accuracy
    pYbetter = nYbetter / accuracy

    return [pXbetter, pYbetter]

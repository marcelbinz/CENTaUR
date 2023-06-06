import numpy as np
from distSample import distSample


def CPC18_BEASTsd_simulation(DistA, DistB, Amb, Corr, probsBetter):
    # Simulating one virtual agent of the type BEAST.sd for one problem defined by the input
    #   Input: Payoff distributions of Option A and Option B respectively, each as a
    #   matrix of outcomes and their respective probabilities; whether ambiguity and correlation
    #   between the outcomes exists, and the probabilities that one option provides a greater
    #   payoff than the other.
    #   DistA and DistB are numpy matrices
    #   Output: the mean choice rate of option B for the current virtual agent, in blocks of 5.
    #   numpy array of size: (nBlocks, 1)

    # Model free parameters

    SIGMA = 13
    KAPA = 3
    BETA = 1.4
    GAMA = 1
    PSI = 0.25
    THETA = 0.7
    SIGMA_COMP = 35
    WAMB = 0.25

    # Setting's constants used
    nTrials = 25
    firstFeedback = 6
    nBlocks = 5

    # Useful variables
    nA = DistA.shape[0]  # num outcomes in A
    nB = DistB.shape[0]  # num outcomes in B
    if Amb == 1:
        ambiguous = True
    else:
        ambiguous = False

    # Initialize variables
    pBias = np.repeat([0.0], nTrials - firstFeedback + 1 + 1)  # probability of choosing biased simulation tool
    ObsPay = np.zeros(shape=(nTrials - firstFeedback + 1, 2))  # observed outcomes in A (col1) and B (col2)
    Decision = np.empty(shape=(nTrials, 1), dtype=bool)
    simPred = np.repeat([0.0], nBlocks)

    # check for complexity of problem
    if (max(nA, nB) > 2) and (min(nA, nB) > 1):
        SIG = SIGMA_COMP
    else:
        SIG = SIGMA

    # draw personal traits
    sigma = SIG * np.random.uniform(size=1)
    kapa = np.random.choice(range(1, KAPA+1), 1)
    beta = BETA * np.random.uniform(size=1)
    gama = GAMA * np.random.uniform(size=1)
    psi = PSI * np.random.uniform(size=1)
    theta = THETA * np.random.uniform(size=1)
    wamb = WAMB * np.random.uniform(size=1)

    # More useful variables
    nfeed = 0  # "t"; number of outcomes with feedback so far
    pBias[nfeed] = beta / (beta + 1 + pow(nfeed, theta))
    MinA = DistA[0, 0]
    MinB = DistB[0, 0]
    MaxOutcome = np.maximum(DistA[nA - 1, 0], DistB[nB - 1, 0])
    SignMax = np.sign(MaxOutcome)

    # Compute "RatioMin"
    if MinA == MinB:
        RatioMin = 1
    elif np.sign(MinA) == np.sign(MinB):
        RatioMin = min(abs(MinA), abs(MinB)) / max(abs(MinA), abs(MinB))
    else:
        RatioMin = 0

    nAwin = 0  # number of times Option A's observed payoff was at least as high as B's
    nBwin = 0  # number of times Option B's observed payoff was at least as high as A's
    sumPayB = 0  # sum of payoffs in Option B (used if B is abmiguous)
    Range = MaxOutcome - min(MinA, MinB)

    UEVa = np.matrix.dot(DistA[:, 0], np.repeat([1 / nA], nA))  # EV of A had all its payoffs been equally likely
    UEVb = np.matrix.dot(DistB[:, 0], np.repeat([1 / nB], nB))  # EV of B had all its payoffs been equally likely
    BEVa = np.matrix.dot(DistA[:, 0], DistA[:, 1])  # Best estimate of EV of Option
    if ambiguous:
        BEVb = (1-psi) * (UEVb+BEVa) / 2 + psi * MinB
        pEstB = np.repeat([float(nB)], 1)  # estimation of probabilties in Amb
        t_SPminb = (BEVb - np.mean(DistB[1:nB+1, 0])) / (MinB - np.mean(DistB[1:nB+1, 0]))
        if t_SPminb < 0:
            pEstB[0] = 0
        elif t_SPminb > 1:
            pEstB[0] = 1
        else:
            pEstB[0] = t_SPminb

        # Add nb-1 rows to pEstB:
        pEstB = np.append(pEstB, np.repeat([(1 - pEstB[0]) / (nB - 1)], nB-1))

    else:
        pEstB = DistB[:, 1]
        BEVb = np.matrix.dot(DistB[:, 0], pEstB)

    # compute subjective dominance for this problem
    subjDom = 0
    if not ambiguous:
        pAbetter = probsBetter[0]
        pBbetter = probsBetter[1]
        if (BEVa > BEVb) and (UEVa >= UEVb):
            subjDom = 1-pBbetter
        elif (BEVa < BEVb) and (UEVa <= UEVb):
            subjDom = 1-pAbetter

    if (MinA > DistB[nB - 1, 0]) or (MinB > DistA[nA - 1, 0]):
        subjDom = 1

    # correct error rate as per subjective dominance component
    sigma = sigma * (1 - subjDom)
    sigmat = sigma

    # simulation of the 25 decisions

    # simulation of decisions
    for trial in range(0, nTrials):
        STa = 0
        STb = 0
        # mental simulations
        for s in range(1, kapa[0]+1):
            rndNum = np.random.uniform(size=2)
            # Unbiased tool
            if rndNum[0] > pBias[nfeed]:
                if nfeed == 0:
                    outcomeA = distSample(DistA[:, 0], DistA[:, 1], rndNum[1])
                    outcomeB = distSample(DistB[:, 0], pEstB, rndNum[1])
                else:
                    uniprobs = np.repeat([1 / nfeed], nfeed)
                    outcomeA = distSample(ObsPay[0:nfeed, 0], uniprobs, rndNum[1])
                    outcomeB = distSample(ObsPay[0:nfeed, 1], uniprobs, rndNum[1])

            elif rndNum[0] > (2 / 3) * pBias[nfeed]:  # Uniform tool
                outcomeA = distSample(DistA[:, 0], np.repeat([1 / nA], nA), rndNum[1])
                outcomeB = distSample(DistB[:, 0], np.repeat([1 / nB], nB), rndNum[1])
            elif rndNum[0] > (1 / 3) * pBias[nfeed]:  # contingent pessimism tool
                if SignMax > 0 and RatioMin < gama:
                    outcomeA = MinA
                    outcomeB = MinB
                else:
                    outcomeA = distSample(DistA[:, 0], np.repeat([1 / nA], nA), rndNum[1])
                    outcomeB = distSample(DistB[:, 0], np.repeat([1 / nB], nB), rndNum[1])

            else:  # Sign tool
                if nfeed == 0:
                    outcomeA = Range * distSample(np.sign(DistA[:, 0]), DistA[:, 1], rndNum[1])
                    outcomeB = Range * distSample(np.sign(DistB[:, 0]), pEstB, rndNum[1])
                else:
                    uniprobs = np.repeat(1 / nfeed, nfeed)
                    outcomeA = Range * distSample(np.sign(ObsPay[0:nfeed, 0]), uniprobs, rndNum[1])
                    outcomeB = Range * distSample(np.sign(ObsPay[0:nfeed, 1]), uniprobs, rndNum[1])

            STa = STa + outcomeA
            STb = STb + outcomeB

        STa = STa / kapa
        STb = STb / kapa

        # error term
        error = sigmat * np.random.normal(size=1)  # positive values contribute to attraction to A

        # decision
        Decision[trial] = (BEVa - BEVb) + (STa - STb) + error < 0
        if (BEVa - BEVb) + (STa - STb) + error == 0:
            Decision[trial] = np.random.choice(range(1, 3), size=1, replace=False) - 1

        if trial >= firstFeedback - 1:
            # Handle feedback if necessary
            nfeed += 1
            pBias[nfeed] = beta / (beta + 1 + pow(nfeed, theta))
            rndNumObs = np.random.uniform(size=1)
            ObsPay[nfeed - 1, 0] = distSample(DistA[:, 0], DistA[:, 1], rndNumObs)  # draw outcome from A
            # draw outcome from B
            if Corr == 1:
                ObsPay[nfeed - 1, 1] = distSample(DistB[:, 0], DistB[:, 1], rndNumObs)
            elif Corr == -1:
                ObsPay[nfeed - 1, 1] = distSample(DistB[:, 0], DistB[:, 1], 1-rndNumObs)
            else:
                # draw outcome from B
                ObsPay[nfeed - 1, 1] = distSample(DistB[:, 0], DistB[:, 1], np.random.uniform(size=1))

            # update number of A or B "wins"
            nAwin = nAwin + (ObsPay[nfeed - 1, 0] >= ObsPay[nfeed - 1, 1])
            nBwin = nBwin + (ObsPay[nfeed - 1, 1] >= ObsPay[nfeed - 1, 0])
            sumPayB += ObsPay[nfeed - 1, 1]
            if ambiguous:
                BEVb = (1 - wamb) * BEVb + wamb * ObsPay[nfeed - 1, 1]  # update best estimate of B's EV
                avgPayB = sumPayB / nfeed
                # update size of error in ambiguous problems
                if subjDom != 1:
                    if (BEVa > avgPayB) and (UEVa >= UEVb):
                        sigmat *= 1 - nAwin / nfeed
                    elif (BEVa < avgPayB) and (UEVa <= UEVb):
                        sigmat *= 1 - nBwin / nfeed

    # compute B-rates for this simulation
    blockSize = nTrials / nBlocks
    for b in range(1, nBlocks+1):
        simPred[b-1] = np.mean(Decision[int(((b - 1) * blockSize + 1)-1):int(b * blockSize)])

    return simPred

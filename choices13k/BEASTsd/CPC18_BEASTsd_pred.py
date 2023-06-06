import numpy as np
from CPC18_getDist import CPC18_getDist
from get_pBetter import get_pBetter
from CPC18_BEASTsd_simulation import CPC18_BEASTsd_simulation


def CPC18_BEASTsd_pred(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr):
    # Prediction of BEAST.sd model for one problem
    #
    #  This function gets as input 12 parameters which define a problem in CPC18
    #  and outputs BEAST.sd model's prediction in that problem for five blocks of
    #  five trials each (the first is without and the others are with feedback
    # Input: for A and B: high outcome (Ha/ Hb: int), its probability (pHa/ pHb: double), low outcome
    #  (La/ Lb: int), the shape of the lottery (LotShapeA/ LotShapeB that can be:'-'/'Symm'/'L-skew'/'R-skew' only),
    #  the number of outcomes in the lottery (lot_numA/ LotNumB: int),
    #  Amb (1 or 0) indicates if there exists ambiguity
    #  Corr is thw correlation between A and B (-1 0 or 1).
    # Output: the prediction of the BEAST.sd model: this is a numpy of size (5,1)

    Prediction = np.repeat([0], 5)
    Prediction.shape = (1, 5)

    # get both options' detailed distributions
    DistA = CPC18_getDist(Ha, pHa, La, LotShapeA, LotNumA)
    DistB = CPC18_getDist(Hb, pHb, Lb, LotShapeB, LotNumB)

    # get the probabilities that each option gives greater value than the other
    probsBetter = get_pBetter(DistA, DistB, corr=1, accuracy=100000)

    # run model simulation nSims times
    nSims = 5000
    for sim in range(0, nSims):
        simPred = CPC18_BEASTsd_simulation(DistA, DistB, Amb, Corr, probsBetter)
        Prediction = np.add(Prediction, (1 / nSims) * simPred)

    return Prediction

import os
import pandas as pd
import numpy as np
import time
from CPC18_BEASTsd_pred import CPC18_BEASTsd_pred
import torch
from tqdm import tqdm

if __name__ == '__main__':
    df = pd.read_csv('../data/c13k_selections.csv')
    df['LotShapeB'] = df['LotShapeB'].replace({
        0: '-',
        1: 'Symm',
        2: 'R-skew',
        3: 'L-skew',
    })

    choice_probs = []
    num_choices = []
    num_B_choices = []

    for index in tqdm(range(len(df))):
        df_row = df.iloc[index]
        if df_row.Feedback and not df_row.Amb:
            Ha = df_row['Ha']
            pHa = df_row['pHa']
            La = df_row['La']
            LotShapeA = '-'
            LotNumA = 1
            Hb = df_row['Hb']
            pHb = df_row['pHb']
            Lb = df_row['Lb']
            LotShapeB = df_row['LotShapeB']
            LotNumB = df_row['LotNumB']
            Amb = df_row['Amb']
            Corr = df_row['Corr']

            Prediction = CPC18_BEASTsd_pred(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr)

            choice_probs.append(Prediction[:, df_row['Block']-1].item())
            num_choices.append(df_row.n * 5)
            num_B_choices.append(df_row.n * 5 * df_row.bRate)

    choice_probs = torch.Tensor(choice_probs)[:9800]
    num_choices = torch.Tensor(num_choices)[:9800]
    num_B_choices = torch.Tensor(num_B_choices)[:9800]

    torch.save([choice_probs, num_choices, num_B_choices], '../data/model=BEAST_full_data_choices13k.pth')

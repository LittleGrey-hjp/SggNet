"""
Used to calculate the PR curve of camouflaged object detection
Torch-based acceleration
"""
import os
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

# replace the prediction paths to yours
result_path = ['CSNet', 'CorrNet', 'FSMINet', 'HVPNet', 'MEAN', 'SAMNet', 'SeaNet', 'SOLNet', 'MSCNet','Ours']

device = "cpu"

for _data_name in ['EORSSD']:
    data = {}
    print("eval-dataset: {}".format(_data_name))
    for result in result_path:
        mask_root = r'F:\SOD_dataset/{}/{}/'.format(_data_name, "test-labels") # change path
        pred_root = r'F:\SOD/{}/{}/'.format(result, _data_name)
        print('eval PR of :', pred_root)
        pred_name_list = sorted(os.listdir(pred_root))

        num = len(pred_name_list)
        Precision = torch.zeros((256, num), device=device)
        Recall = torch.zeros((256, num), device=device)
        TP = torch.zeros((256, num), device=device)
        FP = torch.zeros((256, num), device=device)
        FN = torch.zeros((256, num), device=device)

        for j, pred_name in enumerate(tqdm(pred_name_list, total=len(pred_name_list))):
            mask_path = os.path.join(mask_root, pred_name)
            pred_path = os.path.join(pred_root, pred_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

            pred = cv2.resize(pred, (mask.shape[1], mask.shape[0]))

            mask = torch.tensor(mask, device=device)
            pred = torch.tensor(pred, device=device)

            mask[mask > 0] = 1  # 二值化

            for i in range(256):
                output0 = torch.where(pred > i, 1, 0)
                output1 = output0 * 2  # 排除TN
                TFNP = output1 - mask
                TP[i, j] = torch.sum(TFNP == 1)
                FP[i, j] = torch.sum(TFNP == 2)
                FN[i, j] = torch.sum(TFNP == -1)

                Precision[i, j] = TP[i, j] / (TP[i, j] + FP[i, j]) if (TP[i, j] + FP[i, j]) > 0 else 0
                Recall[i, j] = TP[i, j] / (TP[i, j] + FN[i, j]) if (TP[i, j] + FN[i, j]) > 0 else 0

        # 计算平均精度和召回率
        P = Precision.mean(dim=1).cpu().numpy()
        R = Recall.mean(dim=1).cpu().numpy()

        data['R_{}'.format(result)] = R
        data['P_{}'.format(result)] = P

    df = pd.DataFrame(data)
    df.to_excel(r'F:\SOD\PR_Curve\PR_{}.xlsx'.format(result), index=False, engine='openpyxl')   # replace the .xlsx path to yours

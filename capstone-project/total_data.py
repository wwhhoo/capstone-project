import numpy as np


path = './baseline/data/sorted'

file = ['ori_text','imageID_order', 'textID_order', 'dataset_image', 'dataset_text', 'dataset_category']

for i in range (6):
    if i>=4:
        AP = True
    else:
        AP = False
    print(file[i],':')
    a = np.load(f'{path}/train/train_{file[i]}.npy',allow_pickle=AP)
    print(a.shape)
    # a = np.append(a,np.load(f'{path}/validation/validation_{file[i]}.npy',allow_pickle=AP),axis=0)
    # print(a.shape)
    a = np.append(a,np.load(f'{path}/test/test_{file[i]}.npy',allow_pickle=AP),axis=0)
    print(a.shape)

    np.save(f"{path}/Total/Total_{file[i]}.npy",a)

from lib.coreFunc import evaluate
from lib.my_dataset import TextDataset, ImageDataset, ClassifierDataset, create_mini_batch
from lib.my_model import Propose_model
from torch.utils.data import DataLoader
import argparse
import torch
import os
import pandas as pd


data_path = './baseline/data/sorted'
BATCH_SIZE = 128

# mode = "test_dataset"
file = "test"
# mode = "train"
Bert_test_data = TextDataset(data_path,file,'test')
Bert_testloader = DataLoader(Bert_test_data, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch)
file = "test_dataset"
# mode = "train_dataset"
# mode = "train"
Resnet_test_data = ImageDataset(data_path,file,'test')
Resnet_testloader = DataLoader(Resnet_test_data, batch_size=BATCH_SIZE)

Category_test_data = ClassifierDataset(data_path,file,"test")
Category_testloader = DataLoader(data_path, batch_size=BATCH_SIZE)

file = "Total"
# mode = "test"
Bert_Total_data = TextDataset(data_path,file,'test')
Bert_Totalloader = DataLoader(Bert_Total_data, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch)
file = "Total_dataset"
# mode = "test_dataset"
Resnet_Total_data = ImageDataset(data_path,file,'test')
Resnet_Totalloader = DataLoader(Resnet_Total_data, batch_size=BATCH_SIZE)

#assign model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
print('========================================')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
My_model = Propose_model()
My_model = My_model.to(device)

print('========================================')

# mode = "image_to_text"
# mode = "text_to_image"
# mode = "text_to_text"
mode = "image_to_image"
print(mode)

result = evaluate(Bert_testloader = Bert_testloader, Resnet_testloader = Resnet_testloader,
                    Bert_Totalloader = Bert_Totalloader, Resnet_Totalloader = Resnet_Totalloader,
                    My_model = My_model, device = device, mode = mode)



print('========================================')
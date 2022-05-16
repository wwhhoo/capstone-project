from torch.nn.utils.rnn import pad_sequence
from random import randint
import cv2
import torch
import numpy as np
import json
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset

class ImageTranform(torch.nn.Module):
    def __init__(self, image_size=(448,448)):
        super(ImageTranform,self).__init__()
        self.resize = T.Resize(image_size)
        self.centercrop = T.CenterCrop(448)
        self.randomcrop = T.RandomCrop(392)
        self.randomhorizontalflip = T.RandomHorizontalFlip(p=0.5)
        self.to_tensor = T.ToTensor()
        
    def forward(self, image):
        image = image.convert('RGB')
        image = self.resize(image)
        image = self.centercrop(image)
        image = self.randomcrop(image)
        image = self.randomhorizontalflip(image)
        image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        image = Image.fromarray(image)
        image = self.to_tensor(image)
        image = torch.cat((image,image,image),0)

        return image

class ImageTestTranform(torch.nn.Module):
    def __init__(self, image_size=(448,448)):
        super(ImageTestTranform,self).__init__()
        self.resize = T.Resize(image_size)
        self.to_tensor = T.ToTensor()
        
    def forward(self, image):
        image = image.convert('RGB')
        image = self.resize(image)
        image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)#cv2.CV_8U,
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        image = Image.fromarray(image)
        image = self.to_tensor(image)
        image = torch.cat((image,image,image),0)

        return image

class TextDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, path, file, mode):
        self.mode = mode
        # self.file = file
        if mode == "test" or mode == "mse" or mode == "validation":
            self.data  = np.load(f'{path}/{file}/{file}_dataset_text.npy',allow_pickle=True)
        else:    
            self.data  = np.load(f'{path}/{file}/{file}_text.npy',allow_pickle=True)
            self.dataset = np.load(f'{path}/{file}/{file}_dataset_text.npy',allow_pickle=True)
            self.imageID = np.load(f'{path}/{file}/{file}_imageID_order.npy')
            self.class_pos = np.load(f'{path}/{file}/{file}_class_num.npy')
            dic = "./baseline/data/rough_detail_class.json"
            self.rough_detail_class = json.load(open(dic))
            dic = "./baseline/data/dic_for_class.json"
            self.id_to_class = json.load(open(dic))
        
    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        if self.mode == 'positive':
            if idx % 3 == 0:
                tokens_tensor = self.dataset[idx//3]
            else:
                class_num = self.rough_detail_class[self.id_to_class[self.imageID[idx//3].split('-')[0]]]
                if class_num == 0:
                    ID = randint(0 ,self.class_pos[class_num]-1)
                else:
                    ID = randint(self.class_pos[class_num-1], self.class_pos[class_num]-1)
                tokens_tensor = self.data[ID]
        elif self.mode == 'negative':
            while True:
                ID = randint(0, len(self.dataset)-1)
                # 158
                if self.id_to_class[self.imageID[ID].split('-')[0]] != self.id_to_class[self.imageID[idx//3].split('-')[0]]:
                    tokens_tensor = self.dataset[ID]
                    break
                else:
                    continue

                # 32
                # if self.id_to_class[self.imageID[ID].split('-')[0]][:2] != self.id_to_class[self.imageID[idx//5].split('-')[0]][:2]:
                #     tokens_tensor = self.dataset[ID]
                #     break
                # else:
                #     continue
        elif self.mode == 'mse':
            tokens_tensor = self.data[idx//3]
        elif self.mode == 'test' or self.mode == 'validation':
            tokens_tensor = self.data[idx]
            
        tokens_tensor = torch.tensor(tokens_tensor)
        len_text = len(tokens_tensor)
        segments_tensor = torch.tensor([0] * len_text, dtype=torch.long)

        return (tokens_tensor, segments_tensor)
    
    def __len__(self):
        if self.mode == 'test' or self.mode == 'validation':
            return len(self.data)
        else:    
            return len(self.data)*3
    
# 初始化一個專門讀取訓練樣本的 Dataset，使用中文 BERT 斷詞
class ImageDataset(Dataset):

    def __init__(self, path, file, mode, transform = ImageTranform(), test_transform = ImageTestTranform()):
        self.mode = mode
        self.transformer = transform
        self.test_transformer = test_transform
        if mode == "pretrain":
            self.data      = np.load(f'{path}/{file}/{file}_image_sorted.npy')
            self.dataset   = np.load(f'{path}/{file}/{file}_dataset_image.npy')
            self.class_pos = np.load(f'{path}/{file}/{file}_class_num.npy')
            self.imageID   = np.load(f'{path}/{file}/{file}_imageID_order.npy')
            dic = "./baseline/data/rough_detail_class.json"
            self.rough_detail_class = json.load(open(dic))
            dic = "./baseline/data/dic_for_class.json"
            self.id_to_class = json.load(open(dic))
        else:
            self.data = np.load(f'{path}/{file}/{file}_image.npy')

    def __getitem__(self,idx):
        pos_tensor = []
        neg_tensor = []
        
        if self.mode == 'pretrain':
            tokens_tensor_ID = self.dataset[idx//3]
            tokens_tensor = Image.open(tokens_tensor_ID)
            tokens_tensor = self.transformer(tokens_tensor)
            # positive
            class_num = self.rough_detail_class[self.id_to_class[self.imageID[idx//3].split('-')[0]]]
            only_one = False
            while True:
                if class_num == 0:
                    ID = randint(0 ,self.class_pos[class_num]-1)
                    if (self.class_pos[class_num]-1 == 0):
                        only_one = True
                else:
                    ID = randint(self.class_pos[class_num-1], self.class_pos[class_num]-1)
                    if (self.class_pos[class_num-1] == self.class_pos[class_num]-1):
                        only_one = True
                # if idx % 3 == 0:
                #     tokens_tensor = self.dataset[idx//3]
                # else:
                #     class_num = self.rough_detail_class[self.id_to_class[self.imageID[idx//3].split('-')[0]]]
                #     if class_num == 0:
                #         ID = randint(0 ,self.class_pos[class_num]-1)
                #     else:
                #         ID = randint(self.class_pos[class_num-1], self.class_pos[class_num]-1)
                #     tokens_tensor = self.data[ID]
                # avoid same as anchor
                pos_tensor_ID = self.data[ID]
                if (pos_tensor_ID != tokens_tensor_ID) or only_one:
                    pos_tensor = Image.open(pos_tensor_ID)
                    pos_tensor = self.transformer(pos_tensor)
                    break

            # negative
            while True:
                ID = randint(0, len(self.dataset)-1)
                # 158
                #self.id_to_class[self.imageID[ID].split('-')[0]]
                if self.id_to_class[self.imageID[ID].split('-')[0]] != self.id_to_class[self.imageID[idx//3].split('-')[0]]:
                    neg_tensor = self.dataset[ID]
                    neg_tensor = Image.open(neg_tensor)
                    neg_tensor = self.transformer(neg_tensor)
                    break
                else:
                    continue
                # 32
                # if self.id_to_class[self.imageID[ID].split('-')[0]][:2] != self.id_to_class[self.imageID[idx//5].split('-')[0]][:2]:
                #     tokens_tensor = self.dataset[ID]
                #     break
                # else:
                #     continue
            return tokens_tensor, pos_tensor, neg_tensor

        
        if  self.mode == 'train':
            tokens_tensor = self.data[idx//3]
            tokens_tensor = Image.open(tokens_tensor)
            tokens_tensor = self.transformer(tokens_tensor)
            return tokens_tensor

        elif self.mode == 'validation' or self.mode == 'test':
            tokens_tensor = self.data[idx]
            tokens_tensor = Image.open(tokens_tensor)
            tokens_tensor = self.test_transformer(tokens_tensor)
            return tokens_tensor
    
    def __len__(self):
        if self.mode == 'test' or self.mode =="validation":
            return len(self.data)
        else:    
            return len(self.data)*3

class ClassifierDataset(Dataset):

    def __init__(self, path, file, mode):

        self.data = np.load(f'{path}/{file}/{file}_category.npy')
        self.mode = mode
    def __getitem__(self,idx):
        if self.mode == 'test' or self.mode =="validation":
            class_tensor = self.data[idx]
        else:
            class_tensor = self.data[idx//3]
        class_tensor = torch.from_numpy(np.asarray(class_tensor))
        class_tensor = class_tensor.type(torch.long)

        return class_tensor
    def __len__(self):
        if self.mode == 'test' or self.mode =="validation":
            return len(self.data)
        else:      
            return len(self.data)*3

def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    # zero pad
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, 
                                    batch_first=True)
    
    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape,
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors
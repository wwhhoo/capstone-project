import numpy as np
from PIL import Image, ImageOps

import torch
import json
from torchvision.transforms.functional import to_tensor
from transformers import BertTokenizer
from torchvision.ops.boxes import box_convert
from torchvision.transforms import Resize
from random import randint

dict_test = {}

def match_training_data(paths, save_path, word_pieces, tokenizer, file, file_name, MAX_LENGTH):

    # num = 0
    df_data = np.load(f"{paths}/Info/{file}/{file_name[9]}.npy")
    df_name = np.load(f"{paths}/Info/{file}/{file_name[2]}.npy")
    with open(f"{paths}/annotations/annotations-{file}.json", 'r') as f:
        annFile = json.loads(f.read())
    dic = "./baseline/data/rough_detail_class.json"
    rough_detail_class = json.load(open(dic))
    dic = "./baseline/data/detail_rough_class.json"
    detail_rough_class = json.load(open(dic))
    dic = "./baseline/data/dic_for_class.json"
    id_to_class = json.load(open(dic)) 
    anns = annFile['annotation']
    
    category_dataset = []
    image_dataset = []
    text_dataset = []
    image_sorted = []
    short_data = []
    image_data = []
    text_data = []
    image_ID = []
    text_ID = []
    category = []
    count = 0
    check = 0
    
    class_num = np.zeros(158)
    # class_num = np.zeros(32)
    
    image_id = anns[0]['image_id']
    exclude_cat = [1]
    print(len(anns))
    for i in range(len(anns)):
        # text
        if anns[i]['category_id'] not in exclude_cat:
            if (anns[i]['category_id'] == 0) and ((anns[i]['image_id'].split('-')[0] != image_id.split('-')[0]) or (check == 0)):
                dict_test[anns[i]['image_id'].split('-')[0]] = 0
                df_name[count] =  np.char.add(df_name[count], "。")
                df_data[count] =  np.char.add(df_name[count], df_data[count])
                df_data[count] =  np.char.add(word_pieces, df_data[count])
                if len(df_data[count]) > MAX_LENGTH:
                    df_data[count] = df_data[count][:MAX_LENGTH]
                
                # num += len(df_data[count])
                tokens = tokenizer.tokenize(str(df_data[count]))
                ids = tokenizer.convert_tokens_to_ids(tokens)
                count += 1
                image_id = anns[i]['image_id']
                check = 1
                # text_ID.append(image_id.split('-')[0])
                text_dataset.append(np.asarray(ids))

                short_data.append(df_data[count-1])
                ########################################################

                image_id = anns[i]['image_id']
                
                # label
                # category_dataset.append(rough_detail_class[id_to_class[anns[i]['image_id'].split('-')[0]]])
                category_dataset.append(rough_detail_class[id_to_class[anns[i]['image_id'].split('-')[0]]])
                # 158
                # class_num[rough_detail_class[id_to_class[anns[i]['image_id'].split('-')[0]]]] += 1
                class_num[int(rough_detail_class[id_to_class[anns[i]['image_id'].split('-')[0]]])] += 1

                # image
                bbox_convert = box_convert(torch.tensor(anns[i]['bbox']),"cxcywh","xywh")
                bbox_convert = bbox_convert.numpy()
                img = Image.open(f"{paths}/Images/{file}/{image_id}.png")
                
                
                new_img = img.crop((bbox_convert[0]*img.width, bbox_convert[1]*img.height, 
                                    (bbox_convert[0]+bbox_convert[2])*img.width, (bbox_convert[1]+bbox_convert[3])*img.height))


                new_img.save(f"{paths}/Images/cut_down/{file}/{anns[i]['id']}.png")
                image_ID.append(anns[i]['id'])

                image_dataset.append(f"{paths}/Images/cut_down/{file}/{anns[i]['id']}.png")

            else:
                continue

    for i in range (len(detail_rough_class)):
        for j in range(len(image_dataset)):
            if id_to_class[image_ID[j].split('-')[0]] == detail_rough_class[str(i)]:
                image_sorted.append(image_dataset[j])
    
    # for i in range (len(image_dataset)*5):
    #     image_data.append(image_dataset[i//5])
    #     # label
    #     category.append(category_dataset[i//5])
    image_data = image_dataset
    category = category_dataset
    # 158
    for i in range (len(detail_rough_class)):
        for j in range(len(text_dataset)):
            if id_to_class[image_ID[j].split('-')[0]] == detail_rough_class[str(i)]:
                text_data.append(text_dataset[j])
                text_ID.append(image_ID[j].split('-')[0])

    # 32
    # for i in range (32):
    #     for j in range(len(text_dataset)):
    #         if id_to_class[image_ID[j].split('-')[0]][:2] == detail_rough_class[str(i)][:2]:
    #             text_data.append(text_dataset[j])
    #             text_ID.append(image_ID[j].split('-')[0])

            
    np.save(f"{file}_class_num.npy",class_num)
    total = 0
    for i in range (len(class_num)):
        total += class_num[i]
        class_num[i] = total
    np.save(f"{save_path}/{file}/{file}_class_num.npy",class_num)
    np.save(f'{save_path}/{file}/{file}_image.npy',np.asarray(image_data))
    np.save(f'{save_path}/{file}/{file}_image_sorted.npy',np.asarray(image_sorted))
    np.save(f'{save_path}/{file}/{file}_text.npy',np.asarray(text_data))
    np.save(f'{save_path}/{file}/{file}_ori_text.npy',np.asarray(short_data))
    np.save(f'{save_path}/{file}/{file}_category.npy',np.asarray(category))
    np.save(f'{save_path}/{file}/{file}_imageID_order.npy',np.asarray(image_ID))
    np.save(f'{save_path}/{file}/{file}_textID_order.npy',np.asarray(text_ID))
    np.save(f'{save_path}/{file}/{file}_dataset_text.npy',np.asarray(text_dataset))
    np.save(f'{save_path}/{file}/{file}_dataset_image.npy',np.asarray(image_dataset))
    np.save(f'{save_path}/{file}/{file}_dataset_category.npy',np.asarray(category_dataset))
    
    return 0





if __name__ == '__main__':

    paths = './baseline/data'
    data_save_path = f'{paths}/sorted'

    MAX_LENGTH = 266
    NUM_LABELS = 10
    file_name = ["NO","IQ","TI","AN","AD","GN","GD","AX","IN","AB"]
    
    word_pieces = np.asarray("[CLS]")
    PRETRAINED_MODEL_NAME = "bert-base-chinese"
    
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    print("Train:")
    match_training_data(paths, data_save_path, word_pieces,tokenizer,"train",file_name, MAX_LENGTH)
    print("Validation:")
    match_training_data(paths, data_save_path, word_pieces,tokenizer,"validation",file_name, MAX_LENGTH)
    print("Test:")
    match_training_data(paths, data_save_path, word_pieces,tokenizer,"test",file_name, MAX_LENGTH)
    print(len(dict_test))
    # for i in range (len(text_num)):
    #     print(text_num[i])
    print("Done!")
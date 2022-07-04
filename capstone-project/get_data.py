import os
import json
import random
import pandas as pd
import numpy as np
from PIL import Image

raw_path  = './Patent Dataset with bbox'
save_path = './baseline/data'
image_dir = 'Images'
info_dir  = 'Info'
folds = 10
data_class = {}
rough_detail_class = {}
detail_rough_class = {}
data_name  = {}
random.seed(777)

def get_raw_paths(raw_path):
    
    annotation_dir = f"{raw_path}/Annotation"
    paths = []
    i = 0
    for main_dir in os.listdir(annotation_dir):
        dir = f"{annotation_dir}/{main_dir}"
        # 158
        rough_detail_class[main_dir] = i
        detail_rough_class[i] = main_dir
        # 32
        # print(main_dir)
        # if main_dir[:2] != "99":
        #     rough_detail_class[main_dir] = int(main_dir[:2])-1
        #     detail_rough_class[int(main_dir[:2])-1] = main_dir
        # else:
        #     rough_detail_class[main_dir] = 31
        #     detail_rough_class[31] = main_dir
        i+=1
        for sub_dir in os.listdir(dir):
            
            dir = f"{annotation_dir}/{main_dir}/{sub_dir}"
            paths.append({
                'path': f"{annotation_dir}/{main_dir}/{sub_dir}"
            })
        

    return paths

def get_paths(raw_path,data_class):

    paths       = []
    text_paths  = []

    dic = []
    for i, path in enumerate(raw_path):
        dir = path["path"]
        sub_path = dir.split('/')
        file_num = 0
        
        for file in os.listdir(dir):

            file_num += 1

            if len(file.split('.')) == 2 and file.split('.')[-1] == 'txt':
                paths.append({
                    'id': f"{sub_path[4]}-{file.split('.')[0]}", 
                    'annotation': f"{dir}/{file}", 
                    'image': f"{sub_path[0]}/{sub_path[1]}/ImageSets/{sub_path[3]}/{sub_path[4]}/{file.split('.')[0]}.png",
                    
                })
                data_class[f"{sub_path[4]}-{file.split('.')[0]}"] = sub_path[3]
                data_class[f"{sub_path[4]}"] = sub_path[3]

            else:
                print(f"Except:{dir}/{file}")
        text_paths.append({
            'text': f"{sub_path[0]}/{sub_path[1]}/Info/{sub_path[3]}/{sub_path[4]}.csv"
        })

        dic.append(file_num)


    return paths, text_paths, dic, data_class
                    
def get_info(paths, save_path, info_dir, sub_dir, dic_for_num, data_name):

    if not os.path.isdir(f"{save_path}/{info_dir}/{sub_dir}"):
        os.mkdir(f"{save_path}/{info_dir}/{sub_dir}")
    data_num = len(paths)
    info_list = [list() for i in range(10)]

    for i, path in enumerate(paths):
        df = pd.read_csv(path['text'])
        [info_list[j].append(df.columns[j]) for j in range(10)]
        data_name[info_list[0][i]] = info_list[2][i]

        if i % 100 == 0:
            print(f"Progress: [{i + 1} / {data_num}]", end='\r')

    file_name = ["NO","IQ","TI","AN","AD","GN","GD","AX","IN","AB"]
    for j in range(10):
        np.save(f"{save_path}/{info_dir}/{sub_dir}/{file_name[j]}.npy", np.asarray(info_list[j]))

    np.save(f"{save_path}/{info_dir}/{sub_dir}/{sub_dir}_nums.npy", np.asarray(dic_for_num))
    print()

    return data_name


def get_annotations(paths):
    data_num = len(paths)
    annotation_list = []
    for i, path in enumerate(paths):
        with open(path['annotation'], 'r') as f:
            annotation = f.read()
            
            annotation = annotation.split('\n')
            # print(annotation)
            for j, a in enumerate(annotation[:-1]):
                a = a.split(' ')
                annotation_list.append({
                    'id': f"{path['id']}-{j + 1:03d}", 
                    'image_id': path['id'], 
                    'category_id': int(a[0]), 
                    'bbox': [float(b) for b in a[1:]]
                })

        if i % 100 == 0:
            print(f"Progress: [{i + 1} / {data_num}]", end='\r')
    print()

    return annotation_list


def get_images(paths, save_path, image_dir, sub_dir):
    data_num = len(paths)
    image_list = []

    if not os.path.isdir(f"{save_path}/{image_dir}/{sub_dir}"):
        os.mkdir(f"{save_path}/{image_dir}/{sub_dir}")

    for i, path in enumerate(paths):
        image = Image.open(path['image'])
        w, h = image.size
        file_path = f"{sub_dir}/{path['id']}.png"
        image_list.append({
            'id': path['id'], 
            'width': w, 
            'height': h, 
            'file_path': file_path
        })
        image.save(f"{save_path}/{image_dir}/{file_path}", 'PNG')

        if i % 100 == 0:
            print(f"Progress: [{i + 1} / {data_num}]", end='\r')
    print()

    return image_list


paths = get_raw_paths(raw_path)
fold_data_num = len(paths) // folds

random.shuffle(paths)
train_paths = paths[:fold_data_num * 8]
val_paths = paths[fold_data_num * 7:fold_data_num * 8]
test_paths = paths[fold_data_num * 8:]

if not os.path.isdir(save_path):
    os.mkdir(save_path)

if not os.path.isdir(f"{save_path}/{image_dir}"):
    os.mkdir(f"{save_path}/{image_dir}")

if not os.path.isdir(f"{save_path}/annotations"):
    os.mkdir(f"{save_path}/annotations")

if not os.path.isdir(f"{save_path}/Info"):
    os.mkdir(f"{save_path}/Info")


print('========================================')
print('[Train Dataset]')
train_paths, text_paths, dic_for_nums, data_class = get_paths(train_paths, data_class)
print('Info:')
data_name = get_info(text_paths, save_path, info_dir, 'train', dic_for_nums, data_name)
print('Annotation:')
train_anns = get_annotations(train_paths)
print('Image:')
train_imgs = get_images(train_paths, save_path, image_dir, 'train')

train_ann_file = {
    'image': train_imgs, 
    'annotation': train_anns
}

train_ann_file = json.dumps(train_ann_file)

with open(f"{save_path}/annotations/annotations-train.json", 'w') as f:
    f.write(train_ann_file)
print('========================================')
print('[Val Dataset]')
val_paths, text_paths, dic_for_nums, data_class = get_paths(val_paths, data_class)
print('Info:')
data_name = get_info(text_paths, save_path, info_dir, 'validation', dic_for_nums, data_name)
print('Annotation:')
val_anns = get_annotations(val_paths)
print('Image:')
val_imgs = get_images(val_paths, save_path, image_dir, 'validation')

val_ann_file = {
    'image': val_imgs, 
    'annotation': val_anns
}

val_ann_file = json.dumps(val_ann_file)

with open(f"{save_path}/annotations/annotations-validation.json", 'w') as f:
    f.write(val_ann_file)
print('========================================')
print('[Test Dataset]')
test_paths, text_paths, dic_for_nums, data_class = get_paths(test_paths, data_class)
print('Info:')
data_name = get_info(text_paths, save_path, info_dir, 'test', dic_for_nums, data_name)
print('Annotation:')
test_anns = get_annotations(test_paths)
print('Image:')
test_imgs = get_images(test_paths, save_path, image_dir, 'test')

test_ann_file = {
    'image': test_imgs, 
    'annotation': test_anns
}

test_ann_file = json.dumps(test_ann_file)

with open(f"{save_path}/annotations/annotations-test.json", 'w') as f:
    f.write(test_ann_file)

save_file = open(f"{save_path}/dic_for_class.json", "w")
json.dump(data_class, save_file)
save_file.close()

save_file = open(f"{save_path}/dic_for_name.json", "w")
json.dump(data_name, save_file)
save_file.close()

save_file = open(f"{save_path}/rough_detail_class.json", "w")
json.dump(rough_detail_class, save_file)
save_file.close()

save_file = open(f"{save_path}/detail_rough_class.json", "w")
json.dump(detail_rough_class, save_file)
save_file.close()
print('========================================')
print('Done!')
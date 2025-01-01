import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
from PIL import Image
import os

from multiprocessing import Pool,Manager


vspw_path = Path() / "/datadisk2/lixinhao/vss/data/test/vspw/VSPW_480p/data" 
# coco_stuff = 

file_train_path = "/home/lixinhao/vss/data/test/vspw/VSPW_480p/train.txt"
file_test_path = "/home/lixinhao/vss/data/test/vspw/VSPW_480p/test.txt"
file_val_path = "/home/lixinhao/vss/data/test/vspw/VSPW_480p/val.txt"

train_set = set()
val_set = set()

with open(file_train_path,"r") as file:    
    for line in file:
        line = line.strip()
        train_set.add(line)
        
# with open(file_test_path,"r") as file:
#     for line in file:
#         line = line.strip()
#         train_set.add(line)
        
with open(file_val_path,"r") as file:    
    for line in file:
        line = line.strip()
        val_set.add(line)

with Manager() as manager:
    idset = manager.list()

def worker(file_tuple):
    read_path, out_path_file = file_tuple
    
    print("read: ",read_path,out_path_file)
    
    lab = np.asarray(Image.open(read_path))
    

    # for i in np.unique(lab):
        # if i not in idset:
        #     idset.append(i)
    # if 124 in lab:
    #     print(np.unique(lab))
    #     print(file_tuple)
    #     exit()
                    
    # assert lab.dtype == np.uint8

    # output = np.zeros_like(lab, dtype=np.uint8) + 255
    # for obj_id in np.unique(lab):
    #     # if obj_id in train_class:
    #     # if obj_id <= 111 :
    #     output[lab == obj_id] = obj_id
    #         # output[lab == obj_id] = id_map[obj_id]
            
    #         # if obj_id not in idset:
    #         #     idset.append(obj_id)

    Image.fromarray(lab).save(out_path_file)

pool = Pool(32)

aset = set()
bset = set()
cset = set()
# typemy = "mask"
typemy = "origin"

for t in ["train","val"]:
    # for type in ['mask','origin']:
    file_list = []
    for root,dirs,files in os.walk(vspw_path):
        for filename in files:
            seri = os.path.split(root)[0].split(os.path.sep)[-1]
            type_d = os.path.split(root)[1]
            # print("seri",seri,type_d)
            if t == "train":
                work_set = train_set
            else:
                work_set = val_set
            cset.add(type_d)
            # 如果是label
            if(seri in work_set and type_d == typemy and typemy == "mask"):
                directory, s_filename = os.path.split(root)
                # print(directory,s_filename)
                prefix = directory.split(os.path.sep)[-1]
                read_path = Path() / root / filename
                out_path = Path() / "/home/lixinhao/testd/DeOP/datasets/cocos/stuffthingmaps"/(t+"2017")
                out_path.mkdir(parents=True, exist_ok=True)
                out_path_file = out_path/(prefix+"_"+filename)
                file_list.append((read_path,out_path_file))
                aset.add(out_path_file)
                bset.add(read_path)
            #如果是图像
            elif (seri in work_set and type_d == typemy and typemy == "origin"):
                directory, s_filename = os.path.split(root)
                # print(directory.split(os.path.sep)[-2:])
                prefix = directory.split(os.path.sep)[-1]
                read_path = Path() / root / filename
                out_path = Path() / "/home/lixinhao/testd/DeOP/datasets/cocos"/(t+"2017")
                out_path.mkdir(parents=True, exist_ok=True)
                out_path_file = out_path/(prefix+"_"+filename)
                file_list.append((read_path,out_path_file))
                aset.add(out_path_file)
                bset.add(read_path)
            
            # print(root,filename,file_list)
            # exit()
    print(len(aset),len(bset),len(cset),cset)
    print(file_list[0])
    print(f'total file:{len(file_list)}')
    # exit()
    file_list = file_list[:200]
    # for item in tqdm(file_list, desc="Processing"):
    #     worker(item)        
    pool.map(worker,file_list)
    print('done {}'.format(t))
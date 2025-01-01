import mmcv
import os
import shutil

ann_dir = "/datadisk2/lixinhao/cityscapes_clips/gtFine/val"
img_dir = "/datadisk2/lixinhao/cityscapes_clips/leftImg8bit/val"

img_suffix = "_leftImg8bit.png"
seg_map_suffix = "_gtFine_labelTrainIds.png"

new_img_path = "/datadisk2/lixinhao/vss/data/cityscapes/val2017/"
new_anns_path = "/datadisk2/lixinhao/vss/data/cityscapes/stuffthingmaps/val2017/"

img_infos = []
for ann_name in mmcv.scandir(ann_dir, seg_map_suffix, recursive=True):
    img=ann_name.replace(seg_map_suffix,img_suffix)

    img_info = dict(filename=img)
    if ann_dir is not None:
        seg_map = ann_name
        img_info['seg_map'] = seg_map
    img_infos.append(img_info)
    # print(img_info)
    # break

for i in img_infos:
    # new_img_path_deep = "/".join((new_img_path+i["filename"]).split("/")[:-1])
    # new_ann_path_deep = "/".join((new_anns_path+i["seg_map"]).split("/")[:-1])
    # # print(new_img_path_deep,new_ann_path_deep)

    # if not os.path.exists(new_img_path_deep):
    #     os.makedirs(new_img_path_deep,exist_ok=True)

    # if not os.path.exists(new_ann_path_deep):
    #     os.makedirs(new_ann_path_deep,exist_ok=True)

    #先处理img
    shutil.copyfile(img_dir+"/"+i["filename"],new_img_path+i["filename"].split("/")[-1])

    #后处理label
    shutil.copyfile(ann_dir+"/"+i["seg_map"],new_anns_path+i["seg_map"].split("/")[-1])


import os
import numpy as np
import json
import copy

VSPW_CATEGORIES = [{
    'color': [120, 120, 120],
    'isthing': 1,
    'id': 1,
    'name': 'wall'
}, {
    'color': [180, 120, 120],
    'isthing': 1,
    'id': 2,
    'name': 'ceiling'
}, {
    'color': [6, 230, 230],
    'isthing': 1,
    'id': 3,
    'name': 'door'
}, {
    'color': [80, 50, 50],
    'isthing': 1,
    'id': 4,
    'name': 'stair'
}, {
    'color': [4, 200, 3],
    'isthing': 1,
    'id': 5,
    'name': 'ladder'
}, {
    'color': [120, 120, 80],
    'isthing': 1,
    'id': 6,
    'name': 'escalator'
}, {
    'color': [140, 140, 140],
    'isthing': 1,
    'id': 7,
    'name': 'Playground_slide'
}, {
    'color': [204, 5, 255],
    'isthing': 1,
    'id': 8,
    'name': 'handrail_or_fence'
}, {
    'color': [230, 230, 230],
    'isthing': 1,
    'id': 9,
    'name': 'window'
}, {
    'color': [4, 250, 7],
    'isthing': 1,
    'id': 10,
    'name': 'rail'
}, {
    'color': [224, 5, 255],
    'isthing': 1,
    'id': 11,
    'name': 'goal'
}, {
    'color': [235, 255, 7],
    'isthing': 1,
    'id': 12,
    'name': 'pillar'
}, {
    'color': [150, 5, 61],
    'isthing': 1,
    'id': 13,
    'name': 'pole'
}, {
    'color': [120, 120, 70],
    'isthing': 1,
    'id': 14,
    'name': 'floor'
}, {
    'color': [8, 255, 51],
    'isthing': 1,
    'id': 15,
    'name': 'ground'
}, {
    'color': [255, 6, 82],
    'isthing': 1,
    'id': 16,
    'name': 'grass'
}, {
    'color': [143, 255, 140],
    'isthing': 1,
    'id': 17,
    'name': 'sand'
}, {
    'color': [204, 255, 4],
    'isthing': 1,
    'id': 18,
    'name': 'athletic_field'
}, {
    'color': [255, 51, 7],
    'isthing': 1,
    'id': 19,
    'name': 'road'
}, {
    'color': [204, 70, 3],
    'isthing': 1,
    'id': 20,
    'name': 'path'
}, {
    'color': [0, 102, 200],
    'isthing': 1,
    'id': 21,
    'name': 'crosswalk'
}, {
    'color': [61, 230, 250],
    'isthing': 1,
    'id': 22,
    'name': 'building'
}, {
    'color': [255, 6, 51],
    'isthing': 1,
    'id': 23,
    'name': 'house'
}, {
    'color': [11, 102, 255],
    'isthing': 1,
    'id': 24,
    'name': 'bridge'
}, {
    'color': [255, 7, 71],
    'isthing': 1,
    'id': 25,
    'name': 'tower'
}, {
    'color': [255, 9, 224],
    'isthing': 1,
    'id': 26,
    'name': 'windmill'
}, {
    'color': [9, 7, 230],
    'isthing': 1,
    'id': 27,
    'name': 'well_or_well_lid'
}, {
    'color': [220, 220, 220],
    'isthing': 1,
    'id': 28,
    'name': 'other_construction'
}, {
    'color': [255, 9, 92],
    'isthing': 1,
    'id': 29,
    'name': 'sky'
}, {
    'color': [112, 9, 255],
    'isthing': 1,
    'id': 30,
    'name': 'mountain'
}, {
    'color': [8, 255, 214],
    'isthing': 1,
    'id': 31,
    'name': 'stone'
}, {
    'color': [7, 255, 224],
    'isthing': 1,
    'id': 32,
    'name': 'wood'
}, {
    'color': [255, 184, 6],
    'isthing': 1,
    'id': 33,
    'name': 'ice'
}, {
    'color': [10, 255, 71],
    'isthing': 1,
    'id': 34,
    'name': 'snowfield'
}, {
    'color': [255, 41, 10],
    'isthing': 1,
    'id': 35,
    'name': 'grandstand'
}, {
    'color': [7, 255, 255],
    'isthing': 1,
    'id': 36,
    'name': 'sea'
}, {
    'color': [224, 255, 8],
    'isthing': 1,
    'id': 37,
    'name': 'river'
}, {
    'color': [102, 8, 255],
    'isthing': 1,
    'id': 38,
    'name': 'lake'
}, {
    'color': [255, 61, 6],
    'isthing': 1,
    'id': 39,
    'name': 'waterfall'
}, {
    'color': [255, 194, 7],
    'isthing': 1,
    'id': 40,
    'name': 'water'
}, {
    'color': [255, 122, 8],
    'isthing': 1,
    'id': 41,
    'name': 'billboard_or_Bulletin_Board'
}, {
    'color': [0, 255, 20],
    'isthing': 1,
    'id': 42,
    'name': 'sculpture'
}, {
    'color': [255, 8, 41],
    'isthing': 1,
    'id': 43,
    'name': 'pipeline'
}, {
    'color': [255, 5, 153],
    'isthing': 1,
    'id': 44,
    'name': 'flag'
}, {
    'color': [6, 51, 255],
    'isthing': 1,
    'id': 45,
    'name': 'parasol_or_umbrella'
}, {
    'color': [235, 12, 255],
    'isthing': 1,
    'id': 46,
    'name': 'cushion_or_carpet'
}, {
    'color': [160, 150, 20],
    'isthing': 1,
    'id': 47,
    'name': 'tent'
}, {
    'color': [0, 163, 255],
    'isthing': 1,
    'id': 48,
    'name': 'roadblock'
}, {
    'color': [140, 140, 140],
    'isthing': 1,
    'id': 49,
    'name': 'car'
}, {
    'color': [250, 10, 15],
    'isthing': 1,
    'id': 50,
    'name': 'bus'
}, {
    'color': [20, 255, 0],
    'isthing': 1,
    'id': 51,
    'name': 'truck'
}, {
    'color': [31, 255, 0],
    'isthing': 1,
    'id': 52,
    'name': 'bicycle'
}, {
    'color': [255, 31, 0],
    'isthing': 1,
    'id': 53,
    'name': 'motorcycle'
}, {
    'color': [255, 224, 0],
    'isthing': 1,
    'id': 54,
    'name': 'wheeled_machine'
}, {
    'color': [153, 255, 0],
    'isthing': 1,
    'id': 55,
    'name': 'ship_or_boat'
}, {
    'color': [0, 0, 255],
    'isthing': 1,
    'id': 56,
    'name': 'raft'
}, {
    'color': [255, 71, 0],
    'isthing': 1,
    'id': 57,
    'name': 'airplane'
}, {
    'color': [0, 235, 255],
    'isthing': 1,
    'id': 58,
    'name': 'tyre'
}, {
    'color': [0, 173, 255],
    'isthing': 1,
    'id': 59,
    'name': 'traffic_light'
}, {
    'color': [31, 0, 255],
    'isthing': 1,
    'id': 60,
    'name': 'lamp'
}, {
    'color': [11, 200, 200],
    'isthing': 1,
    'id': 61,
    'name': 'person'
}, {
    'color': [255, 82, 0],
    'isthing': 1,
    'id': 62,
    'name': 'cat'
}, {
    'color': [0, 255, 245],
    'isthing': 1,
    'id': 63,
    'name': 'dog'
}, {
    'color': [0, 61, 255],
    'isthing': 1,
    'id': 64,
    'name': 'horse'
}, {
    'color': [0, 255, 112],
    'isthing': 1,
    'id': 65,
    'name': 'cattle'
}, {
    'color': [0, 255, 133],
    'isthing': 1,
    'id': 66,
    'name': 'other_animal'
}, {
    'color': [255, 0, 0],
    'isthing': 1,
    'id': 67,
    'name': 'tree'
}, {
    'color': [255, 163, 0],
    'isthing': 1,
    'id': 68,
    'name': 'flower'
}, {
    'color': [255, 102, 0],
    'isthing': 1,
    'id': 69,
    'name': 'other_plant'
}, {
    'color': [194, 255, 0],
    'isthing': 1,
    'id': 70,
    'name': 'toy'
}, {
    'color': [0, 143, 255],
    'isthing': 1,
    'id': 71,
    'name': 'ball_net'
}, {
    'color': [51, 255, 0],
    'isthing': 1,
    'id': 72,
    'name': 'backboard'
}, {
    'color': [0, 82, 255],
    'isthing': 1,
    'id': 73,
    'name': 'skateboard'
}, {
    'color': [0, 255, 41],
    'isthing': 1,
    'id': 74,
    'name': 'bat'
}, {
    'color': [0, 255, 173],
    'isthing': 1,
    'id': 75,
    'name': 'ball'
}, {
    'color': [10, 0, 255],
    'isthing': 1,
    'id': 76,
    'name': 'cupboard_or_showcase_or_storage_rack'
}, {
    'color': [173, 255, 0],
    'isthing': 1,
    'id': 77,
    'name': 'box'
}, {
    'color': [0, 255, 153],
    'isthing': 1,
    'id': 78,
    'name': 'traveling_case_or_trolley_case'
}, {
    'color': [255, 92, 0],
    'isthing': 1,
    'id': 79,
    'name': 'basket'
}, {
    'color': [255, 0, 255],
    'isthing': 1,
    'id': 80,
    'name': 'bag_or_package'
}, {
    'color': [255, 0, 245],
    'isthing': 1,
    'id': 81,
    'name': 'trash_can'
}, {
    'color': [255, 0, 102],
    'isthing': 1,
    'id': 82,
    'name': 'cage'
}, {
    'color': [255, 173, 0],
    'isthing': 1,
    'id': 83,
    'name': 'plate'
}, {
    'color': [255, 0, 20],
    'isthing': 1,
    'id': 84,
    'name': 'tub_or_bowl_or_pot'
}, {
    'color': [255, 184, 184],
    'isthing': 1,
    'id': 85,
    'name': 'bottle_or_cup'
}, {
    'color': [0, 31, 255],
    'isthing': 1,
    'id': 86,
    'name': 'barrel'
}, {
    'color': [0, 255, 61],
    'isthing': 1,
    'id': 87,
    'name': 'fishbowl'
}, {
    'color': [0, 71, 255],
    'isthing': 1,
    'id': 88,
    'name': 'bed'
}, {
    'color': [255, 0, 204],
    'isthing': 1,
    'id': 89,
    'name': 'pillow'
}, {
    'color': [0, 255, 194],
    'isthing': 1,
    'id': 90,
    'name': 'table_or_desk'
}, {
    'color': [0, 255, 82],
    'isthing': 1,
    'id': 91,
    'name': 'chair_or_seat'
}, {
    'color': [0, 10, 255],
    'isthing': 1,
    'id': 92,
    'name': 'bench'
}, {
    'color': [0, 112, 255],
    'isthing': 1,
    'id': 93,
    'name': 'sofa'
}, {
    'color': [51, 0, 255],
    'isthing': 1,
    'id': 94,
    'name': 'shelf'
}, {
    'color': [0, 194, 255],
    'isthing': 1,
    'id': 95,
    'name': 'bathtub'
}, {
    'color': [0, 122, 255],
    'isthing': 1,
    'id': 96,
    'name': 'gun'
}, {
    'color': [0, 255, 163],
    'isthing': 1,
    'id': 97,
    'name': 'commode'
}, {
    'color': [255, 153, 0],
    'isthing': 1,
    'id': 98,
    'name': 'roaster'
}, {
    'color': [0, 255, 10],
    'isthing': 1,
    'id': 99,
    'name': 'other_machine'
}, {
    'color': [255, 112, 0],
    'isthing': 1,
    'id': 100,
    'name': 'refrigerator'
}, {
    'color': [143, 255, 0],
    'isthing': 1,
    'id': 101,
    'name': 'washing_machine'
}, {
    'color': [82, 0, 255],
    'isthing': 1,
    'id': 102,
    'name': 'Microwave_oven'
}, {
    'color': [163, 255, 0],
    'isthing': 1,
    'id': 103,
    'name': 'fan'
}, {
    'color': [255, 235, 0],
    'isthing': 1,
    'id': 104,
    'name': 'curtain'
}, {
    'color': [8, 184, 170],
    'isthing': 1,
    'id': 105,
    'name': 'textiles'
}, {
    'color': [133, 0, 255],
    'isthing': 1,
    'id': 106,
    'name': 'clothes'
}, {
    'color': [0, 255, 92],
    'isthing': 1,
    'id': 107,
    'name': 'painting_or_poster'
}, {
    'color': [184, 0, 255],
    'isthing': 1,
    'id': 108,
    'name': 'mirror'
}, {
    'color': [255, 0, 31],
    'isthing': 1,
    'id': 109,
    'name': 'flower_pot_or_vase'
}, {
    'color': [0, 184, 255],
    'isthing': 1,
    'id': 110,
    'name': 'clock'
}, {
    'color': [0, 214, 255],
    'isthing': 1,
    'id': 111,
    'name': 'book'
}, {
    'color': [255, 0, 112],
    'isthing': 1,
    'id': 112,
    'name': 'tool'
}, {
    'color': [92, 255, 0],
    'isthing': 1,
    'id': 113,
    'name': 'blackboard'
}, {
    'color': [0, 224, 255],
    'isthing': 1,
    'id': 114,
    'name': 'tissue'
}, {
    'color': [112, 224, 255],
    'isthing': 1,
    'id': 115,
    'name': 'screen_or_television'
}, {
    'color': [70, 184, 160],
    'isthing': 1,
    'id': 116,
    'name': 'computer'
}, {
    'color': [163, 0, 255],
    'isthing': 1,
    'id': 117,
    'name': 'printer'
}, {
    'color': [153, 0, 255],
    'isthing': 1,
    'id': 118,
    'name': 'Mobile_phone'
}, {
    'color': [71, 255, 0],
    'isthing': 1,
    'id': 119,
    'name': 'keyboard'
}, {
    'color': [255, 0, 163],
    'isthing': 1,
    'id': 120,
    'name': 'other_electronic_product'
}, {
    'color': [255, 204, 0],
    'isthing': 1,
    'id': 121,
    'name': 'fruit'
}, {
    'color': [255, 0, 143],
    'isthing': 1,
    'id': 122,
    'name': 'food'
}, {
    'color': [0, 255, 235],
    'isthing': 1,
    'id': 123,
    'name': 'instrument'
}, {
    'color': [133, 255, 0],
    'isthing': 1,
    'id': 124,
    'name': 'train'
}]

# for item in VSPW_CATEGORIES:
#     item['id'] = item['id'] - 1

# COCO_CATEGORIES_Seen = []
# COCO_CATEGORIES_unseen = []

seen_cls = np.load(r'/home/lixinhao/vss/mmseg/handle_data/group_seen.npy')
val_cls = np.load(r'/home/lixinhao/vss/mmseg/handle_data/group_unseen.npy')
novel_cls = np.load(r'/home/lixinhao/vss/mmseg/handle_data/group_novel.npy')

train_cls = seen_cls.tolist() + val_cls.tolist()


seen_classnames = []
unseen_classnames = []

count_train = 0
count_test = 0
for item in VSPW_CATEGORIES:
    id = item['id']
    name = item['name']

    if int(id)<=111:
        seen_classnames.append(name)
        tmp = copy.deepcopy(item)
        tmp['trainId'] = count_train
        # COCO_CATEGORIES_Seen.append(tmp)
        count_train = count_train + 1
    else:
        unseen_classnames.append(name)
        tmp = copy.deepcopy(item)
        tmp['trainId'] = count_test
        # COCO_CATEGORIES_unseen.append(tmp)
        count_test = count_test + 1

with open(r'/home/lixinhao/vss/mmseg/handle_data/seen_classnames.json',
          'w') as f_out:
    json.dump(seen_classnames, f_out)

with open(r'/home/lixinhao/vss/mmseg/handle_data/unseen_classnames.json',
          'w') as f_out:
    json.dump(unseen_classnames, f_out)

with open(r'/home/lixinhao/vss/mmseg/handle_data/all_classnames.json',
          'w') as f_out:
    json.dump(seen_classnames + unseen_classnames, f_out)

# print(COCO_CATEGORIES_Seen)

# print(COCO_CATEGORIES_unseen)

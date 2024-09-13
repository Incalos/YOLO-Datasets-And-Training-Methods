import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yoloversion", type=str, default='', help="optional: yolov5\yolov6\yolov7\yolov8")
    parser.add_argument("--trainval_percent", type=float, default=0.9,
                        help="percentage of training set and validation set")
    parser.add_argument("--train_percent", type=float, default=0.9, help="percentage of training set")
    parser.add_argument("--mainpath", type=str, default=720, help="the path of the dataset")
    parser.add_argument("--classes", type=str, default='[]', help="Categories of annotations")
    opt = parser.parse_args()
    return opt


# 进行归一化操作
def convert(size, box):  # size:(原图w,原图h) , box:(xmin,xmax,ymin,ymax)
    dw = 1. / size[0]  # 1/w
    dh = 1. / size[1]  # 1/h
    x = (box[0] + box[1]) / 2.0  # 物体在图中的中心点x坐标
    y = (box[2] + box[3]) / 2.0  # 物体在图中的中心点y坐标
    w = box[1] - box[0]  # 物体实际像素宽度
    h = box[3] - box[2]  # 物体实际像素高度
    x = x * dw  # 物体中心点x的坐标比(相当于 x/原图w)
    w = w * dw  # 物体宽度的宽度比(相当于 w/原图w)
    y = y * dh  # 物体中心点y的坐标比(相当于 y/原图h)
    h = h * dh  # 物体宽度的宽度比(相当于 h/原图h)
    return (x, y, w, h)  # 返回 相对于原图的物体中心点的x坐标比,y坐标比,宽度比,高度比,取值范围[0-1]


def convert_annotation(labelsavepath, xmlfilepath, image_id, img_extension):
    in_file = open(os.path.join(xmlfilepath, f'{image_id}.xml'), encoding='utf-8')
    out_file = open(os.path.join(labelsavepath, f'{image_id}.txt'), 'w', encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    if size is not None:
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in eval(opt.classes) or int(difficult) == 1:
                continue
            cls_id = eval(opt.classes).index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            print(image_id, cls, b)
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def find_image_extension(jpgsavepath, image_id):
    for ext in ['.jpg', '.png']:
        if os.path.exists(os.path.join(jpgsavepath, f'{image_id}{ext}')):
            return ext
    return None  # If no valid image is found


def run_main(source_dir, target_dir):
    for root, dirs, files in os.walk(source_dir, topdown=False):
        for name in files:
            file = os.path.join(root, name)
            print(file)
            if file.split(".")[-1] in ["xls", "xlsx", "csv"]:
                tar_file = file.split(".")[-2] + str(num) + "." + file.split(".")[-1]
                if os.path.isfile(target_dir + tar_file.split("\\")[-1]):
                    print("已经存在该文件")
                else:
                    print("正在移动第{}个文件：{}".format(num + 1, tar_file.split("\\")[-1]))
                    os.rename(file, target_dir + tar_file.split("\\")[-1])


if __name__ == '__main__':
    opt = parse_opt()
    xmlfilepath = opt.mainpath + '/Annotations'
    jpgsavepath = opt.mainpath + r'/images'
    txtsavepath = opt.mainpath + '/ImageSets'
    labelsavepath = opt.mainpath + r'/labels'
    if not os.path.exists(txtsavepath):
        os.makedirs(txtsavepath)
    if not os.path.exists(labelsavepath):
        os.makedirs(labelsavepath)
    total_xml = os.listdir(xmlfilepath)
    num = len(total_xml)
    list = range(num)
    tv = int(num * opt.trainval_percent)
    tr = int(tv * opt.train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)
    ftrainval = open(os.path.join(txtsavepath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(txtsavepath, 'test.txt'), 'w')
    ftrain = open(os.path.join(txtsavepath, 'train.txt'), 'w')
    fval = open(os.path.join(txtsavepath, 'val.txt'), 'w')
    for i in list:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    sets = ['train', 'test', 'val']
    for image_set in sets:
        if not os.path.exists(labelsavepath):
            os.makedirs(labelsavepath)
        image_ids = open(os.path.join(txtsavepath, f'{image_set}.txt')).read().strip().split()
        list_file = open(os.path.join(opt.mainpath, f'{image_set}.txt'), 'w')
        for image_id in image_ids:
            img_extension = find_image_extension(jpgsavepath, image_id)
            if img_extension:
                list_file.write(jpgsavepath + f'/{image_id}{img_extension}\n')
                convert_annotation(labelsavepath, xmlfilepath, image_id, img_extension)
        list_file.close()
    if opt.yoloversion == 'yolov6':
        for image_set in sets:
            labelpath = opt.mainpath + r'/labels' + r'/{}'.format(image_set)
            jpgpath = opt.mainpath + r'/images' + r'/{}'.format(image_set)
            if not os.path.exists(jpgpath):
                os.makedirs(jpgpath)
            if not os.path.exists(labelpath):
                os.makedirs(labelpath)
            image_ids = open(os.path.join(txtsavepath, f'{image_set}.txt')).read().strip().split()
            for image_id in image_ids:
                img_extension = find_image_extension(jpgsavepath, image_id)
                if img_extension:
                    shutil.move(jpgsavepath + f'/{image_id}{img_extension}', jpgpath)
                    shutil.move(labelsavepath + f'/{image_id}.txt', labelpath)
        print('Finished！')

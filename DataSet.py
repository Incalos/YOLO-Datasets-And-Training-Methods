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


def convert_annotation(labelsavepath, xmlfilepath, image_id):
    '''
    将对应文件名的xml文件转化为label文件，xml文件包含了对应的bunding框以及图片长款大小等信息，
    通过对其解析，然后进行归一化最终读到label文件中去，也就是说
    一张图片文件对应一个xml文件，然后通过解析和归一化，能够将对应的信息保存到唯一一个label文件中去
    labal文件中的格式：calss x y w h　　同时，一张图片对应的类别有多个，所以对应的ｂｕｎｄｉｎｇ的信息也有多个
    '''
    in_file = open(os.path.join(xmlfilepath, '%s.xml' % (image_id)), encoding='utf-8')
    # 准备在对应的image_id 中写入对应的label，分别为
    # <object-class> <x> <y> <width> <height>
    out_file = open(os.path.join(labelsavepath, '%s.txt' % (image_id)), 'w', encoding='utf-8')
    # 解析xml文件
    tree = ET.parse(in_file)
    # 获得对应的键值对
    root = tree.getroot()
    # 获得图片的尺寸大小
    size = root.find('size')
    # 如果xml内的标记为空，增加判断条件
    if size != None:
        # 获得宽
        w = int(size.find('width').text)
        # 获得高
        h = int(size.find('height').text)
        # 遍历目标obj
        for obj in root.iter('object'):
            # 获得difficult ？？
            difficult = obj.find('difficult').text
            # 获得类别 =string 类型
            cls = obj.find('name').text
            # 如果类别不是对应在我们预定好的class文件中，或difficult==1则跳过
            if cls not in eval(opt.classes) or int(difficult) == 1:
                continue
            # 通过类别名称找到id
            cls_id = eval(opt.classes).index(cls)
            # 找到bndbox 对象
            xmlbox = obj.find('bndbox')
            # 获取对应的bndbox的数组 = ['xmin','xmax','ymin','ymax']
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            print(image_id, cls, b)
            # 带入进行归一化操作
            # w = 宽, h = 高， b= bndbox的数组 = ['xmin','xmax','ymin','ymax']
            bb = convert((w, h), b)
            # bb 对应的是归一化后的(x,y,w,h)
            # 生成 class x y w h 在label文件中
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def run_main(source_dir, target_dir):
    for root, dirs, files in os.walk(source_dir, topdown=False):
        """
        root 表示当前正在访问的文件夹路径
        dirs 表示该文件夹下的子目录名list
        files 表示该文件夹下的文件list
        """
        for name in files:
            file = os.path.join(root, name)  # 拼接文件的完整路径(注意我们对一个文件或文件夹操作，一定要使用绝对路径)
            print(file)
            if file.split(".")[-1] in ["xls", "xlsx", "csv"]:  # 使用split判断获得的文件路径是不是以csv结尾
                # print(file)
                tar_file = file.split(".")[-2] + str(num) + "." + file.split(".")[
                    -1]  # 为了避免有重名文件，给原文件名后加一个递增序号num形成新的文件名
                # print(target_dir+tar_file.split("\\")[-1])
                if os.path.isfile(target_dir + tar_file.split("\\")[-1]):  # 判断目标文件夹是否已存在该文件
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
        '''
        对所有的文件数据集进行遍历,主要做了两个工作：
    　　　　１．将所有图片文件都遍历一遍，并且将其所有的全路径都写在对应的txt文件中去，方便定位
    　　　　２．同时对所有的图片文件进行解析和转化，将其对应的bundingbox以及类别的信息全部解析写到label文件中去最后再通过直接读取文件，就能找到对应的label信息
        '''
        # 先找labels文件夹如果不存在则创建
        if not os.path.exists(labelsavepath):
            os.makedirs(labelsavepath)
        # 读取在ImageSets/Main 中的train、test..等文件的内容
        # 包含对应的文件名称
        image_ids = open(os.path.join(txtsavepath, '%s.txt' % (image_set))).read().strip().split()
        list_file = open(os.path.join(opt.mainpath, '%s.txt' % (image_set)), 'w')
        # 将对应的文件_id以及全路径写进去并换行
        for image_id in image_ids:
            list_file.write(jpgsavepath + '/%s.jpg\n' % (image_id))
            # 调用  year = 年份  image_id = 对应的文件名_id
            convert_annotation(labelsavepath, xmlfilepath, image_id)
        # 关闭文件
        list_file.close()
    if opt.yoloversion == 'yolov6':
        for image_set in sets:
            labelpath = opt.mainpath + r'/labels' + r'/{}'.format(image_set)
            jpgpath = opt.mainpath + r'/images' + r'/{}'.format(image_set)
            if not os.path.exists(jpgpath):
                os.makedirs(jpgpath)
            if not os.path.exists(labelpath):
                os.makedirs(labelpath)
            image_ids = open(os.path.join(txtsavepath, '%s.txt' % (image_set))).read().strip().split()
            for image_id in image_ids:
                shutil.move(jpgsavepath + '/%s.jpg' % (image_id), jpgpath)
                shutil.move(labelsavepath + '/%s.txt' % (image_id), labelpath)
        print('Finished！')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultralytics YOLO训练脚本 - 支持YOLOv8/v9/v10/v11
使用方法：
    python train_simple.py --model yolov8n.pt --data YoloDataSets/data.yaml
"""

import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Ultralytics YOLO Training')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                       help='模型文件路径 (例如: yolov8n.pt, yolov9c.pt, yolov10n.pt, yolov11n.pt)')
    parser.add_argument('--data', type=str, required=True, 
                       help='数据集配置文件路径 (例如: YoloDataSets/data.yaml)')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='图像大小')
    parser.add_argument('--device', type=str, default='0', help='设备 (0, 1, 2, ... 或 cpu)')
    parser.add_argument('--project', type=str, default='runs/train', help='项目目录')
    parser.add_argument('--name', type=str, default='exp', help='实验名称')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"正在加载模型: {args.model}")
    model = YOLO(args.model)
    
    print(f"开始训练...")
    print(f"数据集: {args.data}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch}")
    print(f"图像大小: {args.imgsz}")
    print(f"设备: {args.device}")
    
    # 训练模型
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        plots=True,
        val=True
    )
    
    print("训练完成！")
    print(f"结果保存在: {results.save_dir}")

if __name__ == '__main__':
    main()

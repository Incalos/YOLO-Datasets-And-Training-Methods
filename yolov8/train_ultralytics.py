from ultralytics import YOLO

# 加载模型
# YOLOv8: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
# YOLOv9: yolov9c.pt, yolov9e.pt
# YOLOv10: yolov10n.pt, yolov10s.pt, yolov10m.pt, yolov10b.pt, yolov10l.pt, yolov10x.pt
# YOLOv11: yolov11n.pt, yolov11s.pt, yolov11m.pt, yolov11l.pt, yolov11x.pt
model = YOLO('yolov8n.pt')  # 可替换为其他版本

# 训练模型
results = model.train(
    data='YoloDataSets/data.yaml',  # 数据集配置文件路径
    epochs=300,                     # 训练轮数
    imgsz=640,                      # 输入图像大小
    batch=16,                       # 批次大小，根据GPU内存调整
    device=0,                       # GPU设备，0表示第一个GPU，'cpu'表示使用CPU
    project='runs/train',           # 训练结果保存目录
    name='ultralytics_custom',      # 本次训练的名称
    save=True,                      # 保存训练结果
    save_period=50,                 # 每50个epoch保存一次模型
    cache=False,                    # 是否缓存图像以加快训练
    workers=8,                      # 数据加载的线程数
    optimizer='SGD',                # 优化器：'SGD', 'Adam', 'AdamW', 'RMSProp'
    lr0=0.01,                       # 初始学习率
    momentum=0.937,                 # SGD动量
    weight_decay=0.0005,            # 权重衰减
    warmup_epochs=3,                # 预热轮数
    warmup_momentum=0.8,            # 预热动量
    box=7.5,                        # box损失权重
    cls=0.5,                        # 类别损失权重
    dfl=1.5,                        # DFL损失权重
    val=True,                       # 训练期间进行验证
    plots=True,                     # 生成训练图表
    verbose=True                    # 详细输出
)

# 验证模型
metrics = model.val()
print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")
print(f"mAP75: {metrics.box.map75}")

# 导出模型
model.export(format='onnx')  # 导出为ONNX格式，可选：'torchscript', 'engine', 'coreml' 等

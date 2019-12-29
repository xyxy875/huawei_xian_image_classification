# huawei_xian_image_classification

最近参加了“华为云杯”2019人工智能创新应用大赛，是一个图像分类的比赛，最终准确率0.972，大概排50多/732。但决赛取前20名，遗憾败北（第20名的准确率是0.982）。
第一次参加类似比赛，总结记录一下。

[TOC]
### 赛题
对西安的热门景点、美食、特产、民俗、工艺品等图片进行分类，共有54类。
官方给出了3000余张数据集，官方判分的不公开测试集有1000张图片。
要求：不允许使用“测试时增强”策略和“模型融合”策略，可以使用其他来源的图片数据。

### 数据
#### 数据分析
绘制了混淆矩阵，对难分的类别进行有针对性的增广。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191229223729843.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MDU1ODE4,size_16,color_FFFFFF,t_70)
计算混淆矩阵的代码如下：
```python
from sklearn.metrics import confusion_matrix
def validate(val_loader, model, criterion, args, epoch):
	……
    # confusion_matrix init
    cm = np.zeros((54, 54))

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # compute_confusion _ added by YJ on 20191209
            cm += compute_confusion(output, target)
            ……
        confusion_file_name = os.path.join(args.train_local, 'epoch_{}.npy'.format(epoch))
        np.save(confusion_file_name, cm)

def compute_confusion(output, target):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze()
    pred_class = torch.Tensor([int(idx_to_class[x]) for x in pred.cpu().numpy().tolist()]).int()
    target_class = torch.Tensor([int(idx_to_class[x]) for x in target.cpu().numpy().tolist()]).int()
    cm = confusion_matrix(target_class, pred_class, labels=range(54))
    return cm
```
绘制混淆矩阵的代码如下：
```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cm = np.load('/Users/yujin/Downloads/epoch_18.npy')
# for i in range(54):
#     cm[i] = cm[i] / cm[i].sum(axis=0)
plt.figure(figsize=(30, 24))
sns.set()
ax = sns.heatmap(cm,annot=False, cmap="OrRd",linewidths=0.1,linecolor="white")
plt.savefig('confusion.jpg')
plt.show()
```
#### 数据爬取
根据54个类别，从百度图片爬取数据（[爬虫代码](https://github.com/kong36088/BaiduImageSpider)）
#### 数据增广
经过多次尝试，最后确定的增广方式为：
```python
transforms.Compose([
	transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08),
	transforms.RandomResizedCrop(224, scale=(0.3, 1)),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	normalize,
]))
```
### 方法
#### 网络结构
最终采用的模型是```se_resnext101_32x4d```（使用在imagenet训练过的权重），在网络中加入了```CBAM```
**参考：**
[ResNeXt、SENet、SE-ResNeXt论文代码学习与总结](https://blog.csdn.net/xzy528521717/article/details/86582889)
[CBAM: Convolutional Block Attention Module](https://blog.csdn.net/seniusen/article/details/90166359)
[CBAM的pytorch实现代码](https://github.com/xyxy875/huawei_xian_image_classification/blob/master/pretrained_models/models/senet.py)

#### 训练方法&超参
1. 根据类别数量修改最后的全连接层FC，并在FC前设置0.3的dropout；
2. 使用adam优化器，设置初始lr为3e-4，之后的lr分别为1e-4、1e-5、1e-6；
3. 先冻结其它层，只训练FC的参数，收敛到一定程度后，再放开所有层一起训练，其中，layer1和layer2的lr为其他层的0.3倍；

#### 其他trick
##### mix_up
```python
MIXUP_EPOCH = 50	# 从第50个epoch开始mix_up
……
    if epoch > MIXUP_EPOCH:
        mix_up_flag = True
    else:
        mix_up_flag = False

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if mix_up_flag:
            inputs, targets_a, targets_b, lam = mixup_data(images.cuda(), target.cuda(), alpha=1.0)  # related to mix_up
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)  # related to mix_up
        else:
        	……
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# related to mix_up
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```
##### weighted loss
```python
    weight = torch.Tensor(np.ones(54))
    weak_class = [3, 7, ……]
    strong_class = [0, 11,……]
    weak_idx = [train_dataset.class_to_idx[str(x)] for x in weak_class]
    strong_idx = [train_dataset.class_to_idx[str(x)] for x in strong_class]
    weight[weak_idx] = 1.2
    weight[strong_idx] = 0.8

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(weight=weight).cuda()
```


# remove-stamp

This repository is the implementation of GAN, a neural network for end-to-end remove stamp.用gan实现的印章擦除，同时你也可以用这个方法擦除去其他噪声，比如水印，手写字等


## Environment

```
python = 3.7
pytorch = 1.3.1
torchvision = 0.4.2
```

## Training

```
python train.py --batchSize 2 \
  --dataRoot 'your path' \
  --modelsSavePath 'your path' \
  --logPath 'your path'  \
```

## Testing


```
python test_image_STE.py --dataRoot 'your path'  \
            --batchSize 1 \
            --pretrain 'your path' \
            --savePath 'your path'
```
## Model

- 链接: https://pan.baidu.com/s/1WnfQfvsxRyhFx0ODdWztxg 提取码: gsfe 复制这段内容后打开百度网盘手机App，操作更方便哦 
--来自百度网盘超级会员v4的分享

## dataset
- 数据生成与增强 可以看看。自己继续添加
- 链接: https://pan.baidu.com/s/1CTzcDNHuT0OI4o4rrC1aWA 提取码: bay3 复制这段内容后打开百度网盘手机App，操作更方便哦 来自百度网盘超级会员v4的分享

## case
![0](https://github.com/tommyMessi/remove-stamp/blob/main/image/1.png)
![00](https://github.com/tommyMessi/remove-stamp/blob/main/image/11.png)
![0](https://github.com/tommyMessi/remove-stamp/blob/main/image/2.png)
![00](https://github.com/tommyMessi/remove-stamp/blob/main/image/22.png)
![0](https://github.com/tommyMessi/remove-stamp/blob/main/image/3.png)
![00](https://github.com/tommyMessi/remove-stamp/blob/main/image/33.png)
![0](https://github.com/tommyMessi/remove-stamp/blob/main/image/4.png)
![00](https://github.com/tommyMessi/remove-stamp/blob/main/image/44.png)

## other

- 更多ocr 文档解析相关内容 关注微信公众号 hulugeAI


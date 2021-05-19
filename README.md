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

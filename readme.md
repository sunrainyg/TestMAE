### Toy Example to test an idea

Oct, 2024  Yulu Gan

Due to limit resource available, we only test the model on cifar10.

We mainly follow the implementation details in the original MAE paper. However, due to difference between Cifar10 and ImageNet, we make some modification:
- we use vit-tiny instead of vit-base.
- since Cifar10 have only 50k training data, we increase the pretraining epoch from 400 to 2000, and the warmup epoch from 40 to 200. We noticed that, the loss is still decreasing after 2000 epoches.
- we decrease the batch size for training the classifier from 1024 to 128 to mitigate the overfitting.

### Installation
`pip install -r requirements.txt`

### Run
```bash
# pretrained with mae
python mae_pretrain.py --total_epoch 100 --warmup_epoch 10 

# train a mae encoder, then train a ae decoder
python mae_pretrain.py --use_ae_decoder --total_epoch 200 --warmup_epoch 20
```

See logs by `tensorboard --logdir logs`.

### Result

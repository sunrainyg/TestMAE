import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model import *
from utils import setup_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt')
    parser.add_argument('--use_ae_decoder', action='store_true', help='Use AE decoder instead of MAE decoder')
    parser.add_argument('--train_ae_decoder', action='store_true', help='Train AE decoder after MAE encoder')
    parser.add_argument('--ae_decoder_epochs', type=int, default=100, help='Number of epochs to train AE decoder')

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae-pretrain'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MAE_ViT(mask_ratio=args.mask_ratio, use_ae_decoder=args.use_ae_decoder).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()

    def train_epoch(model, dataloader, optim, lr_scheduler, writer, epoch, prefix=''):
        global step_count
        model.train()
        losses = []
        for img, label in tqdm(iter(dataloader)):
            step_count += 1
            img = img.to(device)
            if model.use_ae_decoder:
                print("using ae decoder")
                predicted_img = model(img)
            else:
                predicted_img, mask = model(img)
            # print("predicted_img.shape:", predicted_img.shape)
            # print("img.shape:", img.shape)
            if model.use_ae_decoder:
                loss = torch.mean((predicted_img - img) ** 2)
            else:
                loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar(f'{prefix}mae_loss', avg_loss, global_step=epoch)
        print(f'In epoch {epoch}, average training loss is {avg_loss}.')

    def visualize(model, val_dataset, writer, epoch, prefix=''):
        model.eval()
        with torch.no_grad():
            val_img = torch.stack([val_dataset[i][0] for i in range(16)])
            val_img = val_img.to(device)
            if model.use_ae_decoder:
                predicted_val_img = model(val_img)
                mask = torch.ones_like(predicted_val_img)
            else:
                predicted_val_img, mask = model(val_img)
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            writer.add_image(f'{prefix}mae_image', (img + 1) / 2, global_step=epoch)

    # Train MAE
    for e in range(args.total_epoch):
        train_epoch(model, dataloader, optim, lr_scheduler, writer, e)
        visualize(model, val_dataset, writer, e)
        torch.save(model, args.model_path)

    # Train AE decoder if specified
    if args.train_ae_decoder:
        print("Training AE decoder...")
        model.use_ae_decoder = True
        model.decoder = AE_Decoder(image_size=32, patch_size=2, emb_dim=192, num_layer=4, num_head=3).to(device)

        # Freeze encoder
        for param in model.encoder.parameters():
            param.requires_grad = False

        # Set up new optimizer and scheduler for AE decoder
        ae_optim = torch.optim.AdamW(model.decoder.parameters(), lr=args.base_learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay)
        ae_lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.ae_decoder_epochs * math.pi) + 1))
        ae_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(ae_optim, lr_lambda=ae_lr_func, verbose=True)

        for e in range(args.ae_decoder_epochs):
            train_epoch(model, dataloader, ae_optim, ae_lr_scheduler, writer, e, prefix='ae_')
            visualize(model, val_dataset, writer, e, prefix='ae_')
            torch.save(model, f'ae_decoder_{args.model_path}')

    print("Training completed.")

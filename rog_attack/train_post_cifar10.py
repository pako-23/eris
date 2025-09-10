import os, math, argparse, random
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from networks.generator import Ccgenerator
from networks.discriminator import Discriminator

# ---------- Data ----------
def make_loaders(data_root, batch_size, num_workers=4):
    tfm = transforms.Compose([
        # transforms.Resize((128,128), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),        # [0,1]
    ])
    train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm)
    val   = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tfm)
    dl_tr = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    return dl_tr, dl_va

@torch.no_grad()
def degrade_like_attack(x128, p_jpeg=0.0, noise_std=(0.0, 0.05)):
    """
    Mimic the attack's input to the generator:
      clean 128 -> down to 32 -> (optional) noise -> bicubic up to 128
    """
    B, C, H, W = x128.shape
    x32  = F.interpolate(x128, scale_factor=1/4, mode="bicubic", align_corners=False, recompute_scale_factor=True)
    if noise_std is not None and noise_std[1] > 0:
        std = torch.empty(B,1,1,1, device=x128.device).uniform_(noise_std[0], noise_std[1])
        x32 = torch.clamp(x32 + torch.randn_like(x32)*std, 0.0, 1.0)
    x128_deg = F.interpolate(x32, scale_factor=4.0, mode="bicubic", align_corners=False, recompute_scale_factor=True)
    return x128_deg

# ---------- Losses ----------
def d_hinge(real_logit, fake_logit):
    loss = F.relu(1.0 - real_logit).mean() + F.relu(1.0 + fake_logit).mean()
    return loss

def g_hinge(fake_logit):
    return -fake_logit.mean()

# ---------- Train ----------
def train(args):
    device = torch.device(args.device)

    dl_tr, dl_va = make_loaders(args.data_root, args.batch_size, args.num_workers)

    G = Ccgenerator(in_channels=3, num_classes=10).to(device)
    D = Discriminator().to(device)

    g_opt = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.5, 0.999))

    best_psnr = -1e9
    best_epoch = 0
    epochs_no_improve = 0

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        G.train(); D.train()
        for x, y in tqdm(dl_tr):
            x = x.to(device); y = y.to(device)

            x_in  = degrade_like_attack(x)   # attack-like input (128x128)
            x_tgt = x                        # clean target (128x128)

            # -- update D --
            with torch.no_grad():
                x_hat = G(x_in, y)
            d_opt.zero_grad(set_to_none=True)
            d_real = D(x_tgt)
            d_fake = D(x_hat)
            d_loss = d_hinge(d_real, d_fake)
            d_loss.backward()
            d_opt.step()

            # -- update G --
            g_opt.zero_grad(set_to_none=True)
            x_hat = G(x_in, y)
            d_fake = D(x_hat)
            # L1 keeps fidelity; adversarial improves sharpness
            l1 = F.l1_loss(x_hat, x_tgt)
            g_loss = args.lambda_l1 * l1 + g_hinge(d_fake)
            g_loss.backward()
            g_opt.step()

        # quick val PSNR
        G.eval()
        with torch.no_grad():
            sse = 0.0; npx = 0
            for xv, yv in dl_va:
                xv = xv.to(device); yv = yv.to(device)
                xin = degrade_like_attack(xv)
                xh  = torch.clamp(G(xin, yv), 0, 1)
                sse += ((xh - xv)**2).sum().item()
                npx += xv.numel()
            mse = sse / npx
            psnr = -10.0 * math.log10(mse + 1e-12)

        print(f"[Epoch {epoch:03d}] val PSNR ~ {psnr:.2f} dB")

        if psnr > best_psnr:
            best_psnr = psnr
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "gen_state_dict": G.state_dict(),
                "dis_state_dict": D.state_dict(),
                "val_psnr": best_psnr,
                "epoch": epoch,
            }, out_dir / "postmodel_cifar10_3232.pth")
            print(f"  -> saved {out_dir/'postmodel_cifar10_3232.pth'} (PSNR {best_psnr:.2f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{args.patience} epoch(s).")
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}. "
                      f"Best PSNR {best_psnr:.2f} at epoch {best_epoch}.")
                break

    print("Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data/_raw")
    ap.add_argument("--out_dir", type=str,  default="./model_zoos")
    ap.add_argument("--device", type=str,   default="cuda:3")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int,     default=100)
    ap.add_argument("--g_lr", type=float,     default=2e-4)
    ap.add_argument("--d_lr", type=float,     default=2e-4)
    ap.add_argument("--lambda_l1", type=float, default=10.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=2)  # <-- early stopping patience
    args = ap.parse_args()
    train(args)

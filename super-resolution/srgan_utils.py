import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm



class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, num_patches=16, crop_size=128):
        self.image_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(subdir, file))
        self.num_patches = num_patches
        self.crop_size = crop_size

    def __getitem__(self, idx):
        idx = idx // self.num_patches
        img = Image.open(self.image_paths[idx]).convert('L')
        img = patch_transform(self.crop_size)(img)
        input = img.copy()
        return transforms.ToTensor()(input), transforms.ToTensor()(img)

    def __len__(self):
        return len(self.image_paths) * self.num_patches


class BrainTumorTestDataset(Dataset):
    def __init__(self, root_dir, crop_size=128):
        self.image_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(subdir, file))
        self.crop_size = crop_size

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        img = test_transform(self.crop_size)(img)
        return transforms.ToTensor()(img), transforms.ToTensor()(img)

    def __len__(self):
        return len(self.image_paths)


# Hyperparameters
BATCH_SIZE = 4
PATCH_CROP_SIZE = 96
UPSCALE_FACTOR = 2
LEARNING_RATE_GENERATOR = 1e-2
LEARNING_RATE_DISCRIMINATOR = 1e-4
GAMMA = 0.8
SCALE_FACTORS = [round(1.1 + 0.1 * i, 1) for i in range(30)]  # [1.1, 1.2, ..., 4.0]

# --- Transform Factories ---
def get_random_crop(crop_size):
    return transforms.RandomCrop(crop_size)

def get_center_crop(crop_size):
    return transforms.CenterCrop(crop_size)

def get_input_transform(crop_size, scale):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(int(crop_size // scale)),
        transforms.ToTensor()
    ])

def get_target_transform(crop_size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(crop_size),
        transforms.ToTensor()
    ])

# --- Data Processing ---
def process_batch(batch):
    inputs, targets = batch
    scale = np.random.choice(SCALE_FACTORS)
    N, C, H, W = inputs.shape
    crop_size = H

    inp_tf = get_input_transform(crop_size, scale)
    tgt_tf = get_target_transform(int(int(crop_size // scale) * scale))

    inputs_processed = [inp_tf(img) for img in inputs]
    targets_processed = [tgt_tf(img) for img in targets]

    return torch.stack(inputs_processed), torch.stack(targets_processed), scale

# --- Residual Block ---
class ResidualBlockSRGAN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

# --- Weight Prediction Network ---
class WeightPredictionNetwork(nn.Module):
    def __init__(self, in_features, out_channels, kernel_size=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, in_features * out_channels * kernel_size * kernel_size)
        )

    def forward(self, v):
        return self.mlp(v)

# --- Meta Upscale Module ---
class MetaUpscaleModule(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, kernel_size=3):
        super().__init__()
        self.scale_factor = scale_factor
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_pred_net = WeightPredictionNetwork(
            in_features=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size
        )

    def update_scale(self, scale):
        self.scale_factor = scale

    def add_dimensions(self, x):
        scale_int = math.ceil(self.scale_factor)
        N, C, H, W = x.size()
        x = x.view(N, C, H, 1, W, 1)
        x = x.expand(-1, -1, -1, scale_int, -1, scale_int)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(-1, C, H, W)

    def forward(self, F_LR, v):
        N, C, inH, inW = F_LR.shape
        W_pred = self.weight_pred_net(v)
        scale_int = math.ceil(self.scale_factor)

        F_LR_ext = self.add_dimensions(F_LR)
        unfolded = nn.functional.unfold(F_LR_ext, kernel_size=self.kernel_size, padding=1)
        F_LR_final = unfolded.view(N, scale_int * scale_int, -1, inH * inW, 1)
        F_LR_final = F_LR_final.permute(0, 1, 3, 4, 2)

        W = W_pred.view(inH, scale_int, inW, scale_int, -1, self.out_channels)
        W = W.permute(1, 3, 0, 2, 4, 5).contiguous()
        W = W.view(scale_int * scale_int, inH * inW, -1, self.out_channels)

        out = torch.matmul(F_LR_final, W)
        out = out.permute(0, 1, 4, 2, 3).contiguous()
        out = out.view(N, scale_int, scale_int, self.out_channels, inH, inW)
        out = out.permute(0, 3, 4, 1, 5, 2).contiguous()
        out = out.view(N, self.out_channels, scale_int * inH, scale_int * inW)

        mask = create_mask(inH, inW, self.scale_factor).to(out.device)
        out = out.masked_select(mask).view(
            N,
            self.out_channels,
            int(self.scale_factor * inH),
            int(self.scale_factor * inW)
        )

        return out

# --- Mask & V Creation ---
def create_v(inH, inW, scale):
    scale_int = math.ceil(scale)
    return torch.ones(int(inH * scale_int) * int(inW * scale_int), 3)


def create_mask(inH, inW, scale):
    scale_int = math.ceil(scale)
    outH, outW = int(inH * scale), int(inW * scale)

    mask_h = torch.zeros(inH, scale_int, 1)
    h_proj = torch.floor(torch.arange(outH).float().div(scale)).int()
    for idx, h in enumerate(h_proj):
        flag = (mask_h[h] == 0).nonzero(as_tuple=False)[0][1]
        mask_h[h, flag, 0] = 1

    mask_w = torch.zeros(1, inW, scale_int)
    w_proj = torch.floor(torch.arange(outW).float().div(scale)).int()
    for idx, w in enumerate(w_proj):
        flag = (mask_w[0, w] == 0).nonzero(as_tuple=False)[0][2]
        mask_w[0, w, flag] = 1

    mask_h = mask_h.repeat(1, inW * scale_int).view(-1, scale_int * inW, 1)
    mask_w = mask_w.repeat(inH * scale_int, 1).view(-1, scale_int * inW, 1)
    mask = (mask_h + mask_w).eq(2).view(scale_int * inH, scale_int * inW)
    return mask

# --- Generator ---
class Generator(nn.Module):
    def __init__(self, scale_factor, in_channels=1, out_channels=1, num_res_blocks=5):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.residuals = nn.Sequential(*[ResidualBlockSRGAN(64) for _ in range(num_res_blocks)])
        self.conv_after_res = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.upscale = MetaUpscaleModule(64, out_channels, scale_factor)

    def update_scale(self, scale):
        self.upscale.update_scale(scale)

    def forward(self, x, v):
        feat = self.initial(x)
        res = self.residuals(feat)
        merged = feat + self.conv_after_res(res)
        return self.upscale(merged, v)

# --- Discriminator ---
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=False)
        ]
        for i in range(3):
            in_c = 64 * (1 << i)
            out_c = 64 * (1 << (i + 1))
            layers += [
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(out_c, out_c, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=False)
            ]
        layers += [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(1024, 1, kernel_size=1)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.model(x)
        out = out.view(batch_size, -1)
        out = out.detach() + (out - out.detach())
        return torch.sigmoid(out)

# --- Data Loaders & Model Setup ---
train_loader = DataLoader(
    BrainTumorDataset(train_dir, num_patches=16, crop_size=PATCH_CROP_SIZE),
    batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    BrainTumorTestDataset(test_dir, crop_size=PATCH_CROP_SIZE),
    batch_size=1, shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator(UPSCALE_FACTOR, in_channels=1, out_channels=1).to(device)
netD = Discriminator(in_channels=1).to(device)

optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE_GENERATOR)
optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE_DISCRIMINATOR)

schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=20, gamma=GAMMA)
schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=20, gamma=GAMMA)

print(f"Generator params: {sum(p.numel() for p in netG.parameters())}")
print(f"Discriminator params: {sum(p.numel() for p in netD.parameters())}")

# --- Training Loop ---
def train_srgan(epoch, loader=train_loader):
    netG.train(); netD.train()
    running = {'g': 0.0, 'd': 0.0}
    iter_losses = {'g': [], 'd': []}

    for idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        inputs, targets, scale = process_batch(batch)
        netG.update_scale(scale)
        inputs, targets = inputs.to(device), targets.to(device)
        v = create_v(inputs.shape[-2], inputs.shape[-1], scale).to(device)

        # Discriminator step
        fake = netG(inputs, v)
        real_out = netD(targets).mean()
        fake_out = netD(fake).mean()
        optimizerD.zero_grad()
        d_loss = 1 - real_out + fake_out
        d_loss.backward()
        optimizerD.step()

        # Generator step
        optimizerG.zero_grad()
        fake = netG(inputs, v)
        fake_det = netD(fake).mean().detach()
        adv_loss = (1 - fake_det).mean()
        l1_loss = nn.L1Loss()(fake, targets)
        g_loss = l1_loss + 0.001 * adv_loss
        g_loss.backward()
        optimizerG.step()

        iter_losses['g'].append(g_loss.item())
        iter_losses['d'].append(d_loss.item())
        running['g'] += g_loss.item()
        running['d'] += d_loss.item()

        if idx % 100 == 0:
            print(f"[{idx}] G_loss: {g_loss:.4f} | D_loss: {d_loss:.4f}")

    avg_g = running['g'] / len(loader)
    avg_d = running['d'] / len(loader)
    print(f"===> Epoch {epoch} Complete: Avg G_loss: {avg_g:.4f}, Avg D_loss: {avg_d:.4f}")
    return avg_g, avg_d, iter_losses['g'], iter_losses['d']

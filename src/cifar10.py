import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
from tqdm import tqdm
import os
import wandb
from src.discrete_diffusion import DiscreteDiffusion, FLDD, DummyX0Model, Uniform
from src.dit import DiT_Llama

if __name__ == "__main__":
    # 1. Initialize WandB
    NAME="FLDDTest"
    wandb.init(project="fldd_cifar10", name=NAME)
    EXP_DIR = f"runs/feb/8/cifar10/{NAME}"

    # 2. Hyperparameters
    N = 8          # Quantization levels (classes per pixel)
    T = 1000       # Timesteps
    C = 3          # CIFAR-10 Channels
    LR = 2e-5      # Learning rate (adjusted for FLDD typically)
    BATCH_SIZE = 64
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 3. Model Setup for FLDD
    # We need two networks: one for the denoising task (x0_model) 
    # and one for the learnable forward process (fldd_net).
    
    print("Initializing models...")
    
    # The Forward Process itself
    # forward_proc = Uniform(num_classes=N, num_timesteps=T)
    
    # forward_proc = Masking(num_classes=N, num_timesteps=T)

    fldd_net = DiT_Llama(C, N, dim=1024)
    forward_proc = FLDD(num_classes=N, num_timesteps=T, forward_net=fldd_net)
    
    # The Denoising Model p(x_{t-1} | x_t)
    x0_model = DiT_Llama(C, N, dim=1024)

    # Wrapper
    model = DiscreteDiffusion(
        x0_model=x0_model,
        num_timesteps=T,
        num_classes=N,
        forward_process=forward_proc,
        hybrid_loss_coeff=0.01 # Typical value for D3PM/FLDD
    ).to(device)
    
    # 4. Optimizer Setup
    # CRITICAL FOR FLDD: We must optimize both the x0_model AND the forward_net
    params = list(model.x0_model.parameters())
    if isinstance(forward_proc, FLDD):
        params += list(forward_proc.forward_net.parameters())
        model.forward_process.forward_net.cuda()
    
    optim = torch.optim.AdamW(params, lr=LR)
    
    print(f"Total Param Count: {sum([p.numel() for p in params])}")

    # 5. Data Loading (CIFAR-10)
    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # Note: We do NOT normalize to [-1, 1] here because the model 
            # handles discretization internally based on [0, 1] floats.
        ]),
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)

    # 6. Training Loop
    n_epoch = 100
    global_step = 0
    
    # Create directory for saving local GIF results
    os.makedirs(f"{EXP_DIR}/results", exist_ok=True)

    model.train()
    
    for i in range(n_epoch):
        pbar = tqdm(dataloader)
        loss_ema = None
        
        for x, cond in pbar:
            optim.zero_grad()
            x = x.to(device)
            cond = cond.to(device)

            # Discretize x to N bins (0 to N-1)
            # Your model expects integers for ground truth or [0,1] floats 
            # that get mapped. Here we discretize explicitly to integer indices.
            x_cat = (x * (N - 1)).round().long().clamp(0, N - 1)
            
            # Forward pass
            # IMPORTANT: Pass global_step so FLDD can manage the Warmup/REINFORCE schedule
            loss, info = model(x_cat, cond, global_step=global_step)

            loss.backward()
            
            # Clip gradients
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            
            # Param norm metric
            with torch.no_grad():
                param_norm = sum([torch.norm(p) for p in params])

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.99 * loss_ema + 0.01 * loss.item()

            # Logging
            if global_step % 10 == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "train_grad_norm": norm,
                    "train_param_norm": param_norm,
                    "vb_loss": info['vb_loss'],
                    "ce_loss": info['ce_loss'],
                    "aux_loss": info['aux_loss'],
                    "fldd_tau": model.forward_process.current_tau if isinstance(model.forward_process, FLDD) else 0,
                })
                
            pbar.set_description(
                f"loss: {loss_ema:.4f}, vb: {info['vb_loss']:.4f}, aux: {info['aux_loss']:.4f}"
            )
            
            optim.step()
            global_step += 1

            # 7. Sampling & Evaluation
            if global_step % 600 == 1:
                model.eval()
                print("\nGenerating samples...")
                
                with torch.no_grad():
                    # CIFAR-10 has 10 classes. We'll generate 16 images.
                    cond_sample = torch.arange(0, 16).to(device) % 10
                    
                    # Initial noise depends on input_num_classes (N for Uniform/FLDD, N+1 for Masking)
                    init_dim = model.forward_process.input_num_classes
                    init_noise = torch.randint(0, init_dim, (16, C, 32, 32)).to(device)

                    # Use your model's sampling method
                    # Stride controls how many intermediate steps we keep for the GIF
                    images = model.sample_with_image_sequence(
                        init_noise, cond_sample, stride=50 
                    )
                    
                    # Process images for GIF
                    gif = []
                    for image in images:
                        # Grab a batch of real data to compare side-by-side
                        x_real_viz = x_cat[:16].cpu().float() / (N - 1)
                        x_gen_viz = image.float().cpu().clamp(0, N-1) / (N - 1)
                        
                        # Concatenate real (top) and generated (bottom)
                        all_images = torch.cat([x_real_viz, x_gen_viz], dim=0)
                        
                        x_as_image = make_grid(all_images, nrow=4)
                        img = x_as_image.permute(1, 2, 0).cpu().numpy()
                        img = (img * 255).astype(np.uint8)
                        gif.append(Image.fromarray(img))

                    # Save GIF locally
                    gif_path = f"{EXP_DIR}/results/sample_{global_step}.gif"
                    gif[0].save(
                        gif_path,
                        save_all=True,
                        append_images=gif[1:],
                        duration=100,
                        loop=0,
                    )

                    last_img = gif[-1]
                    last_img.save(f"{EXP_DIR}/results/sample_{global_step}_last.png")

                    # Log to WandB
                    wandb.log({
                        "sample_gif": wandb.Video(gif_path, fps=10, format="gif"),
                        "sample_last": wandb.Image(last_img),
                    })

                model.train()
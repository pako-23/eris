"""
DLG/iDLG Evaluation with Model Partitioning.

This script evaluates the Deep Leakage from Gradients (DLG) and improved DLG (iDLG) attacks under different sparsity conditions.
It supports MNIST, CIFAR10, CIFAR100, and LFW datasets, and includes:
- LeNet model initialization and gradient computation
- Simulated gradient pruning via random subset selection (controlled by `splits`=Number of aggregators)
- Random reconstructions for baseline comparison
- Image reconstruction using DLG and iDLG
- Evaluation using MSE, PSNR, SSIM, and LPIPS metrics
- Results saving and metric aggregation into Excel files

Optional components include:
- Download and processing of the LFW dataset
- Dataset wrapping for LFW image structure
- LPIPS perceptual similarity metric
- Gradient sparsity logging
"""

import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F 
from torchvision import datasets, transforms
import pickle
import PIL.Image as Image

import sys
import urllib.request
import tarfile

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from public import utils
import lpips

def download_and_extract_lfw(data_dir='../data'):
    """
    Downloads and extracts the LFW dataset into the specified directory.

    Args:
        data_dir (str): The directory where the dataset will be stored.
    """
    lfw_url = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'
    lfw_tar = os.path.join(data_dir, 'lfw.tgz')
    lfw_folder = os.path.join(data_dir, 'lfw')

    if not os.path.exists(lfw_folder):
        os.makedirs(data_dir, exist_ok=True)

        if not os.path.exists(lfw_tar):
            print("Downloading LFW dataset...")
            try:
                urllib.request.urlretrieve(lfw_url, lfw_tar)
                print("Download complete.")
            except Exception as e:
                print(f"Failed to download LFW dataset. Error: {e}")
                return

        print("Extracting LFW dataset...")
        try:
            with tarfile.open(lfw_tar, 'r:gz') as tar:
                tar.extractall(path=data_dir)
            print("Extraction complete.")
        except Exception as e:
            print(f"Failed to extract LFW dataset. Error: {e}")
    else:
        print("LFW dataset already exists. Skipping download and extraction.")

class LeNet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def concatenate_weights(weights_list, n_splits=0, random_seed=1):
    """
    Concatenates a flat list of weight matrices into a single vector and stores their original shapes.
    Optionally zeroes out all but a randomly selected chunk of weights based on the number of splits.
    
    Parameters:
    - weights_list: List of weight matrices (numpy arrays).
    - n_splits: Number of chunks to divide the concatenated weights into. 
                If greater than 0, a single chunk_size of weights is randomly kept, and the rest are zeroed out.
    - random_seed: (Optional) Integer seed for reproducibility of the random sampling.
    
    Returns:
    - concatenated_weights: 1D numpy array of all concatenated weights, with some elements possibly zeroed out.
    - shapes: List of original shapes of each weight matrix for reconstruction.
    """
    flattened_weights = []
    shapes = []

    # Flatten each weight matrix and store its shape
    for weight_matrix in weights_list:
        flattened_weights.append(weight_matrix.flatten())
        shapes.append(weight_matrix.shape)

    # Concatenate all flattened weights into a single vector
    concatenated_weights = np.concatenate(flattened_weights)
    
    # Zero out parameters if n_splits is specified
    if n_splits > 0:
        total_length = len(concatenated_weights)
        chunk_size = int(total_length // n_splits)
        
        if chunk_size == 0:
            raise ValueError("n_splits is too large, resulting in chunk_size=0.")
        
        # Randomly select chunk_size unique indices to keep
        np.random.seed(random_seed)
        keep_indices = np.random.choice(total_length, size=chunk_size, replace=False)
        
        # Debug statements (optional)
        # print(f"Total length of concatenated weights: {total_length}")
        # print(f"Chunk size (number of weights to keep): {chunk_size}")
        # print(f"Indices to keep: {keep_indices}")
        
        # Create a new concatenated_weights vector with zeros
        concatenated_weights_new = np.zeros_like(concatenated_weights)
        concatenated_weights_new[keep_indices] = concatenated_weights[keep_indices]
        
        return concatenated_weights_new, shapes
    
    return concatenated_weights, shapes

def deconcatenate_weights(flat_vector, shapes):
    """
    Reconstructs the list of weight matrices from the flat concatenated vector based on the provided shapes.
    
    Parameters:
    - flat_vector: 1D numpy array of concatenated weights.
    - shapes: List of shapes for each weight matrix.
    
    Returns:
    - reconstructed_weights: List of weight matrices with their original shapes.
    """
    reconstructed_weights = []
    idx = 0

    for shape in shapes:
        size = np.prod(shape)
        weight_matrix = flat_vector[idx:idx + size].reshape(shape)
        reconstructed_weights.append(weight_matrix)
        idx += size

    return reconstructed_weights

def update_model_parameters(gradients, new_weights):
    """
    Updates the model's gradients with the provided new_weights.

    Parameters:
    - model: The PyTorch model whose parameters are to be updated.
    - new_weights: A list of numpy arrays representing the new weights.
    """
    with torch.no_grad():
        for param, new_weight in zip(gradients, new_weights):
            # Ensure the new weight is a numpy array
            if not isinstance(new_weight, np.ndarray):
                raise TypeError("All elements in new_weights must be numpy arrays.")
            
            # Convert numpy array to torch tensor
            new_weight_tensor = torch.from_numpy(new_weight).to(param.device).type_as(param)
            
            # Ensure the shape matches
            if param.cpu().data.shape != new_weight_tensor.shape:
                raise ValueError(f"Shape mismatch: Parameter shape {param.cpu().data.shape} vs new weight shape {new_weight_tensor.shape}")
            
            # Copy the data
            param.copy_(new_weight_tensor)

def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())

class Dataset_from_Image(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs # img paths
        self.labs = labs # labs is ndarray
        self.transform = transform
        del imgs, labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        lab = self.labs[idx]
        img = Image.open(self.imgs[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, lab

def lfw_dataset(lfw_path, shape_img):
    images_all = []
    labels_all = []
    folders = os.listdir(lfw_path)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(lfw_path, fold))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(lfw_path, fold, f))
                labels_all.append(foldidx)

    transform = transforms.Compose([transforms.Resize(size=shape_img)])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst

def count_zero_nonzero(original_dy_dx):
    """
    Counts the total number of zero and non-zero elements in the list of tensors.

    Parameters:
    - original_dy_dx: List of PyTorch tensors.

    Returns:
    - total_zero: Total number of elements equal to zero.
    - total_non_zero: Total number of elements not equal to zero.
    """
    total_zero = 0
    total_non_zero = 0

    for param in original_dy_dx:
        # Ensure the tensor is on CPU and convert to NumPy array
        data = param.cpu().data.numpy()

        # Count zeros and non-zeros
        total_zero += np.sum(data == 0)
        total_non_zero += np.sum(data != 0)

    print(f"Total number of zero elements: {total_zero}")
    print(f"Total number of non-zero elements: {total_non_zero}")

    return total_zero, total_non_zero

def process_and_save_metrics(best_metrics, method_name, result_path='results'):
    """
    Processes the best_metrics dictionary for a specific method, computes mean and std for each metric,
    and saves the results into an Excel file.
    
    Parameters:
    - best_metrics (dict): Dictionary containing metrics for each experiment.
    - method_name (str): The name of the method ('DLG' or 'iDLG').
    - result_path (str): Directory path to save the Excel file.
    """
    # Convert best_metrics dictionary to DataFrame
    df_metrics = pd.DataFrame.from_dict(best_metrics, orient='index')
    
    # Reset index to turn the experiment index into a column
    df_metrics.reset_index(inplace=True)
    
    # Rename the 'index' column to 'experiment' for clarity
    df_metrics.rename(columns={'index': 'experiment'}, inplace=True)
    
    # Display the DataFrame structure (optional)
    print(f"\nAll Metrics for {method_name}:")
    print(df_metrics.head())
    
    # Compute mean
    avg_metrics = df_metrics.mean(numeric_only=True)
    
    # Compute standard deviation
    std_metrics = df_metrics.std(numeric_only=True)
    
    # Create result dictionary with mean and std
    result = {
        'best_loss_mean': avg_metrics.get('best_loss', None),
        'best_loss_std': std_metrics.get('best_loss', None),
        'mse_mean': avg_metrics.get('mse', None),
        'mse_std': std_metrics.get('mse', None),
        'psnr_mean': avg_metrics.get('psnr', None),
        'psnr_std': std_metrics.get('psnr', None),
        'ssim_mean': avg_metrics.get('ssim', None),
        'ssim_std': std_metrics.get('ssim', None),
        'lpips_mean': avg_metrics.get('lpips', None),
        'lpips_std': std_metrics.get('lpips', None)
    }
    
    # Convert result dictionary to DataFrame
    df_result = pd.DataFrame([result])
    
    # Define the output Excel file path
    excel_filename = f'metrics_{method_name}.xlsx'
    excel_path = os.path.join(result_path, 'final_metrics', excel_filename)
    
    # Ensure the result directory exists
    os.makedirs(os.path.join(result_path, 'final_metrics'), exist_ok=True)
    
    # Save the DataFrame to Excel
    df_result.to_excel(excel_path, index=False)
    
    print(f"Mean and Std metrics for {method_name} saved to {excel_path}")


def main(dataset='MNIST', splits=1):

    eris = True
    seed = 1

    lr = 1.0
    num_dummy = 1
    Iteration = 300
    num_exp = 200
    
    root_path = '.'
    data_path = os.path.join(root_path, 'data').replace('\\', '/')
    flag = True
    
    utils.set_seed(seed)
    use_cuda = torch.cuda.is_available()
    device = 'cuda:2' if use_cuda else 'cpu'

    # Ensure the LFW dataset is downloaded and extracted
    if dataset == 'lfw':
        download_and_extract_lfw(data_dir=os.path.join(root_path, 'data'))

    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])
    
    # Choose the desired backbone: 'alex', 'vgg', or 'squeeze'
    lpips_model = lpips.LPIPS(net='alex') 
    lpips_model = lpips_model.to(device) 
    lpips_model.eval() 

    ''' load data '''
    if dataset == 'MNIST':
        shape_img = (28, 28)
        num_classes = 10
        channel = 1
        hidden = 588
        dst = datasets.MNIST(data_path, download=True)

    elif dataset == 'cifar100':
        shape_img = (32, 32)
        num_classes = 100
        channel = 3
        hidden = 768
        dst = datasets.CIFAR100(data_path, download=True)

    elif dataset == 'cifar10':
        shape_img = (32, 32)
        num_classes = 10
        channel = 3
        hidden = 768
        dst = datasets.CIFAR10(data_path, download=True)

    elif dataset == 'lfw':
        shape_img = (32, 32)
        num_classes = 5749
        channel = 3
        hidden = 768
        lfw_path = os.path.join(root_path, 'data/lfw')
        dst = lfw_dataset(lfw_path, shape_img)

    else:
        exit('unknown dataset')



    ''' train DLG and iDLG '''
    best_metrics_DLG = {}
    best_metrics_iDLG = {}
    for idx_net in range(num_exp):
        net = LeNet(channel=channel, hideen=hidden, num_classes=num_classes)
        net.apply(weights_init)

        print('running %d|%d experiment'%(idx_net, num_exp))
        net = net.to(device)
        # idx_shuffle = np.random.permutation(len(dst))

        for method in ['DLG', 'iDLG']:
            print('\n%s, Try to generate %d images' % (method, num_dummy))

            criterion = nn.CrossEntropyLoss().to(device)
            imidx_list = []

            for imidx in range(num_dummy):
                # idx = idx_shuffle[imidx]
                idx = idx_net
                imidx_list.append(idx)
                tmp_datum = tt(dst[idx][0]).float().to(device)
                tmp_datum = tmp_datum.view(1, *tmp_datum.size())
                tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
                tmp_label = tmp_label.view(1, )
                if imidx == 0:
                    gt_data = tmp_datum
                    gt_label = tmp_label
                else:
                    gt_data = torch.cat((gt_data, tmp_datum), dim=0)
                    gt_label = torch.cat((gt_label, tmp_label), dim=0)

            # no optimization 
            if splits == 'random':
                # path
                image_path = os.path.join(root_path, f'images/{dataset}/{'random'}_E{num_exp}_I{Iteration}')
                result_path = os.path.join(root_path, f'results/{dataset}/{'random'}_E{num_exp}_I{Iteration}')

                os.makedirs(image_path, exist_ok=True)
                os.makedirs(result_path, exist_ok=True)
                
                print(dataset, 'root_path:', root_path)
                print(dataset, 'data_path:', data_path)
                print(dataset, 'image_path:', image_path)
                
                # Generate a random reconstructed image
                dummy_data = torch.rand(gt_data.size()).to(device).requires_grad_(True)
                dummy_label = torch.rand((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)

                # save random image
                for imidx in range(num_dummy):
                    
                    img = tp(dummy_data[imidx].cpu())
                    img_filename = f'best_recon_{method}_exp{idx_net}_img{imidx}.png'
                    img.save(os.path.join(result_path, img_filename))
                
                # Directly calculate metrics for the random reconstruction
                mse = torch.mean((dummy_data - gt_data) ** 2).item()

                # Min-Max Normalization for dummy_data to [0, 1]
                dummy_min = dummy_data.view(dummy_data.size(0), -1).min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
                dummy_max = dummy_data.view(dummy_data.size(0), -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
                dummy_data_normalized = (dummy_data - dummy_min) / (dummy_max - dummy_min + 1e-8)

                # Scale normalized data to [-1, 1] for LPIPS
                recon_lpips = dummy_data_normalized * 2 - 1
                gt_lpips = gt_data * 2 - 1  # Assuming gt_data is already in [0, 1]

                # Repeat channel for MNIST dataset and resize for LPIPS
                if dataset == 'MNIST':
                    if recon_lpips.shape[1] == 1:
                        recon_lpips = recon_lpips.repeat(1, 3, 1, 1)
                    if gt_lpips.shape[1] == 1:
                        gt_lpips = gt_lpips.repeat(1, 3, 1, 1)
                    target_size = (32, 32)  # Resize for LPIPS
                    recon_lpips = F.interpolate(recon_lpips, size=target_size, mode='bilinear', align_corners=False)
                    gt_lpips = F.interpolate(gt_lpips, size=target_size, mode='bilinear', align_corners=False)

                # Compute LPIPS
                recon_lpips = recon_lpips.to(device)
                gt_lpips = gt_lpips.to(device)
                with torch.no_grad():
                    lpips_values = lpips_model(recon_lpips, gt_lpips)
                    lpips_total = lpips_values.sum().item()

                # Convert tensors to NumPy arrays for PSNR and SSIM calculations
                gt_np = gt_data.detach().cpu().numpy()
                recon_np = dummy_data_normalized.detach().cpu().numpy()

                psnr_total = 0
                ssim_total = 0
                for imidx in range(num_dummy):
                    gt_img = gt_np[imidx]
                    recon_img = recon_np[imidx]

                    if gt_img.shape[0] == 1:
                        gt_img = gt_img.squeeze(0)
                        recon_img = recon_img.squeeze(0)
                        multichannel = False
                    elif gt_img.shape[0] == 3:
                        gt_img = np.transpose(gt_img, (1, 2, 0))
                        recon_img = np.transpose(recon_img, (1, 2, 0))
                        multichannel = True

                    psnr_val = compare_psnr(gt_img, recon_img, data_range=1.0)
                    ssim_val = compare_ssim(gt_img, recon_img, channel_axis=-1, data_range=1.0) if multichannel else compare_ssim(gt_img, recon_img, data_range=1.0)

                    psnr_total += psnr_val
                    ssim_total += ssim_val

                avg_psnr = psnr_total / num_dummy
                avg_ssim = ssim_total / num_dummy
                avg_lpips = lpips_total / num_dummy

                # Log or save the metrics
                print(f"Random Reconstruction Metrics: MSE = {mse:.4f}, PSNR = {avg_psnr:.4f}, SSIM = {avg_ssim:.4f}, LPIPS = {avg_lpips:.4f}")
            
                # Update best_metrics
                if method == 'DLG':
                    best_metrics_DLG[idx_net] = {
                            'best_loss': 0,
                            'mse': mse,
                            'psnr': avg_psnr,
                            'ssim': avg_ssim,
                            'lpips': avg_lpips
                        } 
                elif method == 'iDLG':
                    best_metrics_iDLG[idx_net] = {
                            'best_loss': 0,
                            'mse': mse,
                            'psnr': avg_psnr,
                            'ssim': avg_ssim,
                            'lpips': avg_lpips
                        } 
                else:
                    raise KeyError  
            
            else:
                # compute original gradient
                out = net(gt_data)
                y = criterion(out, gt_label)
                dy_dx = torch.autograd.grad(y, net.parameters())
                original_dy_dx = list((_.detach().clone() for _ in dy_dx))

                if eris:
                    print(f"Evaluate only 1 split")
                    gradient_list = []
                    for param in original_dy_dx:
                        gradient_list.append(param.cpu().data.numpy())

                    # Flat and select only one split, zeroing out the others
                    w, s = concatenate_weights(gradient_list, n_splits=splits, random_seed=seed)
                    r = deconcatenate_weights(w,s)

                    # update gradient
                    update_model_parameters(original_dy_dx, r)
                    total_zero, total_non_zero = count_zero_nonzero(original_dy_dx)
                else:
                    total_zero = 0
                    total_non_zero = sum(param.numel() for param in original_dy_dx)


                if flag:
                    flag = False
                    image_path = os.path.join(root_path, f'images/{dataset}/{total_zero}_{total_non_zero}_E{num_exp}_I{Iteration}')
                    result_path = os.path.join(root_path, f'results/{dataset}/{total_zero}_{total_non_zero}_E{num_exp}_I{Iteration}')

                    os.makedirs(image_path, exist_ok=True)
                    os.makedirs(result_path, exist_ok=True)
                    
                    print(dataset, 'root_path:', root_path)
                    print(dataset, 'data_path:', data_path)
                    print(dataset, 'image_path:', image_path)
                
                
                # generate dummy data and label
                dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
                dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)

                if method == 'DLG':
                    optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
                elif method == 'iDLG':
                    optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
                    # predict the ground-truth label
                    label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)

                history = []
                history_iters = []
                losses = []
                mses = []
                train_iters = []
                best_loss = 100000
                print('lr =', lr)
                for iters in range(Iteration):

                    def closure():
                        optimizer.zero_grad()
                        pred = net(dummy_data)
                        if method == 'DLG':
                            dummy_loss = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                            # dummy_loss = criterion(pred, gt_label)
                        elif method == 'iDLG':
                            dummy_loss = criterion(pred, label_pred)

                        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                        grad_diff = 0
                        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                            grad_diff += ((gx - gy) ** 2).sum()
                        grad_diff.backward()
                        return grad_diff

                    optimizer.step(closure)
                    current_loss = closure().item()
                    train_iters.append(iters)
                    losses.append(current_loss)
                    mses.append(torch.mean((dummy_data-gt_data)**2).item())
                    
                    # save best 
                    if current_loss < best_loss:
                        best_loss = current_loss
                        # Save best reconstructed image in result_path
                        for imidx in range(num_dummy):
                            
                            img = tp(dummy_data[imidx].cpu())
                            img_filename = f'best_recon_{method}_exp{idx_net}_img{imidx}.png'
                            img.save(os.path.join(result_path, img_filename))
                            
                        # Calculate typical metrics to evaluate the quality of the reconstruction
                        mse = torch.mean((dummy_data - gt_data) ** 2).item()
                        
                        # Calculate LPIPS
                        # Min-Max Normalization for dummy_data to [0, 1]
                        # Perform per-image min-max scaling
                        dummy_min = dummy_data.view(dummy_data.size(0), -1).min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
                        dummy_max = dummy_data.view(dummy_data.size(0), -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
                        dummy_data_normalized = (dummy_data - dummy_min) / (dummy_max - dummy_min + 1e-8)
                        
                        # Scale normalized data to [-1, 1] for LPIPS
                        recon_lpips = dummy_data_normalized * 2 - 1
                        gt_lpips = gt_data * 2 - 1  # Assuming gt_data is already in [0, 1]
        
                        # Convert 1-channel to 3-channel by repeating the channel
                        if dataset == 'MNIST':
                            if recon_lpips.shape[1] == 1:
                                recon_lpips = recon_lpips.repeat(1, 3, 1, 1)  # Shape: (batch_size, 3, H, W)
                            if gt_lpips.shape[1] == 1:
                                gt_lpips = gt_lpips.repeat(1, 3, 1, 1)        # Shape: (batch_size, 3, H, W)
                            
                            # Resize images to 64x64 for LPIPS
                            target_size = (32, 32)  # You can choose 64 or 128 based on your preference
                            recon_lpips = F.interpolate(recon_lpips, size=target_size, mode='bilinear', align_corners=False)
                            gt_lpips = F.interpolate(gt_lpips, size=target_size, mode='bilinear', align_corners=False)
                        
                        # Ensure recon_lpips and gt_lpips are on the same device as lpips_model
                        recon_lpips = recon_lpips.to(device)
                        gt_lpips = gt_lpips.to(device)
                        
                        # Compute LPIPS
                        with torch.no_grad():
                            lpips_values = lpips_model(recon_lpips, gt_lpips)
                            lpips_total = lpips_values.sum().item()
                        
                        # Convert tensors to NumPy arrays for PSNR SSIM calculations
                        gt_np = gt_data.detach().cpu().numpy()
                        recon_np = dummy_data_normalized.detach().cpu().numpy()

                        # Initialize metric accumulators
                        psnr_total = 0
                        ssim_total = 0

                        for imidx in range(num_dummy):
                            gt_img = gt_np[imidx]
                            recon_img = recon_np[imidx]
                            
                            # Handle channels
                            if gt_img.shape[0] == 1:
                                gt_img = gt_img.squeeze(0)  # (H, W)
                                recon_img = recon_img.squeeze(0)
                                multichannel = False
                            elif gt_img.shape[0] == 3:
                                gt_img = np.transpose(gt_img, (1, 2, 0))  # (H, W, C)
                                recon_img = np.transpose(recon_img, (1, 2, 0))
                                multichannel = True
                            else:
                                raise ValueError(f"Unsupported number of channels in ground truth image: {gt_img.shape[0]}")

                            # Compute psnr
                            psnr_val = compare_psnr(gt_img, recon_img, data_range=1.0)
                            
                            # Compute SSIM
                            if multichannel:
                                ssim_val = compare_ssim(gt_img, recon_img, channel_axis=-1, data_range=1.0)
                            else:
                                ssim_val = compare_ssim(gt_img, recon_img, data_range=1.0)
                        
                            psnr_total += psnr_val
                            ssim_total += ssim_val
                            
                            
                        avg_psnr = psnr_total / num_dummy
                        avg_ssim = ssim_total / num_dummy
                        avg_lpips = lpips_total / num_dummy

                        # Update best_metrics
                        if method == 'DLG':
                            best_metrics_DLG[idx_net] = {
                                    'best_loss': best_loss,
                                    'mse': mse,
                                    'psnr': avg_psnr,
                                    'ssim': avg_ssim,
                                    'lpips': avg_lpips
                                } 
                        elif method == 'iDLG':
                            best_metrics_iDLG[idx_net] = {
                                    'best_loss': best_loss,
                                    'mse': mse,
                                    'psnr': avg_psnr,
                                    'ssim': avg_ssim,
                                    'lpips': avg_lpips
                                } 
                        else:
                            raise KeyError                             


                    if iters % int(Iteration / 30) == 0:
                        current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                        print(current_time, iters, 'loss = %.8f, mse = %.8f' %(current_loss, mses[-1]))
                        history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                        history_iters.append(iters)

                        for imidx in range(num_dummy):
                            plt.figure(figsize=(12, 8))
                            plt.subplot(3, 10, 1)
                            plt.imshow(tp(gt_data[imidx].cpu()))
                            for i in range(min(len(history), 29)):
                                plt.subplot(3, 10, i + 2)
                                plt.imshow(history[i][imidx])
                                plt.title('iter=%d' % (history_iters[i]))
                                plt.axis('off')
                            if method == 'DLG':
                                plt.savefig('%s/DLG_on_%s_%05d.png' % (image_path, imidx_list, imidx_list[imidx]))
                                plt.close()
                            elif method == 'iDLG':
                                plt.savefig('%s/iDLG_on_%s_%05d.png' % (image_path, imidx_list, imidx_list[imidx]))
                                plt.close()

                        if current_loss < 0.000001: # converge
                            break

                if method == 'DLG':
                    loss_DLG = losses
                    label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
                    mse_DLG = mses
                elif method == 'iDLG':
                    loss_iDLG = losses
                    label_iDLG = label_pred.item()
                    mse_iDLG = mses

        if not isinstance(splits, str):
            print('imidx_list:', imidx_list)
            print('loss_DLG:', loss_DLG[-1], 'loss_iDLG:', loss_iDLG[-1])
            print('mse_DLG:', mse_DLG[-1], 'mse_iDLG:', mse_iDLG[-1])
            print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_DLG:', label_DLG, 'lab_iDLG:', label_iDLG)

        print('----------------------\n\n')


    # Average 
    process_and_save_metrics(best_metrics_DLG, f'DLG', result_path)
    process_and_save_metrics(best_metrics_iDLG, f'iDLG', result_path)
    print('   -------------------   \n\n\n')



    
if __name__ == '__main__':
    # # MNIST
    # main(dataset='MNIST', splits=1.0)
    # main(dataset='MNIST', splits=1.00001) # 1
    # main(dataset='MNIST', splits=1.0001) # 2
    # main(dataset='MNIST', splits=1.001) # 14
    # main(dataset='MNIST', splits=1.01) # 133
    # main(dataset='MNIST', splits=1.1) # 1221
    # main(dataset='MNIST', splits=1.2) # 2238
    # main(dataset='MNIST', splits=1.4) # 3836
    # main(dataset='MNIST', splits=2.0) # 6713
    # main(dataset='MNIST', splits=4.0) # 10070
    # main(dataset='MNIST', splits=8.0) # 11748
    # main(dataset='MNIST', splits=16.0) # 12587
    # main(dataset='MNIST', splits=32.0) # 13007  (only 419 real params)
    main(dataset='MNIST', splits='random')


    # # CIFAR10
    # main(dataset='cifar10', splits=1.0)
    # main(dataset='cifar10', splits=1.00001) # 1
    # main(dataset='cifar10', splits=1.0001) # 2
    # main(dataset='cifar10', splits=1.001) # 16
    # main(dataset='cifar10', splits=1.01) # 157
    # main(dataset='cifar10', splits=1.1) # 1439
    # main(dataset='cifar10', splits=1.2) # 2638
    # main(dataset='cifar10', splits=1.4) # 4522
    # main(dataset='cifar10', splits=2.0) # 7913
    # main(dataset='cifar10', splits=4.0) # 11870
    # main(dataset='cifar10', splits=8.0) # 13848
    # main(dataset='cifar10', splits=16.0) # 14837
    # main(dataset='cifar10', splits=32.0) # 15332  (only 494 real params)
    main(dataset='cifar10', splits='random')
    
    # LFW
    # main(dataset='lfw', splits=1.0) 
    # main(dataset='lfw', splits=1.0000001) # 1    
    # main(dataset='lfw', splits=1.000001) # 5
    # main(dataset='lfw', splits=1.00001) # 45
    # main(dataset='lfw', splits=1.0001) # 443
    # main(dataset='lfw', splits=1.001) # 4425
    # main(dataset='lfw', splits=1.01) # 43853
    # main(dataset='lfw', splits=1.1) # 402648
    # main(dataset='lfw', splits=1.2) # 738187
    # main(dataset='lfw', splits=1.4) # 1265462
    # main(dataset='lfw', splits=2.0) # 2214559
    # main(dataset='lfw', splits=4.0) # 3321838
    # main(dataset='lfw', splits=8.0) # 3875478
    # main(dataset='lfw', splits=16.0) # 4152298
    # main(dataset='lfw', splits=32.0) # 4290708  (only 138409 real params)
    main(dataset='lfw', splits='random')




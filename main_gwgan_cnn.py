#!/usr/bin/python
# author: Charlotte Bunne

# imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import os
import time 
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import models, transforms
from scipy.linalg import sqrtm
from PIL import Image

# internal imports
from model.utils import *
from model.model_cnn import Generator, Adversary
from model.model_cnn import weights_init_generator, weights_init_adversary
from model.loss import gwnorm_distance, loss_total_variation, loss_procrustes
from model.sgw_pytorch_original import sgw_gpu_original
from model.risgw_original import risgw_gpu_original
from model.rarisgw import rarisgw_gpu
from model.rasgw_pytorch import rasgw_gpu


# get arguments
args = get_args()

# system preferences
seed = np.random.randint(100)
torch.set_default_dtype(torch.double)
np.random.seed(seed)
torch.manual_seed(seed)

# settings
batch_size = 256
z_dim = 100
lr = 0.0002
ngen = 3
beta = args.beta
lam = 0.5
niter = 10
epsilon = 0.005
num_epochs = args.num_epochs
cuda = args.cuda
channels = args.n_channels
id = args.id

model = 'gwgan_{}_eps_{}_tv_{}_procrustes_{}_ngen_{}_channels_{}_{}' \
        .format(args.data, epsilon, lam, beta, ngen, channels, id)
save_fig_path = 'out_rasgw' + model
if not os.path.exists(save_fig_path):
    os.makedirs(save_fig_path)

# data import
if args.data == 'mnist':
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('.data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(32),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])),
        batch_size=batch_size, drop_last=True, shuffle=True)
# data import
elif args.data == 'mnist_':
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(32),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))])),
        batch_size=batch_size, drop_last=True, shuffle=True)
elif args.data == 'fmnist':
    dataloader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data/fmnist', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.Resize(32),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                       (0.5, 0.5, 0.5))])),
        batch_size=batch_size, drop_last=True, shuffle=True)
elif args.data == 'cifar_gray':
    dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data/cifar10', train=True, download=True,
                         transform=transforms.Compose([
                            # transform RGB to grayscale
                            transforms.Grayscale(num_output_channels=3),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,),
                                                 (0.5,))])),
        batch_size=batch_size, drop_last=True, shuffle=True)
elif args.data == 'cifar':
    dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data/cifar10', train=True, download=True,
                         transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                 (0.5, 0.5, 0.5))])),
        batch_size=batch_size, drop_last=True, shuffle=True)
else:
    raise NotImplementedError('dataset does not exist or not integrated.')

# print example images
save_image(next(iter(dataloader))[0][:25],
           os.path.join(save_fig_path, 'real.pdf'), nrow=5, normalize=True)

# define networks and parameters
generator = Generator(output_dim=channels)
adversary = Adversary(input_dim=channels)

# weight initialisation
generator.apply(weights_init_generator)
adversary.apply(weights_init_adversary)

if cuda:
    generator = generator.cuda()
    adversary = adversary.cuda()

# create optimizer
g_optimizer = torch.optim.Adam(generator.parameters(), lr, betas=(0.5, 0.99))
# zero gradients
generator.zero_grad()

c_optimizer = torch.optim.Adam(adversary.parameters(), lr, betas=(0.5, 0.99))
# zero gradients
adversary.zero_grad()


# Load the InceptionV3 model
inception_model = models.inception_v3(pretrained=True)
inception_model.eval()

# Define the transformation to resize and convert grayscale to RGB
resize_transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize images to 299x299
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
])

# Function to convert torch tensor to PIL Image
def tensor_to_pil_image(tensor):
    # Convert from tensor with shape (C, H, W) to PIL Image
    return transforms.ToPILImage()(tensor.cpu())

def calculate_fid(real_images, generated_images, device='cuda'):
    """
    Calculate FID score between real and generated images.
    
    Args:
        real_images (Tensor): A batch of real images (grayscale).
        generated_images (Tensor): A batch of generated images (grayscale).
        device (str): The device to run the model on ('cuda' or 'cpu').

    Returns:
        float: The FID score.
    """
    
    # Ensure model is on the right device
    inception_model.to(device)
    
    # Function to get features from InceptionV3 after transforming the images
    def get_features(images):
        # Apply the resize transformation to the images
        images_resized = []
        for img in images:
            pil_img = tensor_to_pil_image(img)  # Convert tensor to PIL image
            resized_img = resize_transform(pil_img)  # Apply resize and other transforms
            images_resized.append(resized_img)

        # Stack the images back into a batch
        images_resized = torch.stack(images_resized).to(device)

        # Forward pass through the InceptionV3 model (use the logits output)
        with torch.no_grad():
            features = inception_model(images_resized)

        return features

    # Get features for both real and generated images
    real_features = get_features(real_images)
    generated_features = get_features(generated_images)

    # Calculate the mean and covariance of real and generated features
    mu_real = torch.mean(real_features, dim=0)
    mu_generated = torch.mean(generated_features, dim=0)
    
    # Calculate the covariance matrices
    sigma_real = torch.cov(real_features.T)
    sigma_generated = torch.cov(generated_features.T)

    # Calculate the FID score
    fid_score = torch.norm(mu_real - mu_generated) ** 2 + torch.trace(sigma_real + sigma_generated - 2 * torch.sqrt(torch.mm(sigma_real, sigma_generated)))
    
    return fid_score.item()


# sample for plotting
num_test_samples = batch_size
z_ex = torch.randn(num_test_samples, z_dim)
if cuda:
    z_ex = z_ex.cuda()

loss_history = list()
loss_tv = list()
loss_orth = list()
loss_og = 0
is_hist = list()

fid_scores_last_epoch = []          # List to store FID scores for each iteration
reconstruction_losses_last_epoch = []          # To store the time taken for each epoch
epoch_times = []

for epoch in range(num_epochs):
    #t0 = time()
    epoch_start_time = time.time() 
    print("The epoch number now is:",epoch)
    for it, (image, _) in enumerate(dataloader):
        train_c = ((it + 1) % (ngen + 1) == 0)

        x = image.double()
        if cuda:
            x = x.cuda()

        # sample random number z from Z
        z = torch.randn(image.shape[0], z_dim)

        if cuda:
            z = z.cuda()

        if train_c:
            for q in generator.parameters():
                q.requires_grad = False
            for p in adversary.parameters():
                p.requires_grad = True
        else:
            for q in generator.parameters():
                q.requires_grad = True
            for p in adversary.parameters():
                p.requires_grad = False

        # result generator
        g = generator.forward(z)

        # result adversary
        f_x = adversary.forward(x)
        f_g = adversary.forward(g)

        if epoch == num_epochs-1:        #num_epochs - 1
                real_images = x  # Real images from the dataloader
                generated_images = g  # Generated images

                # Ensure that both real and generated images are in the range [0, 1]
                real_images = (real_images + 1) / 2  # Rescale to [0, 1]
                generated_images = (generated_images + 1) / 2  # Rescale to [0, 1]

                # If images have only 1 channel (grayscale), convert them to RGB
                if real_images.shape[1] == 1:  # If single-channel (grayscale)
                        real_images = real_images.repeat(1, 3, 1, 1)  # Repeat across the 3 channels
                if generated_images.shape[1] == 1:  # If single-channel (grayscale)
                        generated_images = generated_images.repeat(1, 3, 1, 1)  # Repeat across the 3 channels

                # Calculate FID score for this iteration
                #fid_score = calculate_fid(real_images, generated_images, device='cuda')

                # Store the FID score for this iteration
                #fid_scores_last_epoch.append(fid_score)
                reconstruction_loss = F.mse_loss(x, g)
                reconstruction_losses_last_epoch.append(reconstruction_loss.item())
                #print("fid score:",fid_score,"recon loss:",reconsruction_loss.item())

        # compute inner distances
        D_g = get_inner_distances(f_g, metric='euclidean', concat=False)
        D_x = get_inner_distances(f_x, metric='euclidean', concat=False)

        # distance matrix normalisation
        D_x_norm = normalise_matrices(D_x)
        D_g_norm = normalise_matrices(D_g)

        # compute normalized gromov-wasserstein distance
        #loss = gwnorm_distance((D_x, D_x_norm), (D_g, D_g_norm),epsilon, niter, loss_fun='square_loss',coupling=False, cuda=cuda)
        #loss = sgw_gpu_original(f_x.to('cuda'), f_g.to('cuda') ,'cuda',nproj=500,tolog=False,P=None)
        loss = rasgw_gpu(f_x.to('cuda'), f_g.to('cuda') ,'cuda',nproj=500,tolog=False,P=None)
        #loss = rarisgw_gpu(f_x.to('cuda'), f_g.to('cuda'),'cuda' ,nproj=500,P=None,lr=0.001, max_iter=20, verbose=False, step_verbose=10, tolog=False, retain_graph=True)
        #loss = risgw_gpu_original(f_x.to('cuda'), f_g.to('cuda') ,'cuda' ,nproj=500,P=None,lr=0.001, max_iter=20, verbose=False, step_verbose=10, tolog=False, retain_graph=True)
    

        if train_c:
            # train adversary
            loss_og = loss_procrustes(f_x, x.view(x.shape[0], -1), cuda)
            loss_to = -loss + beta * loss_og
            loss_to.backward()

            # parameter updates
            c_optimizer.step()
            # zero gradients
            reset_grad(generator, adversary)

        else:
            # train generator
            loss_t = loss_total_variation(g)
            loss_to = loss + lam * loss_t
            loss_to.backward()

            # parameter updates
            g_optimizer.step()
            # zero gradients
            reset_grad(generator, adversary)

    epoch_end_time = time.time()  # Record the end time of the epoch

    # Calculate the time taken for the entire epoch
    epoch_time = epoch_end_time - epoch_start_time

    # Store the epoch time in the list
    epoch_times.append(epoch_time)
    # plotting
    # get generator example
    g_ex = generator.forward(z_ex)
    g_plot = g_ex.cpu().detach()

    # plot result
    save_image(g_plot.data[:25],
               os.path.join(save_fig_path, 'g_%d.pdf' % epoch),
               nrow=5, normalize=True)

    fig1, ax = plt.subplots(1, 3, figsize=(15, 5))
    #ax0 = ax[0].imshow(T.cpu().detach().numpy(), cmap='RdBu_r')
    #colorbar(ax0)
    ax1 = ax[1].imshow(D_x.cpu().detach().numpy(), cmap='Blues')
    colorbar(ax1)
    ax2 = ax[2].imshow(D_g.cpu().detach().numpy(), cmap='Blues')
    colorbar(ax2)
    #ax[0].set_title(r'$T$')
    ax[1].set_title(r'inner distances of $D$')
    ax[2].set_title(r'inner distances of $G$')
    plt.tight_layout(h_pad=1)
    fig1.savefig(os.path.join(save_fig_path, '{}_ccc.pdf'.format(
            str(epoch).zfill(3))), bbox_inches='tight')

    loss_history.append(loss)
    loss_tv.append(loss_t)
    loss_orth.append(loss_og)
    plt.close('all')

# After the training loop, compute the mean and standard deviation of time per epoch
mean_time_per_epoch = np.mean(epoch_times)
std_dev_time_per_epoch = np.std(epoch_times)

# Print the mean and standard deviation of the time per epoch
print(f"\nMean time per epoch: {mean_time_per_epoch:.4f} seconds.")
print(f"Standard deviation in time per epoch: {std_dev_time_per_epoch:.4f} seconds.")
# After finishing all epochs, calculate the mean and variance of the reconstruction losses in the last epoch
if reconstruction_losses_last_epoch:
    mean_fid = np.mean(fid_scores_last_epoch)
    variance_fid = np.var(fid_scores_last_epoch)

    print(f"\nMean FID Score (Epoch {num_epochs}): {mean_fid}")
    print(f"Variance of FID Score (Epoch {num_epochs}): {variance_fid}")
        
    mean_loss = np.mean(reconstruction_losses_last_epoch)
    variance_loss = np.var(reconstruction_losses_last_epoch)

    print(f"Mean Reconstruction Loss (Epoch {num_epochs}): {mean_loss}")
    print(f"Variance of Reconstruction Loss (Epoch {num_epochs}): {variance_loss}")
        
# plot loss history
fig2 = plt.figure(figsize=(2.4, 2))
ax2 = fig2.add_subplot(111)
ax2.plot(loss_history, 'k.')
ax2.set_xlabel('Iterations')
ax2.set_ylabel(r'$\overline{GW}_\epsilon$ Loss')
plt.tight_layout()
plt.grid()
fig2.savefig(save_fig_path + '/loss_history.pdf')

fig3 = plt.figure(figsize=(2.4, 2))
ax3 = fig3.add_subplot(111)
ax3.plot(loss_tv, 'k.')
ax3.set_xlabel('Iterations')
ax3.set_ylabel(r'Total Variation Loss')
plt.tight_layout()
plt.grid()
fig3.savefig(save_fig_path + '/loss_tv.pdf')

fig4 = plt.figure(figsize=(2.4, 2))
ax4 = fig4.add_subplot(111)
ax4.plot(loss_orth, 'k.')
ax4.set_xlabel('Iterations')
ax4.set_ylabel(r'$R_\beta(f_\omega(X), X)$ Loss')
plt.tight_layout()
plt.grid()
fig4.savefig(save_fig_path + '/loss_orth.pdf')

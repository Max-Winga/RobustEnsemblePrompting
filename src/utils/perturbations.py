import torch
import lpips

def gaussian_noise_perturbation(std, mean=0):
    def perturbation(data):
        noise = torch.randn_like(data) * std + mean
        return data + noise.to(data.device)
    return perturbation

def laplacian_noise_perturbation(scale, loc=0):
    def perturbation(data):
        noise = torch.distributions.Laplace(loc, scale).sample(data.shape)
        return data + noise.to(data.device)
    return perturbation

def lp_norm_perturbation(epsilon, p=2):
    def perturbation(data):
        perturbation = torch.randn_like(data)
        perturbation = epsilon * perturbation / perturbation.norm(p=p)
        return data + perturbation.to(data.device)
    return perturbation

def lpips_perturbation(net='vgg', std=1.0):
    """
    Applies a perturbation sampled from the LPIPS distribution.
    
    Args:
        net (str): The network to use for LPIPS ('alex', 'vgg', or 'squeeze').
        std (float): The standard deviation of the noise to be added.
    
    Returns:
        A function that takes a PyTorch tensor `data` and returns the perturbed tensor.
    """
    def perturbation(data):
        device = data.device  # Get the device of the input data
        loss_fn = lpips.LPIPS(net=net).to(device)  # Move the LPIPS model to the same device

        noise = std * torch.randn_like(data)
        perturbed_data = data + noise

        # Compute the LPIPS distance between the original data and the perturbed data
        lpips_dist = loss_fn(data, perturbed_data)

        # Sample from the LPIPS distribution
        sampled_lpips = torch.distributions.Normal(lpips_dist, 1).sample()

        # Scale the perturbed data based on the sampled LPIPS distance
        perturbed_data = data + (sampled_lpips.view(-1, 1, 1, 1) * noise)
  
        return perturbed_data.squeeze(1)
    
    return perturbation

def create_perturbations(descriptions):
    perturbations = []
    for description in descriptions:
        match description[0]:
            case "Gaussian":
                perturbations.append(gaussian_noise_perturbation(description[1]))
            case "Laplacian":
                perturbations.append(laplacian_noise_perturbation(description[1]))
            case "Lp-norm":
                perturbations.append(lp_norm_perturbation(description[1], description[2]))
            case "lpips":
                perturbations.append(lpips_perturbation(description[1]))
    return perturbations
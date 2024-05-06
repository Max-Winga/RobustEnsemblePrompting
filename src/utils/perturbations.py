import torch
import lpips

def gaussian_noise_perturbation(std, mean=0):
    def perturbation(data):
        noise = torch.randn_like(data).to(data.device) * std + mean
        perturbed_data = data + noise
        perturbed_data = torch.clamp(perturbed_data, min=0, max=1)
        return perturbed_data
    return perturbation

def laplacian_noise_perturbation(scale, loc=0):
    def perturbation(data):
        noise = torch.distributions.Laplace(loc, scale).sample(data.shape).to(data.device)
        perturbed_data = data + noise
        perturbed_data = torch.clamp(perturbed_data, min=0, max=1)
        return perturbed_data
    return perturbation

def lp_norm_perturbation(epsilon, p=2):
    def perturbation(data):
        perturbation = torch.randn_like(data).to(data.device)
        perturbation = epsilon * perturbation / perturbation.norm(p=p)
        perturbed_data = data + perturbation
        perturbed_data = torch.clamp(perturbed_data, min=0, max=1)
        return perturbed_data
    return perturbation

def contrast_brightness_perturbation(contrast_factor=0.5, brightness_factor=0.5):
    def perturbation(data):
        contrast = torch.rand(1).to(data.device) * contrast_factor + (1 - contrast_factor)
        brightness = torch.rand(1).to(data.device) * brightness_factor + (1 - brightness_factor)
        perturbed_data = data * contrast + brightness
        perturbed_data = torch.clamp(perturbed_data, min=0, max=1)
        return perturbed_data
    return perturbation

def rotation_flip_perturbation(prob):
    def perturbation(data):
        if torch.rand(1).to(data.device) < prob:
            k = torch.randint(1, 4, (1,)).to(data.device).item()
            perturbed_data = torch.rot90(data, k, dims=[-2, -1])
        else:
            perturbed_data = data
        if torch.rand(1).to(data.device) < prob:
            perturbed_data = torch.flip(perturbed_data, dims=[-1])
        return perturbed_data
    return perturbation

def salt_pepper_noise_perturbation(prob):
    def perturbation(data):
        mask = torch.rand_like(data).to(data.device) < prob
        perturbed_data = data.clone()
        perturbed_data[mask] = torch.randint_like(data[mask], low=0, high=2)
        return perturbed_data
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

def stacked_perturbation(perturbations):
    def perturbation(data):
        perturbed_data = data
        for p in perturbations:
            perturbed_data = p(perturbed_data)
        return perturbed_data
    return perturbation

def create_perturbations(descriptions):
    perturbations = []
    for description in descriptions:
        if isinstance(description[0], list):
            # If the first element is a list, it's a stacked perturbation
            stacked_perturbations = []
            for p_description in description:
                p = create_single_perturbation(p_description)
                if p is not None:
                    stacked_perturbations.append(p)
            perturbations.append(stacked_perturbation(stacked_perturbations))
        else:
            # If the first element is not a list, it's a single perturbation
            p = create_single_perturbation(description)
            if p is not None:
                perturbations.append(p)
    return perturbations

def create_single_perturbation(description):
    match description[0]:
        case "Gaussian":
            return gaussian_noise_perturbation(description[1])
        case "Laplacian":
            return laplacian_noise_perturbation(description[1])
        case "Lp-norm":
            return lp_norm_perturbation(description[1], description[2])
        case "ContrastBrightness":
            return contrast_brightness_perturbation(description[1], description[2])
        case "RotationFlip":
            return rotation_flip_perturbation(description[1])
        case "SaltPepper":
            return salt_pepper_noise_perturbation(description[1])
        case "lpips":
            return lpips_perturbation(description[1])
        case "Identity":
            def identity(x):
                return x
            return identity
        case _:
            print(f"Recieved Improper Perturbation Input: {description}")
            return None
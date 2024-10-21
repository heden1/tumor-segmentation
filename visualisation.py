import matplotlib.pyplot as plt
import numpy as np
import torch
def plot_training(performance_metrics,label=None):
    
    """Plot the training and validation loss and accuracy
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for model_i,(_,train_losses, train_accs, val_losses, val_accs) in enumerate(performance_metrics):
        x_step_val = np.linspace(1, len(val_accs), len(val_accs))
        x_step_train = np.linspace(len(val_accs)/(len(train_accs)),len(val_accs), len(train_accs))

        if label is not None:
            ax1.plot(x_step_train, train_losses, label=f'Train Loss {label[model_i]}',alpha=0.5)
            ax1.plot(x_step_val, val_losses, label=f'Validation Loss {label[model_i]}')
        else:
            ax1.plot(x_step_train, train_losses, label=f'Train Loss {model_i}',alpha=0.5)
            ax1.plot(x_step_val, val_losses, label=f'Validation Loss {model_i}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.legend()
        
        if label is not None:
            ax2.plot(x_step_train, train_accs, label=f'Train Accuracy {label[model_i]}',alpha=0.5)
            ax2.plot(x_step_val, val_accs, label=f'Validation Accuracy {label[model_i]}')
        else:
            ax2.plot(x_step_train, train_accs, label=f'Train Accuracy {model_i}',alpha=0.5)
            ax2.plot(x_step_val, val_accs, label=f'Validation Accuracy {model_i}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy Over Time')
        ax2.legend()

    plt.show()


def save_model_to_file(metrics,  filename):
    model, train_losses, train_accs, val_losses, val_accs = metrics
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_losses": train_losses,
            "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs,
        },
        filename,
    )

# Example of creating and initialising model with a previously saved state dict:
def get_model_and_performance_metrics(filename,model_class):

    checkpoint = torch.load(filename)
    model_class.load_state_dict(checkpoint["model_state_dict"])

    return model_class, checkpoint["train_losses"], checkpoint["train_accs"], checkpoint["val_losses"], checkpoint["val_accs"]


def get_model_and_performance_metrics(filename, model_class):
    checkpoint = torch.load(filename)
    model_class.load_state_dict(checkpoint["model_state_dict"])
    return model_class, checkpoint["train_losses"], checkpoint["train_accs"], checkpoint["val_losses"], checkpoint["val_accs"]

def plot_segmentation(image, mask, model, threshold=0.5):
    """Plot the segmentation mask overlayed on the image"""
    # Set model to evaluation
    model.eval()
    
    # Move image to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
        pred_mask = torch.sigmoid(output) > threshold  # Apply sigmoid and threshold
    
    # Convert tensors to numpy arrays for visualization

    image = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    mask = mask.squeeze(0).cpu().numpy()
    pred_mask = pred_mask.squeeze(0).squeeze(0).cpu().numpy()

    # Rescale the image to its original range (0, 255)
    transform_mean = [0.485, 0.456, 0.406]
    transform_std = [0.229, 0.224, 0.225]
    image = (image * transform_std + transform_mean) * 255
    image = image.astype(np.uint8)

    # Create an alpha channel for the mask
    alpha_mask = np.zeros_like(mask, dtype=np.float32)
    alpha_mask[mask > 0] = 0.5  # Adjust transparency here (0.5 for 50% transparency)
    
    alpha_pred_mask = np.zeros_like(pred_mask, dtype=np.float32)
    print(alpha_pred_mask.max())
    alpha_pred_mask[pred_mask > 0] = 0.5  # Adjust transparency here (0.5 for 50% transparency)
    
    # Plot the original image,
    
    # Plot the original image, ground truth mask, and predicted mask
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image with ground truth mask
    axes[0].imshow(image)
    axes[0].imshow(mask, cmap='jet', alpha=alpha_mask)  # Overlay mask with transparency
    axes[0].set_title("Ground Truth Mask")
    axes[0].axis('off')
    
    # Original image with predicted mask
    axes[1].imshow(image)
    axes[1].imshow(pred_mask, cmap='jet', alpha=alpha_pred_mask)  # Overlay mask with transparency
    axes[1].set_title("Predicted Mask")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_comparison(idx, dataset, model, threshold=0.5):
    """Plot comparison between the ground truth mask and the model's prediction"""
    image, mask = dataset[idx]
    
    # Plot the segmentation
    plot_segmentation(image, mask, model, threshold)

# Example usage
if __name__ == "__main__":
    from preprocessing import create_dataloaders
    from unet import UNet  # Assuming you have a UNet class defined in unet.py
    
    # Load the model and performance metrics
    model, train_losses, train_accs, val_losses, val_accs = get_model_and_performance_metrics("model_checkpoint.pth", UNet())
    
    # Create the dataloaders
    resize_size = (224, 224)
    train_loader, val_loader = create_dataloaders(resize_size)
    
    # Get the validation dataset
    val_dataset = val_loader.dataset
    
    # Plot comparison for a specific index
    plot_comparison(0, val_dataset, model)
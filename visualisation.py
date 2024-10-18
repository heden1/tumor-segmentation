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


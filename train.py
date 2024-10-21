import torch
from tqdm import tqdm 
import numpy as np
from sklearn.metrics import f1_score


def training_loop(
    model, optimizer, loss_fn, train_loader, val_loader, num_epochs=10):
    print("Starting training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    for epoch in range(1,num_epochs+1):
        model, train_loss, train_acc, train_f1 = train_epoch(model,train_loader, loss_fn,optimizer, device)
        val_loss, val_acc,val_f1 = validate(model, loss_fn, val_loader, device)
        print(
            f"Epoch {epoch}/{num_epochs}: "
            f"Train loss: {sum(train_loss)/len(train_loss):.3f}, "
            f"Train f1.: {sum(train_f1)/len(train_f1):.3f}, "
            f"Train accuracy: {sum(train_acc)/len(train_acc):.3f}, "
            f"Val. loss: {val_loss:.3f}, "
            f"Val. f1.: {val_f1:.3f}"
            f" Val. accuracy: {val_acc:.3f}"
        )
        train_losses.extend(train_loss)
        train_accs.extend(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    return model, train_losses, train_accs, val_losses, val_accs

def train_epoch(model,train_loader, loss_fn,optimizer, device):
        model.train()
        epoch_loss = 0
        train_loss_batches, train_acc_batches , train_f1_batches= [], [], []
        for images, masks in tqdm(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(images)#['out']
        
            # Compute the loss
            loss = loss_fn(outputs, masks)
            epoch_loss += loss.item()
            # Backward pass
            train_loss_batches.append(loss.item())
            loss.backward()
            optimizer.step()
            train_f1_batches.append(calculate_f1(outputs, masks))  
            train_acc_batches.append(calculate_accuracy(outputs, masks))  
        return  model, train_loss_batches, train_acc_batches,train_f1_batches

def calculate_accuracy(outputs, masks):
    hard_preds = (outputs > 0.5)
    return (hard_preds == masks).float().mean().item()

def calculate_f1(preds, targets, threshold=0.5):
    """
    Calculates the F1 score per batch for an image segmentation task using sklearn.

    Args:
        preds (torch.Tensor): The predicted output from the model. Shape: (batch_size, height, width).
        targets (torch.Tensor): The ground truth labels. Shape: (batch_size, height, width).
        threshold (float): Threshold to convert logits/probabilities to binary predictions.

    Returns:
        f1_score (float): The F1 score for the batch.
    """
    
    # Convert logits or probabilities to binary predictions (foreground vs background)
    preds = (preds > threshold).float()
    
    
    # Flatten the tensorss
    preds = preds.view(-1).cpu().numpy().astype(int)

    targets = targets.view(-1).cpu().numpy().astype(int)
    
    # Calculate F1 score using sklearn

    return f1_score(targets, preds, average='binary')

def calculate_f11(preds, targets, threshold=0.5, epsilon=1e-7):
    """
    Calculates the F1 score per batch for an image segmentation task.

    Args:
        preds (torch.Tensor): The predicted output from the model. Shape: (batch_size, height, width).
        targets (torch.Tensor): The ground truth labels. Shape: (batch_size, height, width).
        threshold (float): Threshold to convert logits/probabilities to binary predictions.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        f1_score (float): The F1 score for the batch.
    """
    
    # Convert logits or probabilities to binary predictions (foreground vs background)
    preds = (preds > threshold).float()
    
    # Calculate True Positives, False Positives, False Negatives
    TP = (preds * targets).sum(dim=(1, 2))  # True Positives per batch
    FP = (preds * (1 - targets)).sum(dim=(1, 2))  # False Positives per batch
    FN = ((1 - preds) * targets).sum(dim=(1, 2))  # False Negatives per batch

    # Precision, Recall
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    
    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    #espilon is to not divide with 0 
    # Return average F1 score over the batch
    return f1_score.mean().item()

def calculate_f12(outputs, masks, threshold=0.5):
    # Apply threshold to convert outputs to binary
    hard_preds = (torch.sigmoid(outputs) > threshold).cpu().numpy().flatten()
    masks = masks.cpu().numpy().flatten()
    print(type(hard_preds))
    print(type(masks))
    return f1_score(masks, hard_preds, average='binary')


def validate(model, loss_fn, val_loader, device):
    val_loss_cum = 0
    val_acc_cum = 0
    val_f1_cum = 0
    model.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):
            inputs, labels = x.to(device), y.to(device)
            z = model.forward(inputs)
            batch_loss = loss_fn(z, labels)
            val_loss_cum += batch_loss.item()
            
            acc_batch_avg = calculate_accuracy(z, labels)
            f1_batch_avg=calculate_f1(z, labels)
            val_f1_cum += f1_batch_avg
            val_acc_cum+=acc_batch_avg

    return val_loss_cum / len(val_loader) ,val_acc_cum / len(val_loader),val_f1_cum / len(val_loader)


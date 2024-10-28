import torch
from tqdm import tqdm 
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from visualisation import save_model_to_file


def training_loop(
    model, optimizer, loss_fn, train_loader, val_loader, num_epochs=10,warmup_epochs=0):
    print("Starting training")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Training on device {device}")
    model.to(device)
    train_losses, train_accs, val_losses, val_accs ,train_f1s, val_f1s= [], [], [], [], [],[]

    def lambda_lr(epoch):
        if epoch < warmup_epochs:
            return ((epoch + 1) / warmup_epochs)**2
        else:
            return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

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
        train_f1s.extend(train_f1)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        if early_stopping(val_losses):
            print("Early stopping")
            break
        save_model_to_file((model, train_losses, train_f1s, val_losses, val_f1s), "model_during_traning.pth")
        scheduler.step()

    return model, train_losses, train_f1s, val_losses, val_f1s

def train_epoch(model,train_loader, loss_fn,optimizer, device):
        model.train()
        epoch_loss = 0
        train_loss_batches, train_acc_batches , train_f1_batches= [], [], []
        for images, masks in tqdm(train_loader):
            images = images.to(device)
            masks = masks.to(device)
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


def early_stopping(val_losses, patience=3):
    if len(val_losses) > patience:
        # Check if the validation loss has not improved for the specified number of epochs
        recent_losses = val_losses[-patience:]
        if all(x >= recent_losses[0] for x in recent_losses[1:]):
            return True
    return False

def calculate_accuracy(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    preds = preds.view(-1).cpu().numpy().astype(int)
    targets = targets.view(-1).cpu().numpy().astype(int)

    return accuracy_score(targets, preds)


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

def predict_and_calc_f1(model, data_loader, device):
    model.eval()
    f1s = []
    accs=[]
    predictions=[]
    diff_areas=[]
    with torch.no_grad():
        for images, masks in tqdm(data_loader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            if outputs.count_nonzero() == 0:
                predictions.append(0)
                print("No predictions")
            else: 
                predictions.append(1)
            diff_areas.append(outputs.count_nonzero()/masks.count_nonzero())
            acc=calculate_accuracy(outputs, masks)
            accs.append(acc)
            f1 = calculate_f1(outputs, masks)
            f1s.append(f1)
    return f1s,accs,predictions,diff_areas



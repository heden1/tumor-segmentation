import torch
from tqdm import tqdm 
import numpy as np

def training_loop(
    model, optimizer, loss_fn, train_loader, val_loader, num_epochs=10):
    print("Starting training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    for epoch in range(1,num_epochs+1):
        model, train_loss, train_acc = train_epoch(model,train_loader, loss_fn,optimizer, device)
        val_loss, val_acc = validate(model, loss_fn, val_loader, device)
        print(
            f"Epoch {epoch}/{num_epochs}: "
            f"Train loss: {sum(train_loss)/len(train_loss):.3f}, "
            f"Train acc.: {sum(train_acc)/len(train_acc):.3f}, "
            f"Val. loss: {val_loss:.3f}, "
            f"Val. acc.: {val_acc:.3f}"
        )
        train_losses.extend(train_loss)
        train_accs.extend(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    return model, train_losses, train_accs, val_losses, val_accs

def train_epoch(model,train_loader, loss_fn,optimizer, device):
        model.train()
        epoch_loss = 0
        train_loss_batches, train_acc_batches = [], []
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
            train_acc_batches.append(calculate_accuracy(outputs, masks))    
        return  model, train_loss_batches, train_acc_batches

def calculate_accuracy(outputs, masks):
    hard_preds = (outputs > 0.5)
    return (hard_preds == masks).float().mean().item()


def validate(model, loss_fn, val_loader, device):
    val_loss_cum = 0
    val_acc_cum = 0
    model.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):
            inputs, labels = x.to(device), y.to(device)
            z = model.forward(inputs)
            batch_loss = loss_fn(z, labels)
            val_loss_cum += batch_loss.item()
            acc_batch_avg=calculate_accuracy(z, labels)
            val_acc_cum += acc_batch_avg
    return val_loss_cum / len(val_loader), val_acc_cum / len(val_loader)


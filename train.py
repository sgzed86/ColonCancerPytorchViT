import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from model import MedicalVisionTransformer
from dataset import get_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description='Train Medical Vision Transformer')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to HyperKvasir dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    return parser.parse_args()

def train_epoch(model, train_loader, criterion, optimizer, device, scaler, gradient_accumulation_steps=1):
    model.train()
    total_loss = 0
    lesion_preds, lesion_labels = [], []
    polyp_preds, polyp_labels = [], []
    fibrosis_preds, fibrosis_labels = [], []
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)
        lesion_labels_batch = batch['lesion_label'].to(device, non_blocking=True)
        polyp_labels_batch = batch['polyp_label'].to(device, non_blocking=True)
        fibrosis_labels_batch = batch['fibrosis_label'].to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        with autocast(device_type='cuda', dtype=torch.float16, enabled=scaler is not None):
            lesion_out, polyp_out, fibrosis_out = model(images)
            
            # Calculate losses
            lesion_loss = criterion['bce'](lesion_out.view(-1), lesion_labels_batch.view(-1))
            polyp_loss = criterion['ce'](polyp_out, polyp_labels_batch)
            fibrosis_loss = criterion['ce'](fibrosis_out, fibrosis_labels_batch)
            
            # Combined loss
            loss = (lesion_loss + polyp_loss + fibrosis_loss) / gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        if scaler is not None:
            scaler.scale(loss).backward()
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        
        # Store predictions and labels
        lesion_preds.extend((torch.sigmoid(lesion_out.view(-1)) > 0.5).cpu().numpy())
        lesion_labels.extend(lesion_labels_batch.view(-1).cpu().numpy())
        
        polyp_preds.extend(polyp_out.argmax(dim=1).cpu().numpy())
        polyp_labels.extend(polyp_labels_batch.cpu().numpy())
        
        fibrosis_preds.extend(fibrosis_out.argmax(dim=1).cpu().numpy())
        fibrosis_labels.extend(fibrosis_labels_batch.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
    
    # Calculate metrics
    lesion_acc = accuracy_score(lesion_labels, lesion_preds)
    polyp_acc = accuracy_score(polyp_labels, polyp_preds)
    fibrosis_acc = accuracy_score(fibrosis_labels, fibrosis_preds)
    
    return {
        'loss': total_loss / len(train_loader),
        'lesion_acc': lesion_acc,
        'polyp_acc': polyp_acc,
        'fibrosis_acc': fibrosis_acc
    }

def validate(model, val_loader, criterion, device, scaler):
    model.eval()
    total_loss = 0
    lesion_preds, lesion_labels = [], []
    polyp_preds, polyp_labels = [], []
    fibrosis_preds, fibrosis_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            images = batch['image'].to(device, non_blocking=True)
            lesion_labels_batch = batch['lesion_label'].to(device, non_blocking=True)
            polyp_labels_batch = batch['polyp_label'].to(device, non_blocking=True)
            fibrosis_labels_batch = batch['fibrosis_label'].to(device, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=torch.float16, enabled=scaler is not None):
                lesion_out, polyp_out, fibrosis_out = model(images)
                
                # Calculate losses
                lesion_loss = criterion['bce'](lesion_out.view(-1), lesion_labels_batch.view(-1))
                polyp_loss = criterion['ce'](polyp_out, polyp_labels_batch)
                fibrosis_loss = criterion['ce'](fibrosis_out, fibrosis_labels_batch)
                
                loss = lesion_loss + polyp_loss + fibrosis_loss
            
            total_loss += loss.item()
            
            # Store predictions and labels
            lesion_preds.extend((torch.sigmoid(lesion_out.view(-1)) > 0.5).cpu().numpy())
            lesion_labels.extend(lesion_labels_batch.view(-1).cpu().numpy())
            
            polyp_preds.extend(polyp_out.argmax(dim=1).cpu().numpy())
            polyp_labels.extend(polyp_labels_batch.cpu().numpy())
            
            fibrosis_preds.extend(fibrosis_out.argmax(dim=1).cpu().numpy())
            fibrosis_labels.extend(fibrosis_labels_batch.cpu().numpy())
    
    # Calculate metrics
    lesion_acc = accuracy_score(lesion_labels, lesion_preds)
    polyp_acc = accuracy_score(polyp_labels, polyp_preds)
    fibrosis_acc = accuracy_score(fibrosis_labels, fibrosis_preds)
    
    return {
        'loss': total_loss / len(val_loader),
        'lesion_acc': lesion_acc,
        'polyp_acc': polyp_acc,
        'fibrosis_acc': fibrosis_acc
    }

def main():
    print("Starting training script...")
    args = parse_args()
    print(f"Arguments: {args}")
    
    # Create output directory
    os.makedirs('checkpoints', exist_ok=True)
    print("Created checkpoints directory")
    
    print("Loading datasets...")
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )
    print(f"Train loader size: {len(train_loader.dataset)}")
    print(f"Val loader size: {len(val_loader.dataset)}")
    
    print("Initializing model...")
    # Initialize model
    model = MedicalVisionTransformer().to(args.device)
    print(f"Model moved to {args.device}")
    
    # Define loss functions
    criterion = {
        'bce': nn.BCEWithLogitsLoss(),
        'ce': nn.CrossEntropyLoss()
    }
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if args.amp else None
    
    print("Starting training loop...")
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # Train
        train_metrics = train_epoch(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            args.device,
            scaler,
            args.gradient_accumulation_steps
        )
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Lesion Acc: {train_metrics['lesion_acc']:.4f}, "
              f"Polyp Acc: {train_metrics['polyp_acc']:.4f}, "
              f"Fibrosis Acc: {train_metrics['fibrosis_acc']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, args.device, scaler)
        print(f"Val - Loss: {val_metrics['loss']:.4f}, "
              f"Lesion Acc: {val_metrics['lesion_acc']:.4f}, "
              f"Polyp Acc: {val_metrics['polyp_acc']:.4f}, "
              f"Fibrosis Acc: {val_metrics['fibrosis_acc']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
            }, 'checkpoints/best_model.pth')
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
        
        scheduler.step()

if __name__ == '__main__':
    main() 
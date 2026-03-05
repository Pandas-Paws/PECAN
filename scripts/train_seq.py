import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import wandb
from scripts.config import cfg
from scripts.NSE_loss import NSELoss
from scripts.weighted_MSE_loss import WeightedRMSELoss
import pdb
import matplotlib.pyplot as plt
import pickle
import numpy as np
from pprint import pprint

test_csv_name = "test_preds_mcrouting_retention_cs_0.80.2_180_0.2do_epoch80_randval_cosannealing_mcr_do0.1_notc"
saving_pkl_name = "pecan_best_model_mcrouting_retention_cs_0.80.2_180_0.2do_epoch80_randval_cosannealing_mcr_do0.1_notc"

test_csv_name = "test_preds_mcrouting_retention_cs_0.80.2_60_0.2do_epoch80_randval_cosannealing_mcr_lr0.001_retrain"
saving_pkl_name = "pecan_best_model_mcrouting_retention_cs_0.80.2_60_0.2do_epoch80_randval_cosannealing_mcr_lr0.001_retrain"

#test_csv_name = "test_preds_mcrouting_retention_cs_0.90.1_180_0.2do_epoch80_randval_cosannealing_vbrouting"
#saving_pkl_name = "pecan_best_model_mcrouting_retention_cs_0.90.1_180_0.2do_epoch80_randval_cosannealing_vbrouting"

def train_model(model, train_loader, val_loader):
    """
    Train the Pecan model and save the best-performing model as a .pkl file.
    """

    #criterion = WeightedRMSELoss()
    criterion = NSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.NUM_EPOCHS,   # full cosine cycle = total epochs
        eta_min=1e-4            # final LR floor
    )
    

    # Early stopping parameters
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_state = None

    # Ensure checkpoint directory exists
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    pprint({k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})


    print("Starting Training...")
    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (batch, mask, target, timestamps) in enumerate(train_loader):
            batch, mask, target = batch.to(cfg.DEVICE), mask.to(cfg.DEVICE), target.to(cfg.DEVICE)
            # batch: [B, Seq, in_size, Y, X]
            # mask: [B, Seq, in_size, Y, X]
            # target: [B, station_num]
            xm = batch[:, :, 0:1, :, :]
            xa = batch[:, :, 1:, :, :]

            optimizer.zero_grad()
            q_pred, *_ = model(xm, xa, mask) # todo, fastslow
            
            loss = criterion(q_pred, target)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

        # Compute validation loss
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch, mask, target, timestamps in val_loader:
                batch, mask, target = batch.to(cfg.DEVICE), mask.to(cfg.DEVICE), target.to(cfg.DEVICE)
                xm = batch[:, :, 0:1, :, :]
                xa = batch[:, :, 1:, :, :]

                q_pred, *_ = model(xm, xa, mask)
                val_loss += criterion(q_pred, target).item()

        # Average losses
        train_loss = running_loss / len(train_loader)
        val_loss /= len(val_loader)

        scheduler.step() # val_loss if scheduler is ReduceLROnPlateau
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"Gradient of {name}: mean={param.grad.mean().item()}, std={param.grad.std().item()}")


        print(f"Epoch [{epoch+1}/{cfg.NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # **Save the best model based on validation loss**
        if val_loss < best_val_loss:
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            best_val_loss = val_loss
            epochs_no_improve = 0
            model_path = os.path.join(checkpoint_dir, f"{saving_pkl_name}.pkl")
            
            # Save model as a .pkl file
            with open(model_path, "wb") as f:
                pickle.dump({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss
                }, f)

            print(f"Model improved, checkpoint saved at {model_path}")

        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")            
            break
            
    if best_state is not None:
        model.load_state_dict(best_state)
    print("Training complete (best val loss {:.4f}).".format(best_val_loss))
    return model



def test_model(model, test_loader):
    """
    Evaluate the Pecan model on test data and save predictions in CSV format.

    Parameters
    ----------
    model : torch.nn.Module
        The trained Pecan model.
    test_loader : DataLoader
        DataLoader for test data (not shuffled).

    Returns
    -------
    float
        The final test loss.
    """
    print("Starting Testing...")
    
    #criterion = WeightedRMSELoss()
    criterion = NSELoss()
    
    test_loss = 0.0

    # Ensure directory for predictions exists
    pred_dir = "predictions"
    os.makedirs(pred_dir, exist_ok=True)

    all_test_data = []

    # Get actual number of test days from dataset
    num_test_days = len(test_loader.dataset)

    model.eval()
    with torch.no_grad():
        for batch_idx, (batch, mask, target, timestamps) in enumerate(test_loader):
            timestamps = list(timestamps)  # list of date strings, e.g., "2015-11-03"

            batch = batch.to(cfg.DEVICE)
            mask = mask.to(cfg.DEVICE)
            target = target.to(cfg.DEVICE)

            xm = batch[:, :, 0:1, :, :]
            xa = batch[:, :, 1:, :, :]
            
            # PECAN initial state: shape (1, out_channels, H, W)
            pecan_init = model.initial_state.detach().cpu().numpy()[0]  # shape: (out_channels, H, W)
        
            # Sum over channels to get total water per grid
            pecan_init_sum = pecan_init.sum(axis=0)  # shape: (H, W)
        
            # Mean over all valid grid cells
            pecan_areal_sum = np.mean(pecan_init_sum)  # mm per grid cell
            print(f"PECAN Areal Mean Initial State: {pecan_areal_sum:.4f} mm")
            
            #q_pred, trash_cell, hidden_state, cell_state = model(xm, xa, mask) # todo, no fast flow version
            q_pred, trash_cell, hidden_state, cell_state, retention, mr_gate, mr_flux = model(xm, xa, mask) # todo, fast flow version
            
            # Compute loss on last timestep
            test_loss += criterion(q_pred, target).item()

            # Save trash cell and cell state matrices using date-based names
            model.save_trash_cell(trash_cell, timestamps, saving_path="trash_cell_outputs")
            model.save_cell_state(cell_state, timestamps, saving_path="cell_state_outputs")
            model.save_cell_state(retention, timestamps, saving_path="retention_state_outputs")
            model.save_mr_flux(mr_flux, timestamps, "mr_flux_outputs", sum_channels=True)
            model.save_mr_gate(mr_gate,  timestamps, "mr_gate_outputs", sum_channels=True)

            # Save predictions and observations with timestamps
            for b in range(target.shape[0]):
                row = [timestamps[b]] + list(target[b, :].cpu().numpy()) + list(q_pred[b, :].cpu().numpy())
                all_test_data.append(row)

    # Save test predictions to CSV
    gauge_cols = [f"Observed_{i+1}" for i in range(target.shape[-1])] + [f"Predicted_{i+1}" for i in range(target.shape[-1])]
    test_df = pd.DataFrame(all_test_data, columns=["Time"] + gauge_cols)
    test_df.to_csv(f"{pred_dir}/{test_csv_name}.csv", index=False)
    # Log final test loss
    final_test_loss = test_loss / len(test_loader)
    if wandb.run is not None:
        wandb.log({"test_loss": final_test_loss})

    return final_test_loss


def load_trained_model(model, checkpoint_path):
    """
    Load the trained model weights from a `.pkl` file.
    """
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        print(f"Loaded trained model from {checkpoint_path}, trained until epoch {checkpoint['epoch']}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")





















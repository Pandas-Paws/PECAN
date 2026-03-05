import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pickle
import numpy as np
import wandb
from scripts.config import cfg
from scripts.NSE_loss import NSELoss
import pdb
import numpy as np
from pathlib import Path

def load_scaler(pkl_path: str):
    """Load {mean, std} for QObs from a pickle file."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)          # expects a dict with …_mean / …_std keys

def inverse_scale_Q(z, stats):
    """Convert z-scores back to streamflow units."""
    print(stats["q_mean"], stats["q_std"])
    return z * stats["q_std"] + stats["q_mean"]

def train_lumped_mcmodel(model, train_loader, val_loader):
    criterion = NSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.NUM_EPOCHS,   # full cosine cycle = total epochs
        eta_min=1e-4            # final LR floor
    )

    best_val_loss = float('inf')
    patience = 10
    no_improve_epochs = 0
    best_state = None
    os.makedirs("checkpoints", exist_ok=True)

    print("Starting MC-LSTM Training...")
    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        total_loss = 0

        for x, _, y, _ in train_loader:
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            x = x.mean(dim=(-2, -1))  # (B, T, C)
            xm, xa = x[..., 0:1], x[..., 1:]

            optimizer.zero_grad()
            m_out, _, _ = model(xm, xa)  # (B, T, H)
            pred = m_out[:, :, 1:].sum(dim=-1, keepdim=True)
            loss = criterion(pred[:, -1], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _, y, _ in val_loader:
                x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
                x = x.mean(dim=(-2, -1))
                xm, xa = x[..., 0:1], x[..., 1:]  # [B, T, 1], [B, T, 2]
                m_out, cell_state, _ = model(xm, xa)
                pred = m_out[:, :, 1:].sum(dim=-1, keepdim=True)
                val_loss += criterion(pred[:, -1], y).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step() # avg_val_loss

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
            model_path = os.path.join("checkpoints", "mclstm_best_model_hidden32_sq180_ep80_seed3407_cosannealing_retrain.pkl") # todo
            with open(model_path, "wb") as f:
                pickle.dump({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss
                }, f)
            print(f"Saved best model to {model_path}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping.")
                break
    if best_state is not None:
      model.load_state_dict(best_state)
    print("Training complete (best val loss {:.4f}).".format(best_val_loss))


def test_lumped_mcmodel(model, test_loader):
    criterion = NSELoss()
    test_loss = 0
    pred_dir = "predictions"
    trash_dir = "trash_cell_outputs_mclstm"
    cell_dir = "cell_state_outputs_mclstm"

    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(trash_dir, exist_ok=True)
    os.makedirs(cell_dir, exist_ok=True)

    all_results = []
    saved_dates = set()  # to track which dates we've saved to avoid duplicates

    model.eval()
    with torch.no_grad():
        for x, _, y, timestamps in test_loader:
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            x = x.mean(dim=(-2, -1))
            xm, xa = x[..., 0:1], x[..., 1:]

            m_out, cell_state, _ = model(xm, xa)
            
            # MC-LSTM initial state: shape (1, out_dim)
            #mclstm_init = model.init_state_param.detach().cpu().numpy()[0]  # shape: (out_dim,)
        
            # Sum over all hidden units
            #mclstm_sum = np.sum(mclstm_init)
            #print(f"MC-LSTM Initial State: {mclstm_sum:.4f} mm")
            
            
            pred = m_out[:, :, 1:].sum(dim=-1, keepdim=True)
            trash_cell = m_out[:, :, 0]  # (B, T)
            final_pred = pred[:, -1, :]  # (B, 1)

            test_loss += criterion(final_pred, y).item()

            for i in range(y.size(0)):
                ts = timestamps[i]
                date_str = pd.to_datetime(ts).strftime('%Y%m%d')

                obs_val = y[i].item()
                pred_val = final_pred[i].item()
                all_results.append([ts, obs_val, pred_val])

                # Save hidden states only once per unique date
                if date_str not in saved_dates:
                    np.save(os.path.join(trash_dir, f"trash_cell_{date_str}.npy"), trash_cell[i].cpu().numpy())
                    np.save(os.path.join(cell_dir, f"cell_state_{date_str}.npy"), cell_state[i].cpu().numpy())
                    saved_dates.add(date_str)

    df = pd.DataFrame(all_results, columns=["Time", "Observed", "Predicted"])
    df.to_csv(os.path.join(pred_dir, "test_preds_mclstm_hidden32_sq180_ep80_seed3407_cosannealing_retrain.csv"), index=False) # todo

    final_loss = test_loss / len(test_loader)
    print(f"Test Loss: {final_loss:.4f}")
    if wandb.run:
        wandb.log({"test_loss": final_loss})
    return final_loss
    
    
    
# ------------------------------------------------------------------
#  TRAIN  — lumped MCR-LSTM in the same style as train_lumped_mcmodel
# ------------------------------------------------------------------
def train_lumped_mcrmodel(model, train_loader, val_loader):
    """Full-training loop (early-stopping, ckpt) for the MCR-LSTM."""
    criterion  = NSELoss()
    optimizer  = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    #scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.NUM_EPOCHS,   # full cosine cycle = total epochs
        eta_min=1e-4            # final LR floor
    )

    best_val_loss     = float("inf")
    patience          = 10           # epochs without improvement
    no_improve_epochs = 0
    os.makedirs("checkpoints", exist_ok=True)
    best_state = None

    print("Starting MCR-LSTM Training")
    for epoch in range(cfg.NUM_EPOCHS):
        # -- TRAIN ------------------------------------------------
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            # loader can be (x, _, y, _)  OR  (x, y) – handle both
            if len(batch) == 4:
                x, _, y, _  = batch
            else:
                x, y        = batch

            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)

            # average spatial dims if present (B,T,C,H,W) ? (B,T,C)
            if x.ndim == 5:
                x = x.mean(dim=(-2, -1))

            xm, xa = x[..., :1], x[..., 1:]

            optimizer.zero_grad()
            m_out, c, *_ = model(xm, xa)                      # forward
            output = m_out[:, :, 1:].sum(dim=-1, keepdim=True)  # drop trash cell
            loss   = criterion(output[:, -1, :], y)           # NSE on final step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # -- VALIDATION -------------------------------------------
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    x, _, y, _ = batch
                else:
                    x, y       = batch
                x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
                if x.ndim == 5:
                    x = x.mean(dim=(-2, -1))
                xm, xa = x[..., :1], x[..., 1:]
                m_out, _, *_ = model(xm, xa)
                output = m_out[:, :, 1:].sum(dim=-1, keepdim=True)
                val_running += criterion(output[:, -1, :], y).item()

        val_loss = val_running / len(val_loader)
        scheduler.step() #val_loss

        print(f"Epoch {epoch+1}: Train {train_loss:.4f}, Val {val_loss:.4f}")

        # -- EARLY-STOP & CHECKPOINT -----------------------------
        if val_loss < best_val_loss:
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            best_val_loss      = val_loss
            no_improve_epochs  = 0
            ckpt_path = os.path.join("checkpoints",
                                     "mcrlstm_best_model_hidden32_sq90_ep50_seed3407_cosannealing_retrain.pkl") # todo
            ckpt = {
                "epoch"               : epoch + 1,
                "model_state_dict"    : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss"            : best_val_loss
            }
            with open(ckpt_path, "wb") as f:
                pickle.dump(ckpt, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved best model: {ckpt_path}")
        else:
            no_improve_epochs += 1
            print(f"  (no improvement {no_improve_epochs}/{patience})")
            if no_improve_epochs >= patience:
                print("Early stopping.")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Training complete (best val loss {best_val_loss:.4f}).")


# ------------------------------------------------------------------
#  TEST  — MCR-LSTM  (returns full internal states)
# ------------------------------------------------------------------
def test_lumped_mcrmodel(model, test_loader):
    """
    Evaluate MCR-LSTM on the test set.

    Returns
    -------
    obs           : Tensor   [N, 1]   observed discharge
    preds         : Tensor   [N, 1]   model prediction
    hidden_state  : Tensor   [N, H]   final hidden layer for every sample
    cell_state    : Tensor   [N, H]   final summed cell state
    MR_flow       : Tensor   [N, H]   MR flow at final step
    final_loss    : float    average NSE loss over the loader
    """
    criterion   = NSELoss()

    # output folders
    pred_dir  = "predictions"
    trash_dir = "trash_cell_outputs_mcrlstm"
    cell_dir  = "cell_state_outputs_mcrlstm"
    mr_dir    = "mr_flow_outputs_mcrlstm"
    os.makedirs(pred_dir,  exist_ok=True)
    os.makedirs(trash_dir, exist_ok=True)
    os.makedirs(cell_dir,  exist_ok=True)
    os.makedirs(mr_dir,  exist_ok=True)

    # lists to accumulate results
    all_rows, saved_dates = [], set()
    obs_lst, preds_lst, trashcell_lst, cell_lst = [], [], [], []
    MR_flow_lst = []
    test_running = 0.0

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # unpack loader (x, _, y, timestamps)
            x, _, y, timestamps = batch
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)

            # spatial average if 5-D
            if x.ndim == 5:
                x = x.mean(dim=(-2, -1))

            xm, xa = x[..., :1], x[..., 1:]
            m_out, c, o, mr, o_prime, mr_flow, o_flow = model(xm, xa)


            output = m_out[:, :, 1:].sum(dim=-1, keepdim=True)
            preds  = output[:, -1, :]                 # (B,1)
            trash  = m_out[:, :, 0]                   # (B,T)
            
            
            # loss
            test_running += criterion(preds, y).item()
            
            # CSV + per-day NPY (like old routine)
            for i in range(y.size(0)):
                ts   = pd.to_datetime(timestamps[i])
                date = ts.strftime("%Y%m%d")
                all_rows.append([ts, y[i].item(), preds[i].item()])
                
                '''
                if date not in saved_dates:
                    np.save(os.path.join(trash_dir, f"trash_cell_{date}.npy"),
                            trash[i].cpu().numpy())
                    np.save(os.path.join(cell_dir,  f"cell_state_{date}.npy"),
                            c [i].cpu().numpy())
                    np.save(os.path.join(mr_dir,    f"mr_flow_{date}.npy"),  
                            mr_flow[i].cpu().numpy())  
                                                
                    saved_dates.add(date)
                '''

    # write CSV with obs / preds
    pd.DataFrame(all_rows, columns=["Time", "Observed", "Predicted"])\
      .to_csv(os.path.join(pred_dir, "test_preds_mcrlstm_hidden32_sq90_ep50_seed3407_cosannealing_retrain.csv"), index=False) # todo

    final_loss = test_running / len(test_loader)
    print(f"Test Loss (NSE): {final_loss:.4f}")
    if wandb.run:
        wandb.log({"test_loss": final_loss})
    return final_loss
    
# -------------------------------------------------------------
#  TRAIN — lumped LSTM  (tuple-aware, pred shape = (B,1))
# -------------------------------------------------------------
def train_lumped_lstm(model: nn.Module,
                      train_loader,
                      val_loader):

    crit       = NSELoss()
    optimizer  = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
    '''
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.NUM_EPOCHS,   # full cosine cycle = total epochs
        eta_min=1e-4            # final LR floor
    )
    '''

    best_val, patience, no_improve = float("inf"), 10, 0
    os.makedirs("checkpoints", exist_ok=True)
    best_state = None

    print("Starting LSTM training")
    for epoch in range(cfg.NUM_EPOCHS):
        # ---- TRAIN ----
        model.train()
        tr_running = 0.0
        for batch in train_loader:
            x, y = (batch[0], batch[2]) if len(batch) == 4 else batch
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)

            if x.ndim == 5:                         # spatial mean
                x = x.mean(dim=(-2, -1))

            optimizer.zero_grad()
            stream_seq, *_ = model(x)               # tuple ? first element
            pred = stream_seq[:, -1].unsqueeze(-1)   # (B,1) to match y
            
            loss = crit(pred, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_running += loss.item()
        train_loss = tr_running / len(train_loader)

        # ---- VALIDATION ----
        model.eval()
        va_running = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, y = (batch[0], batch[2]) if len(batch) == 4 else batch
                x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
                if x.ndim == 5:
                    x = x.mean(dim=(-2, -1))
                stream_seq, *_ = model(x)
                pred = stream_seq[:, -1].unsqueeze(-1)
                va_running += crit(pred, y).item()
        val_loss = va_running / len(val_loader)
        scheduler.step(val_loss) #val_loss

        print(f"Epoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f}")

        # ---- EARLY-STOP / CKPT ----
        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            ckpt_path = os.path.join("checkpoints", "lstm_best_model_hidden64_sq180_ep200_seed200_rol_retrain.pkl") # todo
            with open(ckpt_path, "wb") as f:
                pickle.dump({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"saved best model {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Training complete (best val loss {best_val:.4f}).")


# -------------------------------------------------------------
#  TEST — lumped LSTM
# -------------------------------------------------------------
def test_lumped_lstm(model: nn.Module,
                     test_loader,
                     scaler_pickle: str):
    """
    Evaluate the live LSTM and save obs / preds in *cfs*.
    using the basin area for cfg.BASIN_ID.
    """
    # -- helpers ----------------------------------------------------------
    def _load_scaler(pkl):
        import pickle, pathlib
        with open(pathlib.Path(pkl), "rb") as f:
            return pickle.load(f)

    def _inverse_scale_Q(z, stats):
        return z * stats["q_std"] + stats["q_mean"]       # mm d?¹

    def _catchment_area_km2():
        import pandas as pd
        attr = Path(cfg.BASIN_DATA_ROOT) / "catchment_info.txt"
        df   = pd.read_csv(attr, sep=";", dtype={"gauge_id": str})
        return float(df.loc[df["basin_id"].astype(str) == cfg.BASIN_ID,
                            "area_gages2"].values[0])

    # --------------------------------------------------------------------
    stats   = _load_scaler(scaler_pickle)
    area_km2 = _catchment_area_km2()
    mm_to_cfs = area_km2 * 0.40873                         # unit factor

    crit  = NSELoss()
    pred_dir = "predictions"; os.makedirs(pred_dir, exist_ok=True)
    rows, running = [], 0.0

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4:
                x, _, y, ts = batch
            else:
                x, y = batch; ts = [pd.NaT] * y.size(0)

            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            if x.ndim == 5:
                x = x.mean(dim=(-2, -1))

            stream_seq, *_ = model(x)
            z_pred = stream_seq[:, -1].unsqueeze(-1)       # (B,1) mm z-score

            pred_mm = _inverse_scale_Q(z_pred, stats)
            obs_mm  = _inverse_scale_Q(y,       stats)

            # convert to cfs
            pred = pred_mm * mm_to_cfs
            obs  = obs_mm  * mm_to_cfs

            running += crit(pred, obs).item()

            # move to CPU for CSV
            pred_cpu, obs_cpu = pred.cpu(), obs.cpu()
            for i in range(obs_cpu.size(0)):
                rows.append([pd.to_datetime(ts[i]),
                             float(obs_cpu[i]),
                             float(pred_cpu[i])])

    pd.DataFrame(rows, columns=["Time", "Observed", "Predicted"])\
      .to_csv(os.path.join(pred_dir, "test_preds_lstm_hidden64_sq180_ep200_seed200_rol_retrain.csv"), index=False) # todo

    final_loss = running / len(test_loader)
    print(f"Test Loss (NSE, cfs units): {final_loss:.4f}")
    if wandb.run:
        wandb.log({"test_loss": final_loss})
    return final_loss





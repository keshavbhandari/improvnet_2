import argparse
import torch
from torch.utils.data import DataLoader

# Import configs, processor, and the EXACT dataset from the training script
from improvnet.model.twotower_config import *
from improvnet.utils.ar_utils import ProcessData
from improvnet.model.twotower_train import TwoTowerDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run the dataloader debugger")
    args = parser.parse_args()

    processor = ProcessData()
    
    # Using the exact dataset class from twotower_train.py
    dataset = TwoTowerDataset(JSONL_FILES, processor, augment=False)
    
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, 
        collate_fn=dataset.collate_fn
    )

    print("\n" + "="*100)
    print(" TwoTower Dataloader Debugger (Exact Parity) ")
    print("="*100)

    for batch in loader:
        # Extract the first item in the batch
        prefix_tensor = batch["prefix"][0]
        traj_input = batch["draft_traj"][0]
        traj_target = batch["targets"][0]
        traj_ts = batch["timesteps"][0]
        traj_wt = batch["weights"][0]
        
        # Reconstruct the raw prefix strings directly from the tensor to verify the pipeline
        raw_prefix = [processor.tokenizer.id_to_tok.get(tid.item(), f"ID:{tid.item()}") 
                      for tid in prefix_tensor if tid.item() != PAD_ID]
        
        print(f"\n--- PREFIX (Length: {len(raw_prefix)}) ---")
        # Print a snapshot of the prefix (last 20 tokens to save console space)
        if len(raw_prefix) > 20:
            print("... (truncated) ...")
            for tok in raw_prefix[-20:]:
                print(f"  {tok}")
        else:
            for tok in raw_prefix:
                print(f"  {tok}")
        
        print(f"\n--- NON-MARKOVIAN TRAJECTORY (Denoiser Input) ---")
        print("-" * 100)
        print(f"{'Idx':<5} | {'Time':<5} | {'Weight':<6} | {'Input Token':<35} | {'Target (Ground Truth)':<35}")
        print("-" * 100)
        
        current_draft = 1
        
        for i in range(len(traj_input)):
            inp_id = traj_input[i].item()
            tgt_id = traj_target[i].item()
            ts_val = traj_ts[i].item()
            wt_val = traj_wt[i].item()
            
            inp_tok = processor.tokenizer.id_to_tok.get(inp_id, f"ID:{inp_id}")
            tgt_tok = processor.tokenizer.id_to_tok.get(tgt_id, f"ID:{tgt_id}")
            
            # Print a clean separator whenever we hit a new draft (indicated by <SEP>)
            if inp_id == SEP_ID:
                current_draft += 1
                print("-" * 100)
                print(f"--- ENTERING DRAFT {current_draft} (t={ts_val:.3f}) ---")
                print("-" * 100)
                continue
                
            # Skip bulk padding to keep console output readable
            if inp_id == PAD_ID and tgt_id == PAD_ID:
                if i > 0 and traj_input[i-1].item() != PAD_ID:
                    print(f"{i:<5} | {ts_val:.3f} | {wt_val:.3f} | {'<PAD> ...':<35} | {'<PAD> ...':<35}")
                continue
                
            # Formatting strings for clean columns
            inp_str = str(inp_tok)
            tgt_str = str(tgt_tok)
            
            print(f"{i:<5} | {ts_val:.3f} | {wt_val:.3f} | {inp_str:<35} | {tgt_str:<35}")
            
        print("\nDebugger complete. Exiting.\n")
        break # Only process one batch
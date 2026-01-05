import torch
import numpy as np
from chess_env import chess_gym
from datasets import load_dataset
from torch.utils.data import IterableDataset

def reward_bin(arr: np.ndarray) -> np.ndarray:
    return np.clip((arr + 1024) >> 4, 0, 127).astype(np.int64)

def upload_data(batch):
    env = chess_gym()
    
    obs = []
    policy_labels = []
    value_labels = []
    
    curr_obs = []
    curr_policy_labels = []
    curr_value_labels = []
    
    fens = batch['fen']
    batch_size = len(fens)
    
    evals = batch['cp']
    lines = batch['line']
    mates = batch['mate']
    
    for i in range(batch_size):
        # Set env to current fen
        # Run through the entire line, giving same CP
        # Upload to policy and value labels
        first_obs, _ = env.reset(fen=fens[i])
        start_turn = env.get_turn() # 1 if white
        
        # Check for mate
        curr_eval = evals[i]
        if evals[i] is None:
            curr_eval = 10000 if mates[i] > 0 else -10000
        line_cp = start_turn * curr_eval
        
        # Append first one
        curr_obs.append(first_obs)
        curr_value_labels.append(line_cp)
        
        curr_line = lines[i].strip().split(" ")
        for m in range(len(curr_line)):
            move = curr_line[m]
            
            action_id = env.uci_to_action(move)
            curr_policy_labels.append(action_id)
            
            # Policy will always be 1 short of obs-value, so dont append last one
            if m == len(curr_line) - 1:
                break
            
            # Check if we are making an illegal move
            if env.get_illegal_mask()[action_id] == 1:
                curr_obs = []
                curr_policy_labels = []
                curr_value_labels = []
                break
            
            try:
                new_obs, _, term, trunc, _ = env.step(action_id)
                
                if term or trunc:
                    break
            except ValueError as e:
                assert len(curr_value_labels) == len(curr_policy_labels)
                break
            
            # Append obs: note its next move so cp *= -1
            line_cp *= -1
            curr_obs.append(new_obs)
            curr_value_labels.append(line_cp)
        
        # Append to actual obs labels etc.
        obs.extend(curr_obs)
        value_labels.extend(curr_value_labels)
        policy_labels.extend(curr_policy_labels)
        
        curr_obs = []
        curr_policy_labels = []
        curr_value_labels = []
    
    assert len(value_labels) == len(policy_labels) == len(obs)
    obs_tensor = torch.from_numpy(np.stack(obs).astype(np.float32))
    value_tensor = torch.from_numpy(reward_bin(np.stack(value_labels)))
    policy_tensor = torch.from_numpy(np.stack(policy_labels).astype(np.int64))
    
    return obs_tensor, value_tensor, policy_tensor        

class ChessStreamDataset(IterableDataset):
    def __init__(self, streaming_dataset, source_chunk_size, target_batch_size):
        self.dataset = streaming_dataset
        # source_chunk_size: How many raw items to download at once (e.g. 512)
        # This will expand to ~5120 items
        self.source_chunk_size = source_chunk_size
        
        # target_batch_size: What the GPU actually needs (e.g. 1024)
        self.target_batch_size = target_batch_size

    def __iter__(self):
        stream = self.dataset
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            stream = stream.shard(num_shards=worker_info.num_workers, index=worker_info.id)
        
        # 1. Get a "Macro Batch" from the stream
        # If we grab 256 raw items, upload_data expands them to ~2,560 tensors
        batched_ds = stream.batch(batch_size=self.source_chunk_size, drop_last_batch=True)
        
        for batch in batched_ds:
            # 2. Process: o, v, p are now LARGE tensors (approx 10x source_chunk_size)
            # Example shape: o = [2560, 111, 8, 8]
            o, v, p = upload_data(batch)
            
            total_samples = o.shape[0]
            
            # 3. Slice the large tensor into exact GPU-sized batches
            # We step through the large tensor in jumps of 'target_batch_size'
            for i in range(0, total_samples, self.target_batch_size):
                end_idx = i + self.target_batch_size
                
                # If we don't have enough data left for a full batch, drop the remainder
                # (This ensures the GPU never gets a weird sized batch like 432)
                if end_idx > total_samples:
                    break
                
                # Yield the exact slice. 
                # Since these are already Tensors, this is fast (views).
                yield o[i:end_idx], v[i:end_idx], p[i:end_idx]
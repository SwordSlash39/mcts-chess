import numpy as np
import torch, pygame, time
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from stockfish import Stockfish
"""
/usr/games/stockfish
"""
from old_model import *
from chess_env import chess_gym

torch.set_printoptions(threshold=float('inf'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyChessTransformer().to(device).to(torch.float32)
model.load_checkpoint("TinyChessTransformer.pt")
model.eval()


with open('puzzles.txt', 'r', encoding='utf-8') as f:
    # Use a list comprehension for speed
    puzzles = [line.strip() for line in f]
    
eff_dtype = torch.bfloat16

R_LIST = ["human", "rgb_array"]
RENDER = "rgb_array"

env = chess_gym(render_mode=RENDER)
total = 0
checkmates = 0

correct_legal = 0
total_moves = 0
for p in puzzles:
    observation, info = env.reset(fen=p)
    
    hasCheckmate = False
    for i in range(16):
        illegalMask = torch.as_tensor(env.get_illegal_mask(), dtype=torch.bool, device=device)
        obv_tensor = torch.as_tensor(observation.copy(), device=device, dtype=torch.float32).unsqueeze(0)
        
        with torch.inference_mode():
            logits, _ = model(obv_tensor, training=False)
            logits = logits.view(-1)
            
            # Check correct legal
            full_argmax = torch.argmax(logits).item()
            
            logits[illegalMask] = -1e9
            
            if logits[full_argmax] > -1e8:
                correct_legal += 1
            total_moves += 1
        
        action = torch.argmax(logits).item()
        
        observation, reward, termination, truncation, info = env.step(action)
        if reward != 0:
            hasCheckmate = True
        
        if termination or truncation:
            break
    
    total += 1
    if hasCheckmate:
        checkmates += 1
    
    if total % 5 == 0:
        print(f"Puzzles: {total}\nAccuracy: {(checkmates * 100 / total):.2f}%\nLegal move accuracy: {(correct_legal * 100 / total_moves):.2f}%")


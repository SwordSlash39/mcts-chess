import numpy as np
import torch
import math
import sys
import threading
import pygame
import chess
import chess.pgn

# --- Imports ---
try:
    from chess_env import chess_gym
    from old_model import TinyChessTransformer
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# --- Configuration ---
C_PUCT = 1.5
NUM_SIMULATIONS = 800
BATCH_SIZE = 8          # Evaluate 16 positions in one GPU call
MAX_CENTIPAWN = 1024.0 
VIRTUAL_LOSS = 1.0        # Penalty applied to nodes currently being explored
FPU_DELTA = 0.2

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

torch.set_float32_matmul_precision('high')
compile_options = {
    "triton.cudagraphs": False,
    "epilogue_fusion": True,
    "freezing": True,
    "shape_padding": True,
    "layout_optimization": False, 
}

model = TinyChessTransformer().to(device).to(torch.bfloat16)
try:
    model.load_checkpoint("TinyChessTransformer.pt")
    print("Model loaded.")
except:
    print("Using random weights.")
    
model.eval()
model.forward = torch.compile(
    model.forward_no_ckpt,
    options=compile_options
)

print("Compiling...")
dummy_input = torch.rand((BATCH_SIZE, 111, 8, 8), dtype=torch.bfloat16).to(device)
with torch.inference_mode():
    _ = model(dummy_input)
print("Compiled model.")

# Global Lock
board_lock = threading.Lock()
global_running = True

# Batched mcts
class VectorizedMCTSNode:
    def __init__(self, fen, prior_probs, legal_actions):
        self.fen = fen
        self.legal_actions = legal_actions
        self.p_priors = prior_probs
        
        # Stats
        self.n_visits = np.zeros_like(prior_probs, dtype=np.int32)
        self.w_sums   = np.zeros_like(prior_probs, dtype=np.float32)
        self.q_values = np.zeros_like(prior_probs, dtype=np.float32)
        
        self.children = {} # Map action_id -> VectorizedMCTSNode
        self.is_expanded = False

    def best_action_ucb(self, c_puct, remaining_sims, is_root=False):
        parent_visits = np.sum(self.n_visits)
        sqrt_parent = math.sqrt(parent_visits) if parent_visits > 0 else 1.0
        
        if parent_visits > 0:
            # Use mean Q of visited moves or a default low value
            v_fpu = np.mean(self.q_values[self.n_visits > 0]) - FPU_DELTA
        else:
            v_fpu = 0.0

        # 2. Apply FPU to moves with 0 visits
        adjusted_q = np.where(self.n_visits > 0, self.q_values, v_fpu)
        
        # UCB = Q + U
        # Note: self.q_values is updated dynamically during backprop
        u_scores = c_puct * self.p_priors * sqrt_parent / (1.0 + self.n_visits)
        ucb_scores = adjusted_q + u_scores
        
        # Mask out shit moves if we are looking at root
        if is_root and parent_visits > 0:
            max_visits = np.max(self.n_visits)
            impossible_to_win = (self.n_visits + remaining_sims) < max_visits
            ucb_scores[impossible_to_win] = -np.inf 
        
        best_idx = np.argmax(ucb_scores)
        return self.legal_actions[best_idx], best_idx

class BatchedMCTS:
    def __init__(self, model, device, num_simulations, batch_size):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.value_bins = np.array([k * 16 - 1024 for k in range(128)], dtype=np.float32)
        
        self.root_node = None
        self.transposition_table = {}
        
        # Pre-allocate environments for the batch to save init time
        self.sim_envs = [chess_gym(render_mode="rgb_array") for _ in range(batch_size)]

    def run(self, root_env):
        root_board = root_env.env.unwrapped.board
        root_fen = root_board.fen()
            
        # 1. Expand Root if needed
        if root_fen in self.transposition_table:
            self.root_node = self.transposition_table[root_fen]
        else:
            self.root_node = self._expand_node_from_env(root_env)
            self.transposition_table[root_fen] = self.root_node
        
        # Prune table
        self._prune_tp_table()
        
        # Calculate how many batches needed
        # Since there is persistence, we search less for some nodes
        remaining_sims = self.num_simulations - np.sum(self.root_node.n_visits)
        print(f"\n\nRemaining sims to search: {remaining_sims}")
        num_batches = math.ceil(remaining_sims / self.batch_size)
        
        for i in range(num_batches):
            sims_left = self.num_simulations - (i * self.batch_size)
            self._run_batch(self.root_node, root_board.copy(), sims_left)
            
            # Quick stopping
            if len(self.root_node.n_visits) > 1:
                # Efficiently get top two visit counts
                top_two = np.partition(self.root_node.n_visits, -2)[-2:]
                v_max = top_two[1]
                v_2nd = top_two[0]
                
                sims_left = self.num_simulations - ((i + 1) * self.batch_size)
                
                # If runner-up + all remaining sims cannot beat leader, exit
                if v_2nd + sims_left < v_max:
                    break 
            
        if len(self.root_node.legal_actions) == 0:
            return None
            
        best_idx = np.argmax(self.root_node.n_visits)
        best_action = self.root_node.legal_actions[best_idx]
        return best_action

    def _run_batch(self, root_node, root_board, sims_left):
        paths = []
        leaf_nodes = []
        ready_envs = []
        
        # --- 1. SELECTION (Serial CPU, but fast) ---
        # We find 'batch_size' leaf nodes to evaluate
        for i in range(self.batch_size):
            env = self.sim_envs[i]
            env.reset_with_board(root_board) # Fast Reset
            
            node = root_node
            path = [] # List of (node, action_index_in_node)
            
            # Traverse
            is_root_node = True
            curr_sims_left = sims_left - i
            while node.is_expanded:
                # Check terminal at board level
                if env.is_game_over():
                    break
                
                action, idx = node.best_action_ucb(C_PUCT, remaining_sims=curr_sims_left, is_root=is_root_node)
                if is_root_node:
                    is_root_node = False
                
                # Apply VIRTUAL LOSS
                # We pretend we visited this node so the next iteration in this batch 
                # picks a different path.
                node.n_visits[idx] += 1
                node.w_sums[idx] -= VIRTUAL_LOSS 
                node.q_values[idx] = node.w_sums[idx] / node.n_visits[idx]
                
                path.append((node, idx))
                
                env.step(action)
                
                if action in node.children:
                    node = node.children[action]
                else:
                    curr_fen = env.env.unwrapped.board.fen()
                    if curr_fen in self.transposition_table:
                        child = self.transposition_table[curr_fen]
                        node.children[action] = child
                        node = child
                    else:
                        # Found a leaf (unexpanded child)
                        # Create the node object but don't expand yet
                        child = VectorizedMCTSNode(curr_fen, np.array([]), np.array([]))
                        
                        node.children[action] = child
                        node = child
                        break
            
            paths.append(path)
            leaf_nodes.append(node)
            ready_envs.append(env)

        # --- 2. EVALUATION (Parallel GPU) ---
        # Collect observations
        obs_array = np.zeros((self.batch_size, *self.sim_envs[0].obv.shape), dtype=np.float32)
        valid_indices = []
        have_obs = False
        
        for i, env in enumerate(ready_envs):
            # Check if game ended naturally (checkmate) before evaluation
            if env.is_game_over():
                # No NN eval needed, value is deterministic
                continue
                
            obs_array[i] = env.obv
            valid_indices.append(i)
            have_obs = True
        
        values = np.zeros(self.batch_size) # Default 0 for terminators
        
        # Only run model if we have valid non-terminal states
        if have_obs:
            batch_tensor = torch.as_tensor(obs_array, dtype=torch.bfloat16, device=self.device)
            
            with torch.inference_mode():
                logits_batch, val_bins_batch = self.model(batch_tensor)
                
                # Bring to float32
                logits_batch = logits_batch.float()
                val_bins_batch = val_bins_batch.float()
                
                # Process Value Heads
                val_probs = torch.softmax(val_bins_batch, dim=-1).cpu().numpy() # (B, 128)
                expected_cp = np.sum(val_probs * self.value_bins, axis=1) # (B,)
                
                # Normalize [-1, 1]
                # Note: We need the turn from each env to normalize correctly
                for k, batch_idx in enumerate(valid_indices):
                    env = ready_envs[batch_idx]
                    turn = env.get_turn()
                    norm_val = np.clip((expected_cp[k] / MAX_CENTIPAWN), -1.0, 1.0)
                    values[batch_idx] = norm_val
                    
                    # Process Policy Heads & Expand
                    self._expand_node_with_logits(leaf_nodes[batch_idx], logits_batch[k], env.get_legal_mask())

        # Handle terminals (Game Over) values
        for i, env in enumerate(ready_envs):
            board = env.env.unwrapped.board
            if env.is_game_over():
                # Calculate terminal result
                res = board.result() # worst case its just draws, could bug but not impt
                turn = env.get_turn()
                if res == "1-0": values[i] = 1.0 * turn
                elif res == "0-1": values[i] = -1.0 * turn
                else: values[i] = 0.0

        # --- 3. BACKPROPAGATION (Serial CPU) ---
        for i in range(self.batch_size):
            value = values[i]
            path = paths[i]
            
            # The value we got is for the player at the leaf.
            # We traverse up. For each step up, the perspective flips.
            # However, standard AlphaZero:
            # If leaf is white's turn, value is white-relative.
            # Parent is Black (who moved to get here).
            # We want Q to represent expected return for the player at that node.
            
            # Simplified Alternating Backprop:
            # Leaf Value is 'V'.
            # Parent of leaf (who played move) -> Update with -V.
            # Grandparent -> Update with V.
            
            curr_val = -value # First backprop is to parent, so flip
            
            for node, idx in reversed(path):
                # Remove VIRTUAL LOSS (add it back)
                node.w_sums[idx] += VIRTUAL_LOSS
                
                # Add REAL Value
                node.w_sums[idx] += curr_val
                
                # Note: n_visits was already incremented in selection
                # Recalculate Q
                node.q_values[idx] = node.w_sums[idx] / node.n_visits[idx]
                
                curr_val = -curr_val

    def _expand_node_from_env(self, env: chess_gym):
        """Helper to create a root node."""
        node = VectorizedMCTSNode(env.env.unwrapped.board.fen(), np.array([]), np.array([]))
        
        padded_obs = np.zeros((self.batch_size, *env.obv.shape), dtype=np.float32)
        padded_obs[0] = env.obv
        obs_tensor = torch.as_tensor(padded_obs, dtype=torch.bfloat16, device=self.device)
        with torch.inference_mode():
            logits, _ = self.model(obs_tensor)
            logits = logits.float()
            
            self._expand_node_with_logits(node, logits[0], env.get_legal_mask())
            
        return node

    def _expand_node_with_logits(self, node: VectorizedMCTSNode, logits_tensor, legal_mask_np):
        """Populates a node with policy data from model output."""
        # Check if node is already expanded
        if node.is_expanded:
            return
        
        # append to transposition table
        self.transposition_table[node.fen] = node
        
        logits = logits_tensor.cpu().numpy()
        legal_indices = np.flatnonzero(legal_mask_np)
        
        if len(legal_indices) == 0:
            node.is_expanded = True
            return

        legal_logits = logits[legal_indices]
        # Stable Softmax
        exp_logits = np.exp(legal_logits - np.max(legal_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        node.legal_actions = legal_indices
        node.p_priors = probs
        node.n_visits = np.zeros_like(probs, dtype=np.int32)
        node.w_sums = np.zeros_like(probs, dtype=np.float32)
        node.q_values = np.zeros_like(probs, dtype=np.float32)
        node.children = {}
        node.is_expanded = True
    
    def _prune_tp_table(self):
        reachable_fens = set()
        new_tb = {}
        
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if node.fen not in reachable_fens:
                assert node.fen is not None
                reachable_fens.add(node.fen)
                new_tb[node.fen] = node
                
                # Add to stack
                stack.extend(node.children.values())
        
        self.transposition_table = new_tb
# ==========================================
# THREADED GAME LOGIC (Same as before)
# ==========================================

def get_player_move_pgn(env):
    board = env.env.unwrapped.board
    while global_running:
        try:
            for _ in range(100):
                user_input = input(f"\nYour move (Nodes: {NUM_SIMULATIONS}): ").strip()
                if user_input.lower() in ['quit', 'exit']: return None
                elif user_input.lower() in ['nodes']:
                    NUM_SIMULATIONS = int(input(f"\nEnter new node search: "))
                else:
                    break
            
            move = board.parse_san(user_input)
            action = env.uci_to_action(move.uci())
            
            mask = env.get_legal_mask()
            if mask[action] == 1: return action
            else: print(f"Move {user_input} illegal in mask.")
        except ValueError: print(f"Invalid move.")
        except Exception as e: print(f"Error: {e}")
    return None

import time
def game_logic_thread(env, mcts, player_is_white):
    global global_running
    print("Logic Thread Started.")
    
    while global_running:
        with board_lock:
            board = env.env.unwrapped.board
            if board.is_game_over() or env.is_game_over():
                print(f"Game Over! Result: {board.result()}")
                break
            is_white_turn = (board.turn == chess.WHITE)
            
        if (is_white_turn and player_is_white) or (not is_white_turn and not player_is_white):
            action = get_player_move_pgn(env)
            if action is None: break
            with board_lock:
                print(f"Player: {env.action_to_uci(action)}")
                env.step(action)
        else:
            print("AI Thinking (Batched)...", end="\r")
            
            # --- AI RUNS HERE ---
            # t = time.time()
            action = mcts.run(env)
            # --------------------
            # print(str(time.time() - t) + " " * 60)
            if action is None: break
            with board_lock:
                print(f"AI: {env.action_to_uci(action)}" + " " * 60)
                env.step(action)

    global_running = False

def main():
    global global_running
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    
    model = TinyChessTransformer().to(device).to(torch.float32)
    try:
        model.load_checkpoint("TinyChessTransformer.pt")
        print("Model loaded.")
    except:
        print("Using random weights.")
    model.eval()

    # Human Render Env
    env = chess_gym(render_mode="human")
    env.reset(
        # fen="r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    )
    
    try:
        # Batched MCTS
        mcts = BatchedMCTS(model, device, num_simulations=NUM_SIMULATIONS, batch_size=BATCH_SIZE)
        
        side = input("Play as White (w) or Black (b)? ").lower()
        player_is_white = side.startswith('w')

        t = threading.Thread(target=game_logic_thread, args=(env, mcts, player_is_white))
        t.daemon = True
        t.start()
        
        clock = pygame.time.Clock()
        while global_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    global_running = False
            with board_lock:
                try: env.render()
                except: pass
            clock.tick(30)
    except KeyboardInterrupt:
        pass
    
    final_board = env.env.unwrapped.board
    game = chess.pgn.Game.from_board(final_board)

    # 3. Add some headers (optional)
    game.headers["Event"] = "PettingZoo AI Match"
    game.headers["Site"] = "Local Machine"
    game.headers["White"] = "idk"
    game.headers["Black"] = "idk"
    game.headers["Result"] = final_board.result()

    # 4. Write to pgn.txt
    with open("pgn.txt", "w") as f:
        f.write(str(game))

    print("PGN saved to pgn.txt")
        
    env.close()

if __name__ == "__main__":
    main()
import berserk
import threading
import chess
import chess.polyglot
import torch, math, time, os
import numpy as np

# --- Import your existing logic ---
# Ensure your provided code is saved as bot_logic.py 
# or paste your classes (chess_gym, BatchedMCTS, TinyChessTransformer, etc.) here.
from chess_env import chess_gym, game_over, action_idx_to_move
from model import TinyChessTransformer

# --- Configuration ---
API_TOKEN = "<insert here>"
C_PUCT = 1.2
NUM_SIMULATIONS = 1000
BATCH_SIZE = 8          # Evaluate 16 positions in one GPU call
MAX_CENTIPAWN = 1024.0 
VIRTUAL_LOSS = 1.0        # Penalty applied to nodes currently being explored
MAX_GAMES = 2
FPU_DELTA = 0.1

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

    def run(self, root_env): # Swapcolor true IF AND ONLY IF SELFPLAY
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
        print(f"Remaining sims to search: {remaining_sims}")
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

    def _run_batch(self, root_node, root_board: chess.Board, sims_left):
        paths = []
        leaf_nodes = []
        ready_envs = []
        
        # --- 1. SELECTION (Serial CPU, but fast) ---
        # We find 'batch_size' leaf nodes to evaluate
        for i in range(self.batch_size):
            env = root_board.copy()
            
            node = root_node
            path = [] # List of (node, action_index_in_node)
            
            # Traverse
            is_root_node = True
            curr_sims_left = sims_left - i
            while node.is_expanded:
                # Check terminal at board level
                if game_over(env):
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
                
                move = action_idx_to_move(env, action)
                env.push(move)
                
                if action in node.children:
                    node = node.children[action]
                else:
                    curr_fen = env.fen()
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
            
            self.sim_envs[i].reset_with_moves(env)
            chess_env = self.sim_envs[i]
            ready_envs.append(chess_env)

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

# --- Global Model Init (Load once) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on {device}...")

model = TinyChessTransformer().to(device).to(torch.bfloat16) # Match your training dtype
try:
    model.load_checkpoint("TinyChessTransformer.pt")
    print("Model loaded successfully.")
except:
    print("Warning: Could not load checkpoint, using random weights.")
model.eval()

# Helper to compile model (optional, matches your original code)
compile_options = {
    "triton.cudagraphs": False,
    "epilogue_fusion": True,
    "freezing": True,
    "shape_padding": True,
    "layout_optimization": False, 
}
model.forward = torch.compile(model.forward_no_ckpt, options=compile_options)

# Warmup
dummy_input = torch.rand((BATCH_SIZE, 111, 8, 8), dtype=torch.bfloat16).to(device)
with torch.inference_mode():
    _ = model(dummy_input)
print("Model compiled and ready.")

class LichessBot:
    def __init__(self, token):
        global MAX_GAMES
        self.session = berserk.TokenSession(token)
        self.client = berserk.Client(self.session)
        self.my_id = self.client.account.get()['id']
        
        # Track running games
        self.running_game_ids = set()
        
        try:
            self.book = chess.polyglot.open_reader("OPTIMUS4.bin")
            print("Opening book loaded successfully.")
        except FileNotFoundError:
            self.book = None
            print("Warning: 'OPTIMUS4.bin' not found. Bot will not use opening book.")
        
        self.active_games = 0
        self.pending_challenges = {}
        self.lock = threading.Lock()
        print(f"Logged in as {self.my_id} | Mode: {MAX_GAMES} Game Max | Rapid Only")

    def run(self):
        self._check_ongoing_games()
        print("Listening for events...")
        # This loop blocks forever, which is why we need os._exit(0) to stop it
        for event in self.client.bots.stream_incoming_events():
            if event['type'] == 'challenge':
                self.handle_challenge(event['challenge'])
            elif event['type'] == 'gameStart':
                print(f"Game started: {event['game']['id']}")
                self.start_game_thread(event['game']['id'])
    
    def start_game_thread(self, game_id):
        with self.lock:
            if game_id in self.running_game_ids:
                return 
            self.active_games += 1
            self.running_game_ids.add(game_id)
        
        print(f"Starting thread for game: {game_id}")
        game_thread = threading.Thread(target=self.play_game, args=(game_id,))
        game_thread.start()
    
    def _check_ongoing_games(self):
        print("Checking for ongoing games...")
        try:
            ongoing_games = self.client.games.get_ongoing()
            for game in ongoing_games:
                game_id = game['gameId']
                print(f" -> Found ongoing game: {game_id}")
                self.start_game_thread(game_id)
            if self.active_games == 0:
                print(" -> No ongoing games found.")
        except Exception as e:
            print(f"Error checking ongoing games: {e}")
    
    def send_challenge(self, username, minutes, increment):
        print(f"Attempting to challenge {username} ({minutes}+{increment})...")
        try:
            if username in self.pending_challenges:
                print(f" -> Pending challenge exists for {username}. Cancel it first.")
                return
            
            response = self.client.challenges.create(
                username, rated=True, clock_limit=minutes*60, clock_increment=increment, color='random'
            )
            c_id = response.get('id')
            if c_id:
                self.pending_challenges[username] = c_id
                print(f" -> Challenge sent to {username}! (ID: {c_id})")
        except berserk.exceptions.ResponseError as e:
            print(f" -> Failed to challenge {username}: {e}")
    
    def cancel_challenge(self, target_username="all"):
        def _do_cancel(user, c_id):
            print(f"Canceling challenge to {user}...")
            try:
                self.client.challenges.cancel(c_id)
                print(" -> Cancelled.")
            except Exception as e:
                print(f" -> Error: {e}")
            if user in self.pending_challenges:
                del self.pending_challenges[user]

        if target_username == "all":
            if not self.pending_challenges:
                print("No pending challenges.")
                return
            for user in list(self.pending_challenges.keys()):
                _do_cancel(user, self.pending_challenges[user])
        else:
            if target_username in self.pending_challenges:
                _do_cancel(target_username, self.pending_challenges[target_username])
            else:
                print(f"No pending challenge found for '{target_username}'.")

    def resign_game(self, opponent_name):
        print(f"Looking for active games against: {opponent_name}...")
        try:
            ongoing = self.client.games.get_ongoing()
            found = False
            for game in ongoing:
                opp = game.get('opponent', {})
                current_opp_name = opp.get('username', 'Unknown')
                g_id = game['gameId']

                if opponent_name == "all" or current_opp_name.lower() == opponent_name.lower():
                    print(f" -> Resigning game {g_id} vs {current_opp_name}...")
                    try:
                        self.client.bots.resign_game(g_id)
                        print(" -> Resigned.")
                        found = True
                    except Exception as e:
                        print(f" -> Failed to resign: {e}")

            if not found and opponent_name != "all":
                print(f" -> No active game found against {opponent_name}.")
        except Exception as e:
            print(f"Error fetching games: {e}")

    def handle_challenge(self, challenge):
        c_id = challenge['id']
        challenger_name = challenge['challenger']['name']
        if challenger_name.lower() == self.my_id.lower(): return
        
        speed = challenge['speed']
        print(f"Challenge from {challenger_name} ({speed})")

        with self.lock:
            if self.active_games >= MAX_GAMES:
                print(f" -> Declined: Full")
                self.client.bots.decline_challenge(c_id, reason='later')
                return

        if speed != 'rapid':
            print(f" -> Declined: Time Control")
            self.client.bots.decline_challenge(c_id, reason='timeControl')
            return

        if challenge['variant']['key'] != 'standard':
            print(f" -> Declined: Variant")
            self.client.bots.decline_challenge(c_id, reason='variant')
            return

        print(" -> Accepted!")
        self.client.bots.accept_challenge(c_id)

    def play_game(self, game_id):
        try:
            local_mcts = BatchedMCTS(model, device, batch_size=BATCH_SIZE, num_simulations=NUM_SIMULATIONS)
            env = chess_gym(render_mode="rgb_array")
            stream = self.client.bots.stream_game_state(game_id)
            current_board = chess.Board()
            is_white = False 

            for event in stream:
                if event['type'] == 'gameFull':
                    is_white = event['white']['id'] == self.my_id
                    self.process_state(game_id, event['state'], current_board, env, is_white, local_mcts)
                elif event['type'] == 'gameState':
                    self.process_state(game_id, event, current_board, env, is_white, local_mcts)
                    if event['status'] != 'started': break

        except Exception as e:
            print(f"Game {game_id} Error: {e}")
        finally:
            with self.lock:
                self.active_games -= 1
                if game_id in self.running_game_ids:
                    self.running_game_ids.remove(game_id)
            print(f"--- Game Finished: {game_id} ---")
    
    def make_move_with_retry(self, game_id, uci_move, max_retries=5):
        for i in range(max_retries):
            try:
                self.client.bots.make_move(game_id, uci_move)
                return True 
            except Exception as e:
                error_str = str(e)
                if "Not your turn" in error_str or "illegal move" in error_str.lower():
                    print(f"[{game_id}] Move likely already accepted.")
                    return True
                print(f"[{game_id}] Retry ({i+1}): {e}")
                time.sleep(0.5)
        print(f"[{game_id}] FAILED to send move.")
        return False

    def process_state(self, game_id, state, board, env, is_white, mcts_instance):
        moves = state['moves']
        board.reset()
        if moves:
            for move_uci in moves.split(' '):
                board.push_uci(move_uci)

        if board.is_game_over() or state['status'] != 'started':
            return

        is_turn = (board.turn == chess.WHITE and is_white) or \
                  (board.turn == chess.BLACK and not is_white)

        if is_turn:
            print(f"\n\n[{game_id}] Thinking... (Move {board.fullmove_number})")
            if self.book and board.fullmove_number <= 10:
                try:
                    entry = self.book.weighted_choice(board)
                    self.make_move_with_retry(game_id, entry.move.uci())
                    return 
                except IndexError:
                    pass

            env.reset_with_board(board)
            best_action = mcts_instance.run(env)
            if best_action is not None:
                uci_move = env.action_to_uci(best_action)
                print(f"[{game_id}] AI Move: {uci_move}")
                self.make_move_with_retry(game_id, uci_move)
            else:
                self.client.bots.resign_game(game_id)

def console_input_handler(bot_instance):
    print("\ncommands:")
    print("  challenge <user> <min> <inc>")
    print("  cancel <user> (or 'all')")
    print("  resign <user> (or 'all')")
    print("  stop (force quit bot)\n")
    
    while True:
        try:
            user_input = input()
            if user_input.strip() == "": continue
            parts = user_input.split()
            cmd = parts[0].lower()
            
            if cmd == "challenge":
                if len(parts) < 2:
                    print("Usage: challenge <username> [min] [inc]")
                    continue
                target = parts[1]
                mins = int(parts[2]) if len(parts) > 2 else 10
                inc = int(parts[3]) if len(parts) > 3 else 5
                bot_instance.send_challenge(target, mins, inc)

            elif cmd == "cancel":
                target = parts[1] if len(parts) > 1 else "all"
                bot_instance.cancel_challenge(target)
                
            elif cmd == "resign":
                if len(parts) < 2:
                    print("Usage: resign <username> or 'resign all'")
                    continue
                target = parts[1]
                bot_instance.resign_game(target)

            elif cmd == "stop" or cmd == "exit":
                print("Force stopping bot...")
                # This kills the process immediately, stopping the 
                # blocking Lichess listener thread.
                os._exit(0)

            else:
                print("Unknown command.")
                
        except ValueError:
            print("Error: Invalid numbers.")
        except Exception as e:
            print(f"Input Error: {e}")

if __name__ == "__main__":
    bot = LichessBot(API_TOKEN)
    input_thread = threading.Thread(target=console_input_handler, args=(bot,), daemon=True)
    input_thread.start()
    try:
        bot.run()
    except KeyboardInterrupt:
        print("Bot stopped.")
from pettingzoo.classic import chess_v6
import numpy as np
from pettingzoo.classic.chess import chess_utils
from pettingzoo.utils.agent_selector import agent_selector
import chess

def action_idx_to_move(board: chess.Board, action_idx: int) -> chess.Move:
    """
    Converts a PettingZoo action index to a python-chess Move object
    based on the current board state (turn).
    """
    # 0 = White, 1 = Black
    player_idx = 0 if board.turn == chess.WHITE else 1
    move = chess_utils.action_to_move(board, action_idx, player_idx)
    return move

def game_over(board: chess.Board):
    return (board.is_game_over() or 
                board.can_claim_threefold_repetition() or 
                board.can_claim_fifty_moves())

class chess_gym:
    def __init__(self, render_mode="rgb_array"):
        self.env = chess_v6.env(render_mode=render_mode)
        self.legal_mask = None
        self.obv = None

    def reset(self, seed=None, fen=None):
        self.env.reset(seed=seed)
        unwrapped = self.env.unwrapped
        
        if fen is not None:
            unwrapped.board.set_fen(fen)
            # --- FIX: Sync internal pointer logic ---
            unwrapped.agents = unwrapped.possible_agents[:]
            unwrapped._agent_selector = agent_selector(unwrapped.agents)
            
            if unwrapped.board.turn == chess.WHITE:
                unwrapped.agent_selection = unwrapped._agent_selector.next()
            else:
                unwrapped._agent_selector.next()
                unwrapped.agent_selection = unwrapped._agent_selector.next()

            # --- FIX: Don't blindly set terminations to False ---
            # Check if the FEN provided is already a Game Over state
            is_over = (unwrapped.board.is_game_over() or 
                       unwrapped.board.can_claim_threefold_repetition() or 
                       unwrapped.board.can_claim_fifty_moves())
            
            unwrapped.terminations = {a: is_over for a in unwrapped.agents}
            unwrapped.truncations = {a: False for a in unwrapped.agents}
            unwrapped.rewards = {a: 0 for a in unwrapped.agents}
            unwrapped._cumulative_rewards = {a: 0 for a in unwrapped.agents}
            unwrapped.infos = {a: {} for a in unwrapped.agents}

        self._update_state()
        _, _, _, _, info = self.env.last()
        return self.obv, info
    
    def reset_with_board(self, board: chess.Board, seed=None):
        self.env.reset(seed=seed)
        unwrapped = self.env.unwrapped
        
        unwrapped.board = board.copy()
        
        unwrapped.agents = unwrapped.possible_agents[:]
        unwrapped._agent_selector = agent_selector(unwrapped.agents)
        
        if unwrapped.board.turn == chess.WHITE:
            unwrapped.agent_selection = unwrapped._agent_selector.next()
        else:
            unwrapped._agent_selector.next()
            unwrapped.agent_selection = unwrapped._agent_selector.next()
            
        is_over = (unwrapped.board.is_game_over() or 
                   unwrapped.board.can_claim_threefold_repetition() or 
                   unwrapped.board.can_claim_fifty_moves())

        unwrapped.terminations = {a: is_over for a in unwrapped.agents}
        unwrapped.truncations = {a: False for a in unwrapped.agents}
        unwrapped.rewards = {a: 0 for a in unwrapped.agents}
        unwrapped._cumulative_rewards = {a: 0 for a in unwrapped.agents}
        unwrapped.infos = {a: {} for a in unwrapped.agents}

        self._update_state()
        
        _, _, _, _, info = self.env.last()
        return self.obv, info
    
    def reset_with_moves(self, board: chess.Board):
        replay_board = board.copy()
        hist_depth = 7
        moves = []
        
        for i in range(hist_depth):
            if not replay_board.move_stack:
                break
            moves.append(replay_board.pop())
        
        self.reset_with_board(replay_board)
        
        for i in range(len(moves)-1, -1, -1):
            self.env.step(self.uci_to_action(str(moves[i])))
        self._update_state()
        
        return self.obv

    def _update_state(self):
        current_agent = self.env.unwrapped.agent_selection
        
        # look at me now
        observation = self.env.unwrapped.observe(current_agent)
        
        self.obv = self._reshape_obs(observation["observation"])
        self.legal_mask = observation["action_mask"]
        
        return self.obv
    
    def step(self, action):
        self.env.step(action)
        _, reward, termination, truncation, info = self.env.last()
        reward *= -1         
        
        self._update_state()
        
        return self.obv, reward, termination, truncation, info

    def get_legal_mask(self):
        return self.legal_mask
    def get_illegal_mask(self):
        return 1 - self.legal_mask
    
    def get_random_legal_move(self):        
        # Get the current agent whose turn it is
        current_agent = self.env.agent_selection

        # Use the action space for the current agent to sample a random action
        # The mask ensures that only legal moves are selected
        action = self.env.action_space(current_agent).sample(self.legal_mask)
        return action
    
    def get_random_illegal_move(self):        
        # Get the current agent whose turn it is
        current_agent = self.env.agent_selection

        # Use the action space for the current agent to sample a random action
        # The mask ensures that only legal moves are selected
        action = self.env.action_space(current_agent).sample(1 - self.legal_mask)
        return action
    
    def close(self):
        self.env.close()
    
    def _reshape_obs(self, obs):
        return np.transpose(obs, (2, 0, 1))
    
    def get_fen(self):
        return self.env.unwrapped.board.fen()

    def get_50move_clock(self):
        return self.env.unwrapped.board.halfmove_clock

    def get_num_moves(self):
        return len(self.env.unwrapped.board.move_stack)
    
    def get_board(self):
        return self.env.unwrapped.board
    
    def is_game_over(self):
        # 1. Access unwrapped to avoid 'reset() needs to be called' error
        # 2. Access .board directly because internal PettingZoo dicts (terminations)
        #    are not updated when we manually inject the board in MCTS.
        board = self.env.unwrapped.board
        
        return (board.is_game_over() or 
                board.can_claim_threefold_repetition() or 
                board.can_claim_fifty_moves())

    def get_turn(self):
        if self.env.unwrapped.board.turn == chess.WHITE:
            return 1
        else:
            return -1
            
    def action_to_uci(self, action: int):
        """
        Converts a PettingZoo chess action ID (0-4671) back to a UCI string (e.g., 'e2e4').
        Handles the board orientation and inversion logic.
        """
        # 1. Get the current board state
        board = self.env.unwrapped.board # Directly access the internal board object
        
        # 2. Determine the player index (0 for White, 1 for Black)
        # This is crucial because action_to_move uses it.
        current_index = self.env.agents.index(self.env.agent_selection)

        # 3. Use PettingZoo's internal function to get the python-chess Move object
        # This function internally handles the action -> plane -> source square mapping,
        # and also applies the player_index to correctly orient the move.
        chosen_move = chess_utils.action_to_move(board, action, current_index)

        # 4. Convert the chess.Move object to UCI string
        return chosen_move.uci()

    def uci_to_action(self, uci_move: str):
        """
        Converts a UCI string (e.g., 'e2e4', 'e8h8', 'a7a8q') to a PettingZoo action ID.
        Handles Lichess-style castling and all promotion types.
        """
        board = self.env.unwrapped.board
        
        # 1. Determine the player index (0 for White, 1 for Black)
        # This matches the logic in action_to_uci
        current_index = self.env.agents.index(self.env.agent_selection)

        # 2. Pre-process Lichess-style castling (King-takes-Rook)
        # PettingZoo/python-chess expects King-moves-two-squares (e.g., e1g1)
        if uci_move == "e1h1" and board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE):
            uci_move = "e1g1"
        elif uci_move == "e1a1" and board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE):
            uci_move = "e1c1"
        elif uci_move == "e8h8" and board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK):
            uci_move = "e8g8"
        elif uci_move == "e8a8" and board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK):
            uci_move = "e8c8"

        # 3. Create the Move object (automatically handles promotion suffix like 'q')
        move = chess.Move.from_uci(uci_move.lower())

        # 4. Handle Orientation (Relative Perspective)
        # If it's Black's turn (player_1), PettingZoo mirrors the board.
        # We must mirror the move coordinates so they match the "White-oriented" action space.
        if current_index == 1:
            # chess.square_mirror flips the rank (Rank 1 <-> Rank 8)
            move.from_square = chess.square_mirror(move.from_square)
            move.to_square = chess.square_mirror(move.to_square)

        # 5. Use chess_utils to get the plane (0-72)
        # This handles Knight moves, Queen moves, and under-promotions automatically.
        plane = chess_utils.get_move_plane(move)

        # 6. Calculate the Action Source Square (Column-Major format)
        # PettingZoo source square index = column * 8 + row
        from_square = move.from_square
        column = chess.square_file(from_square)
        row = chess.square_rank(from_square)
        action_source = column * 8 + row

        # 7. Combine for final action ID
        return action_source * 73 + plane
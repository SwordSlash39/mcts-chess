# mcts-chess
MCTS chess model trained off value action pairs of stockfish

Data trained from `https://huggingface.co/datasets/Lichess/chess-position-evaluations`
Model plays at ~2500 elo (with 1000 node search) after 50k steps (~45 hours on a rtx 3060)

Note MCTS search is coded very poorly here, please improve it if you want to put it to production

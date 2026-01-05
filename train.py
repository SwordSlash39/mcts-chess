import torch
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from datetime import datetime
from datasets import load_dataset
from gen_dataset import ChessStreamDataset

from old_model import TinyChessTransformer, hl_gauss_loss

def train():
    from torch.utils.tensorboard import SummaryWriter
    
    # UPDATE STEPS & EPOCH PER UPDATE, THATS IT.
    BATCH_SIZE = 1024
    BATCH_PER_UPDATE = 4
    STEPS = 57577
    GRAD_UPDATES = 22319
    QUERY_BATCH_SIZE = 8700

    folder_name = f"runs/tinytransformer{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(folder_name)

    dataset = load_dataset("Lichess/chess-position-evaluations", streaming=True, split="train").shuffle(buffer_size=1_000_000, seed=42).skip(STEPS * BATCH_SIZE)
    stream_ds = ChessStreamDataset(dataset, source_chunk_size=QUERY_BATCH_SIZE, target_batch_size=BATCH_SIZE)
    loader = DataLoader(
        stream_ds, 
        batch_size=None, 
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True
    )

    print("Data loaded!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyChessTransformer().to(device)
    model.load_checkpoint("TinyChessTransformer.pt")
    # im interested!
    # dummy_input = torch.zeros((1, 111, 8, 8), dtype=torch.float32, device=device)
    # writer.add_graph(model, dummy_input)
    model = torch.compile(
        model,
        options={
            "triton.cudagraphs": False,
            "max_autotune": False,
            "epilogue_fusion": True,
            "shape_padding": True,
        }
    )

    criterion = torch.nn.CrossEntropyLoss()

    decay_sched = lr_scheduler.CosineAnnealingLR(
        model.optim, T_max=300_000, eta_min=5e-7, last_epoch=GRAD_UPDATES-1
    )

    warmup_steps = 1024
    warmup_sched = lr_scheduler.LinearLR(
        model.optim, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps, last_epoch=GRAD_UPDATES-1
    )

    scheduler = lr_scheduler.SequentialLR(
        model.optim, 
        schedulers=[warmup_sched, decay_sched], 
        milestones=[warmup_steps],
    )
    model.optim.zero_grad()

    k = STEPS + 1
    try:        
        for x, value_label, policy_label in tqdm(loader):            
            x = x.to(device, non_blocking=True)
            value_label = value_label.to(device, non_blocking=True)
            policy_label = policy_label.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                policy, value = model(x, training=True)
                
                policy_loss = criterion(policy, policy_label)
                value_loss = hl_gauss_loss(value, value_label)
                
                writer.add_scalar('Policy Loss', policy_loss.item(), k)
                writer.add_scalar('Value Loss', value_loss.item(), k)
                
                loss = (policy_loss + value_loss) / BATCH_PER_UPDATE
                
                loss.backward()
            
            if k % BATCH_PER_UPDATE == 0:   
                model.optim.step()
                model.optim.zero_grad()
                scheduler.step() 
            
                GRAD_UPDATES += 1
            if k % 250 == 0:
                model.save_checkpoint("TinyChessTransformer.pt")
            
            k += 1
            
        model.save_checkpoint("TinyChessTransformer.pt")   
                
    except KeyboardInterrupt:
        pass

    model.save_checkpoint("TinyChessTransformer.pt")    
    print("Saved model!")
    print(k-1)
    print(f"K: {k-1},  Grad updates: {GRAD_UPDATES}")

if __name__ == '__main__':
    train()
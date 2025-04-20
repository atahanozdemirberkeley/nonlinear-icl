import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from model import InContextModel
from tasks import get_task

def train_model(config):
    # Initialize wandb if enabled
    if config.use_wandb:
        wandb.init(project=config.project_name, config=config.__dict__)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = InContextModel(
        n_dims=config.n_dims,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    for step in tqdm(range(config.train_steps)):
        # Sample input data
        xs = torch.randn(config.batch_size, config.n_points, config.n_dims).to(device)
        
        # Get task and evaluate
        task = get_task(
            config.task_name,
            config.n_dims,
            config.batch_size,
            scale=config.task_scale,
            hidden_size=config.task_hidden_size
        )
        ys = task.evaluate(xs)
        
        # Forward pass
        predictions = model(xs, ys)
        
        # Compute loss
        loss = nn.MSELoss()(predictions, ys.unsqueeze(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log metrics
        if config.use_wandb and step % config.log_every == 0:
            wandb.log({
                "loss": loss.item(),
                "step": step
            })
    
    return model

if __name__ == "__main__":
    # Example configuration
    class Config:
        # Model parameters
        n_dims = 10
        n_positions = 100
        n_embd = 128
        n_layer = 4
        n_head = 4
        
        # Training parameters
        batch_size = 32
        learning_rate = 1e-4
        train_steps = 10000
        
        # Task parameters
        task_name = "linear"  # or "quadratic" or "relu"
        task_scale = 1.0
        task_hidden_size = 100
        
        # Logging parameters
        use_wandb = False
        project_name = "in-context-learning"
        log_every = 100
    
    config = Config()
    model = train_model(config) 
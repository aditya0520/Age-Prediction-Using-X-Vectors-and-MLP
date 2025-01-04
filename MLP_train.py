import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
import wandb
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp

wandb.login(key="7664f3b17a98ffe7c64b549e349123b61a9d3024") 

class XVectorDataset(Dataset):
    def __init__(self, excel_path, xvector_base_path, num_workers=4):
        """
        Args:
            excel_path (str): Path to the Excel file containing metadata.
            xvector_base_path (str): Base path where xvector .npy files are stored.
            num_workers (int): Number of threads to use for preloading.
        """
        self.data = pd.read_excel(excel_path)
        self.xvector_base_path = xvector_base_path
        self.samples = []

        # Filter valid indices and preload data
        self._filter_valid_indices()
        self._preload_data(num_workers)

    def _filter_valid_indices(self):
        """
        Filters out rows where the corresponding x_vector file does not exist
        using multithreading and progress tracking with tqdm.
        """
        def is_valid(idx):
            row = self.data.iloc[idx]
            pin = row['PIN']
            file_name = row['File Name'] + '.npy'
            x_vector_path = os.path.join(self.xvector_base_path, str(pin), file_name)
            return idx if os.path.isfile(x_vector_path) else None

        with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers as needed
            results = list(tqdm(
                executor.map(is_valid, range(len(self.data))),
                total=len(self.data),
                desc="Filtering Valid Indices"
            ))

        # Filter out None values from the results
        self.valid_indices = [idx for idx in results if idx is not None]

    def _load_sample(self, idx):
        """Loads a single sample."""
        row = self.data.iloc[idx]
        pin = row['PIN']
        file_name = row['File Name'] + '.npy'
        age = row['AGE']

        # Construct the X-Vector file path
        x_vector_path = os.path.join(self.xvector_base_path, str(pin), file_name)

        # Load X-Vector features
        x_vector = np.load(x_vector_path)

        # Convert to torch tensor
        x_vector = torch.tensor(x_vector, dtype=torch.float32)
        age = torch.tensor(age, dtype=torch.float32)

        return x_vector, age

    def _preload_data(self, num_workers):
        """Preloads the dataset into memory using threading with progress tracking."""
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            self.samples = list(tqdm(
                executor.map(self._load_sample, self.valid_indices),
                total=len(self.valid_indices),
                desc="Preloading Dataset"
            ))

    def __len__(self):
        """Returns the number of valid samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to fetch.
        Returns:
            x_vector (torch.Tensor): X-Vector features.
            age (torch.Tensor): Corresponding age label.
        """
        return self.samples[idx]
    

class Model(torch.nn.Module):
    def __init__(self, input_size=512):
        super(Model, self).__init__()

        self.model = torch.nn.Sequential(

            torch.nn.Linear(input_size, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.1),
            
            torch.nn.Linear(1024, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.1),

            # Output layer
            torch.nn.Linear(512, 1)  # Single output for scalar regression
        )

    def forward(self, x):
        return self.model(x)

# Train one epoch with Mixed-Precision
def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0

    for x_vector, age in dataloader:
        x_vector = x_vector.to(device)
        age = age.to(device)

        optimizer.zero_grad()

        # Mixed-Precision Forward Pass
        with autocast():
            outputs = model(x_vector).squeeze()
            loss = criterion(outputs, age)

        # Backward Pass and Optimization with Scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Validate one epoch
def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x_vector, age in dataloader:
            x_vector = x_vector.to(device)
            age = age.to(device)

            # Forward pass
            outputs = model(x_vector).squeeze()
            loss = criterion(outputs, age)

            total_loss += loss.item()

    return total_loss / len(dataloader)

def train_process(local_rank, num_epochs=100, batch_size=128):
    # Initialize process group
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.barrier()

    num_workers = 8
    train_path = "/ocean/projects/cis240138p/aaayush/fisher_segmented_audio/train.xlsx"
    train_xvector_base_path = "/ocean/projects/cis240138p/aaayush/fisher_x-vector/train"
    dev_path = "/ocean/projects/cis240138p/aaayush/fisher_segmented_audio/dev.xlsx"
    dev_xvector_base_path = "/ocean/projects/cis240138p/aaayush/fisher_x-vector/dev"
    test_path = "/ocean/projects/cis240138p/aaayush/fisher_segmented_audio/test.xlsx"
    test_xvector_base_path = "/ocean/projects/cis240138p/aaayush/fisher_x-vector/test"


    # Create datasets and samplers
    train_dataset = XVectorDataset(excel_path=train_path, xvector_base_path=train_xvector_base_path, num_workers=num_workers)
    val_dataset = XVectorDataset(excel_path=dev_path, xvector_base_path=dev_xvector_base_path, num_workers=num_workers)
    test_dataset = XVectorDataset(excel_path=test_path, xvector_base_path=test_xvector_base_path, num_workers=num_workers)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=0)

    # Initialize model, optimizer, and criterion
    model = Model(input_size=512).to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    criterion = torch.nn.L1Loss()

    # Mixed-Precision GradScaler
    scaler = GradScaler()

    # Initialize W&B only for rank 0
    if dist.get_rank() == 0:
        wandb.init(
            project="Age Prediction",
            config={
                "learning_rate": 1e-3,
                "batch_size": batch_size,
                "epochs": num_epochs,
                "input_size": 512,
            }
        )

    try:
        # Training loop
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)

            # Train and validate
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
            val_loss = validate_one_epoch(model, val_loader, criterion, device)

            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)

            # Log results only on rank 0
            if dist.get_rank() == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr})

        if dist.get_rank() == 0:
            test_loader = DataLoader(test_dataset, batch_size=128)
            test_loss = validate_one_epoch(model, test_loader, criterion, device)
            print(f"Test Loss: {test_loss:.4f}")
            wandb.log({"Test Loss": test_loss})

    finally:
        if dist.is_initialized() and dist.get_rank() == 0:
            wandb.finish()
        if dist.is_initialized():
            dist.destroy_process_group()
    

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])  
    train_process(local_rank)
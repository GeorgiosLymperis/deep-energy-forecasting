from nf_utils import NormalizingFlowUMNN
from utils_data import GEFcomSolarLoader
import torch

dataset = GEFcomSolarLoader()
dataset.build_features()
dataset.split()

train_dataloader, validation_dataloader, test_dataloader = dataset.get_dataloaders(batch_size=32)

x_dim = len(dataset.active_hours)
c_dim = len(dataset.features + dataset.zones) * len(dataset.active_hours)

flow = NormalizingFlowUMNN(x_dim, c_dim, hidden_features=[16, 16], signal=8)

optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3, weight_decay=1e-5)

epochs = 20

for epoch in range(1, epochs + 1):
    losses = []

    for x, label in train_dataloader:
        # x_batch = label  
        x_batch = torch.clamp(label, -10, 10)
        c_batch = x.view(x.size(0), -1)  # flatten context per day
 
        loss = -flow.log_prob(x_batch, c_batch).mean()
                

        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=0.5)
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.detach())

    losses = torch.stack(losses)
    print(f"Epoch {epoch}/{epochs}, NLL: {losses.mean():.4f}")
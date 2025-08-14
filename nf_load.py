from nf_utils import NormalizingFlowUNAF, NormalizingFlowNAF, NormalizingFlowNSF
from utils_data import GEFcomLoadLoader
import torch


dataset = GEFcomLoadLoader()
dataset.build_features()
dataset.split()

train_dataloader, validation_dataloader, test_dataloader = dataset.get_dataloaders(batch_size=32)

x_dim = 24
c_dim = 24 * len(dataset.features)

# flow = NormalizingFlowUNAF(x_dim, c_dim, hidden_features=[16, 16], signal=8)
# flow = NormalizingFlowNAF(x_dim, c_dim, hidden_features=[16, 16], signal=16, randperm=True)
flow = NormalizingFlowNSF(features_dim=x_dim, transforms=3, hidden_features=[16, 16])

epochs = 2

flow.fit(train_dataloader, lr=5e-4)
# for epoch in range(1, epochs + 1):
#     losses = []

#     for x, label in train_dataloader:
#         # x_batch = label  
#         x_batch = torch.clamp(label, -10, 10)
#         c_batch = x.view(x.size(0), -1)  # flatten context per day
 
#         loss = -flow.log_prob(x_batch, c_batch).mean()
                

#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=0.5)
#         optimizer.step()
#         optimizer.zero_grad()

#         losses.append(loss.detach())

#     losses = torch.stack(losses)
#     print(f"Epoch {epoch}/{epochs}, NLL: {losses.mean():.4f}")
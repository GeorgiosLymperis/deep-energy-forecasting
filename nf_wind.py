from nf_utils import NormalizingFlowUMNN
from utils_data import GEFcomWindLoader
import torch
from evaluation.metrics import crps_batch_per_marginal, energy_score_per_batch, variogram_score_per_batch, quantile_score_averaged_fast

dataset = GEFcomWindLoader()
dataset.build_features()
dataset.split()

train_dataloader, validation_dataloader, test_dataloader = dataset.get_dataloaders(batch_size=32)

flow = NormalizingFlowUMNN(x_dim=24, c_dim=480, hidden_features=[16, 16], signal=8)

optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3, weight_decay=1e-5)

epochs = 1

for epoch in range(1, epochs + 1):
    losses = []

    for x, label in train_dataloader:
        # x_batch = label  # shape [batch_size, 24]
        x_batch = torch.clamp(label, -10, 10)
        c_batch = x.view(x.size(0), -1)  # flatten context per day: [batch_size, 480]
 
        loss = -flow.log_prob(x_batch, c_batch).mean()
                

        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=0.5)
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.detach())

    losses = torch.stack(losses)
    print(f"Epoch {epoch}/{epochs}, NLL: {losses.mean():.4f}")

with torch.no_grad():
    for x, label in test_dataloader:
        c_batch = x.view(x.size(0), -1)
        x_batch = torch.clamp(label, -10, 10)
        # quantiles = flow.quantiles(c_batch, [0.01*i for i in range(1, 100)])

        samples = flow.sample(20, c_batch)
        # print(samples.shape)
        # print(x_batch.shape)
        
        samples_np = samples.detach().cpu().numpy()
        x_batch_np = x_batch.detach().cpu().numpy()

        print('crps')
        print(crps_batch_per_marginal(samples_np, x_batch_np))
        # print('energy score')
        # print(energy_score_per_batch(samples_np, x_batch_np))
        # print('variogram score')
        # print(variogram_score_per_batch(samples_np, x_batch_np))
        # quantile_score = quantile_score_averaged_fast(quantiles, x_batch_np)
        # print("quantile_score")
        # print(quantile_score)


        # break
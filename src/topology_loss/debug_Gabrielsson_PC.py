from topologylayer.nn import AlphaLayer, LevelSetLayer2D, BarcodePolyFeature, PartialSumBarcodeLengths, SumBarcodeLengths, TopKBarcodeLengths
import torch, numpy as np, matplotlib.pyplot as plt
from topology import TopologyLoss, TopologyLossBaseline
from torch.nn.functional import sigmoid

# random pointcloud
np.random.seed(0)
data = np.random.rand(100, 3)

size = (32, 32)
k = 40

baseline = False
mean = 0
means = [-0.5, 0, 0.5]

if baseline:    
    topo_loss = TopologyLossBaseline(k=k, size=size)
else:
    topo_loss = TopologyLoss(k=k, size=size)

results = []
loss_set = []

for mean in means:
    if mean == 0:
        data = np.random.rand(1, 1, 32,32) - 0.5
    if mean == 0.5:
        data = np.random.rand(1, 1, 32,32)
    if mean == -0.5:
        data = -1*np.random.rand(1, 1, 32,32)

    x = torch.autograd.Variable(torch.tensor(data).type(torch.float), requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=1e-2)

    losses = []
    for i in range(70):
        optimizer.zero_grad()
        # y = sigmoid(x)
        loss = topo_loss(sigmoid(x))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    z = x.detach().clone().squeeze()
    results.append(z)
    loss_set.append(losses)
    
    # p = 60

    # if baseline:
    #     topo_loss = TopologyLossBaseline(k=p, size=size)
    # else:
    #     topo_loss = TopologyLoss(k=p, size=size)

    # for i in range(200):
    #     optimizer.zero_grad()
    #     # y = sigmoid(x)
    #     loss = topo_loss(sigmoid(x))
    #     loss.backward()
    #     optimizer.step()
    
    # y = x.detach().squeeze()

plt.subplot(2,3,1)
plt.imshow(results[0] > 0)
plt.title(f'Noisy input mean {means[0]}, n_cc = {k}')
plt.subplot(2,3,2)
plt.imshow(results[1] > 0)
plt.title(f'Noisy input mean {means[1]}, n_cc = {k}')
plt.subplot(2,3,3)
plt.imshow(results[2] > 0)
plt.title(f'Noisy input mean {means[2]}, n_cc = {k}')
plt.subplot(2,3,4)
plt.plot(loss_set[0])
plt.title(f'Loss curve for input mean {means[0]}, n_cc = {k}')
plt.subplot(2,3,5)
plt.plot(loss_set[1])
plt.title(f'Loss curve for input mean {means[1]}, n_cc = {k}')
plt.subplot(2,3,6)
plt.plot(loss_set[2])
plt.title(f'Loss curve for input mean {means[2]}, n_cc = {k}')
plt.show()


# pattern retrieval is subset based, adding things without losing them.
# https://www.cs.sfu.ca/~keval/
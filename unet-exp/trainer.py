'''
# The following is a basic training procedure to train the network
# You need to update the code to get the best performance
# TODO: approx ? lines
'''

import matplotlib.pyplot as plt

# Set the hyperparameters
num_epochs = 20
batch_size = 16
learning_rate = 0.003
weight_decay = 1e-5

model = ResNetUNet(1)
model = model.cuda() # move the model to GPU
loader, _ = get_plane_dataset('train', batch_size) # initialize data_loader
#crit = nn.BCEWithLogitsLoss() # Define the loss function
# crit = nn.BCEWithLogitsLoss()
#optim = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # Initialize the optimizer as SGD
optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# start the training procedure
for epoch in range(num_epochs):
    total_loss = 0
    for (img, mask) in tqdm(loader):
        img = torch.tensor(img, device=torch.device('cuda'), requires_grad = True)
        mask = torch.tensor(mask, device=torch.device('cuda'), requires_grad = True)
        pred = model(img)
        # loss = crit(pred, mask) + soft_dice_loss(mask, pred)
        loss = calc_loss(pred, mask)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.cpu().data
    print("Epoch: {}, Loss: {}".format(epoch, total_loss/len(loader)))
    torch.save(model.state_dict(), '{}/output/{}_segmentation_model.pth'.format(BASE_DIR, epoch))

'''
# Saving the final model
'''
torch.save(model.state_dict(), '{}/output/final_segmentation_model.pth'.format(BASE_DIR))

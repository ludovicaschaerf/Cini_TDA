import torch
from torch import nn
import time
import copy
from torch.optim import lr_scheduler

def train_replica(model, loaders, dataset_sizes, num_epochs=20):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    
    triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=torch.cdist, margin=1.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_loss = 100000000

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']: #, 'val'
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            
            # Iterate over data.
            for a, b, c in loaders[phase]:
                a = a.squeeze(1).to(device)
                b = b.squeeze(1).to(device)
                c = c.squeeze(1).to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass
                    A, B, C = model(a, b ,c)
                    # Compute and print loss
                    loss = triplet_loss(A, B, C)
    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * a.size(0)
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss > best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model
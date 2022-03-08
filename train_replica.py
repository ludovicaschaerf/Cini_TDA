import torch
from torch import nn
import time
import copy
from torch.optim import lr_scheduler
from tqdm import tqdm
import pickle
from scipy import sparse

def train_replica(model, loaders, dataset_sizes, dts, device='cpu', data_dir='/scratch/students/schaerf/', num_epochs=20, batch_size=8):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    triplet_loss = nn.TripletMarginLoss(
        margin=0.3, reduction='sum' # to be optimized margin
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6) # to be optimized lr and method
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # to be optimized step and gamma

    best_loss = 1000
    train_accuracy = 0
    test_accuracy = 0
    accuracies = []

    with open(data_dir + "dict2emb.pkl", "rb") as infile:
            dict2emb = pickle.load(infile)

    for epoch in range(num_epochs):
               
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "test"]:  # , 'val'
            
            running_loss = 0.0

            # Iterate over data.
            for i, [a, b, c] in tqdm(enumerate(loaders[phase])):
                a = a.squeeze(1).to(device)
                b = b.squeeze(1).to(device)
                c = c.squeeze(1).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # Forward pass
                    A, B, C = model(a, b, c)
                    # Compute and print loss
                    loss = triplet_loss(A, B, C)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * a.size(0)
                
                if epoch % 2 == 0:
                    for j in range(batch_size):
                        if i*batch_size + j < dataset_sizes[phase]:
                            uid, set_b, set_c = dts[phase].__get_metadata__(i*batch_size + j)
                            dict2emb[uid] = sparse.csr_matrix(A[j].cpu().detach().numpy().T)

                        if phase == "train":
                            train_accuracy += model.evaluate(set_b, set_c)
                        else:
                            test_accuracy += model.evaluate(set_b, set_c)

                #break 

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            if epoch % 2 == 0:
                if phase == "train":
                    epoch_train_acc = train_accuracy / dataset_sizes[phase]
                    print("{} Loss: {:.4f} Accuracy @ 4 {:.4f}".format(phase, epoch_loss, epoch_train_acc))

                else:
                    epoch_test_acc = train_accuracy / dataset_sizes[phase]
                    accuracies.append(epoch_test_acc)
                    print("{} Loss: {:.4f} Accuracy @ 4 {:.4f}".format(phase, epoch_loss, epoch_test_acc))

                with open(data_dir + "dict2emb.pkl", "wb") as outfile: ## not updating images in subset
                    pickle.dump(dict2emb, outfile)

            else:
                print("{} Loss: {:.4f}".format(phase, epoch_loss))
            

            # deep copy the model
            if (
                phase == "test" and epoch_loss < best_loss
            ):  # needs to be changed to val
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val loss: {:4f}".format(best_loss))
    print("Best val acc: {:4f}".format(max(accuracies)))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model

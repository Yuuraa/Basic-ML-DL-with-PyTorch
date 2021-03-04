import torch
import numpy as np
import os

# Saving & Loading Model
def save(checkpoint_dir, model_name, model, optimizer, epoch, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join(checkpoint_dir, 'name'+f'{epoch:05d}'))
    
    print(f"Saved model: {name}{epoch:05d}")


# Visualize data
def show_tensor_to_img(input, label, predicted):
    input = input.cpu().squeeze()
    label = label.cpu().squeeze()
    predicted = predicted.detach().cpu().squeeze()

    plt.subplot(131)
    plt.imshow(input)

    plt.title('input')

    plt.subplot(132)
    plt.imshow(label)
    plt.title('label')

    plt.subplot(133)
    plt.imshow(predicted)
    plt.title('predicted')

    plt.show()


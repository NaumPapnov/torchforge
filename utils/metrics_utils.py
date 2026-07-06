from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch import max

# utils to save metrics, generate plots, etc.

def training_plot(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(12, 9))

    plt.subplot(2, 1, 1)
    plt.xlabel('epoch')
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='valid loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel('epoch')
    plt.plot(val_accuracies, label='valid accuracy')
    plt.legend()

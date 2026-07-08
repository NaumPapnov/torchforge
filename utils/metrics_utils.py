import matplotlib.pyplot as plt

# utils to save metrics, generate plots, etc.

def training_plot(train_losses, val_losses, val_metrics_history=None):
    plt.figure(figsize=(12, 9))

    plt.subplot(2, 1, 1)
    plt.xlabel('epoch')
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='valid loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel('epoch')
    if val_metrics_history:
        for metric_name, metric_values in val_metrics_history.items():
            plt.plot(metric_values, label=f'valid {metric_name}')
        plt.legend()

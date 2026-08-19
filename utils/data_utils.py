import matplotlib.pyplot as plt
from matplotlib import patches


def show_images(
        dataset,
        pred_labels,
        pred_bboxes,
        label_titles,
        indices,
        subplots_size = (3, 5),
    ):
    _, axes = plt.subplots(*subplots_size, figsize=(30, 15))
    for ax, idx in zip(axes.flat, indices):
        image, label, bbox = dataset[idx]
        _, image_height, image_width = image.shape
        image = image.permute(1, 2, 0).cpu()
        ax.imshow(image)
        x_min, y_min, x_max, y_max = bbox
        x_min *= image_width
        y_min *= image_height
        x_max *= image_width
        y_max *= image_height
        width = x_max - x_min
        height = y_max - y_min
        true_bbox = patches.Rectangle((x_min, y_min), width, height, fill=False, color='red')
        x_min, y_min, x_max, y_max = pred_bboxes[idx]
        x_min *= image_width
        y_min *= image_height
        x_max *= image_width
        y_max *= image_height
        width = x_max - x_min
        height = y_max - y_min
        pred_bbox = patches.Rectangle((x_min, y_min), width, height, fill=False, color='blue')
        ax.add_patch(true_bbox)
        ax.add_patch(pred_bbox)
        pred_label = label_titles[pred_labels[idx].argmax().item()]
        label = label_titles[label.item()]
        ax.set_title(f'True label: {label} | Pred label: {pred_label}')
        ax.axis("off")
    plt.tight_layout()
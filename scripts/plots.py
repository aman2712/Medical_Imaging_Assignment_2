import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import label

REFERENCE_COLOUR = "#56B4E9"
PREDICTION_COLOUR = "#E69F00"
ERROR_COLOURS = {1: "#009E73", 2: "#E69F00", 3: "#CC79A7"}  # true positive, false positive, false negative
ERROR_NAMES = {1: "True positive", 2: "False positive", 3: "False negative"}


def three_planes(volume, point):
    x, y, z = point
    return {"Axial": volume[:, :, z].T, "Coronal": volume[:, y, :].T, "Sagittal": volume[x, :, :].T}


def error_map(prediction, reference):
    prediction, reference = prediction.astype(bool), reference.astype(bool)
    return (prediction & reference) * 1 + (prediction & ~reference) * 2 + (~prediction & reference) * 3


def centroid_point(mask):
    coords = np.argwhere(mask)
    centroid = coords.mean(axis=0)
    return tuple(coords[np.argmin(((coords - centroid) ** 2).sum(axis=1))])


def largest_error_point(prediction, reference):
    errors = error_map(prediction, reference)
    best_mask, best_size = None, 0
    for value in (2, 3):
        components, n_components = label(errors == value, structure=np.ones((3, 3, 3)))
        if n_components == 0:
            continue
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        if sizes.max() > best_size:
            best_size, best_mask = sizes.max(), components == sizes.argmax()
    return centroid_point(best_mask)


def plot_contour_rows(cases):
    fig, axes = plt.subplots(len(cases), 3, figsize=(10, 3.4 * len(cases)))
    for row, (title, flair, reference, prediction, point) in zip(axes, cases):
        backgrounds, references, predictions = (three_planes(v, point) for v in (flair, reference, prediction))
        for ax, name in zip(row, backgrounds):
            ax.imshow(backgrounds[name], cmap="gray", origin="lower")
            if references[name].any():
                ax.contour(references[name], levels=[0.5], colors=REFERENCE_COLOUR, linewidths=1.2)
            if predictions[name].any():
                ax.contour(predictions[name], levels=[0.5], colors=PREDICTION_COLOUR, linewidths=1.2)
            ax.set_title(f"{name}", fontsize=9)
            ax.axis("off")
        row[0].text(-0.05, 0.5, title, transform=row[0].transAxes, rotation=90, va="center", ha="right", fontsize=9)
    fig.legend(handles=[Patch(edgecolor=REFERENCE_COLOUR, facecolor="none", label="Reference"),
                        Patch(edgecolor=PREDICTION_COLOUR, facecolor="none", label="Prediction")],
               loc="lower center", ncol=2, frameon=False)
    plt.tight_layout(rect=(0.03, 0.03, 1, 1))
    plt.show()


def plot_error_rows(cases):
    cmap = ListedColormap([ERROR_COLOURS[k] for k in (1, 2, 3)])
    fig, axes = plt.subplots(len(cases), 3, figsize=(10, 3.4 * len(cases)))
    for row, (title, flair, reference, prediction, point) in zip(axes, cases):
        backgrounds = three_planes(flair, point)
        errors = three_planes(error_map(prediction, reference), point)
        for ax, name in zip(row, backgrounds):
            ax.imshow(backgrounds[name], cmap="gray", origin="lower")
            ax.imshow(np.ma.masked_equal(errors[name], 0), cmap=cmap, vmin=0.5, vmax=3.5, alpha=0.75, origin="lower", interpolation="nearest")
            ax.set_title(f"{name}", fontsize=9)
            ax.axis("off")
        row[0].text(-0.05, 0.5, title, transform=row[0].transAxes, rotation=90, va="center", ha="right", fontsize=9)
    fig.legend(handles=[Patch(facecolor=ERROR_COLOURS[k], label=ERROR_NAMES[k]) for k in (1, 2, 3)],
               loc="lower center", ncol=3, frameon=False)
    plt.tight_layout(rect=(0.03, 0.03, 1, 1))
    plt.show()

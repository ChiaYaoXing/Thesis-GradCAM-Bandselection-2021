import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as f
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import h5py



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file = (h5py.File(
    "/media/Datasets/CHRISS/Hypercubes/Hyperspectral20210830_234812 ("
    "ErdeSteinePflanzenreste)/Hyperspectral20210830_234812_Reflectance.mat"))


datacube = np.asarray(file['reflectance'])
datacube = datacube.reshape(2402, -1)
datacube = datacube.transpose()

# Normalise the data between zero and one.
datacube = datacube - np.amin(datacube)

if np.amax(datacube) != 0:
    datacube /= np.amax(datacube)

wavelengths = np.asarray(file['wavelengths'])
wavelengths = wavelengths.reshape(1, 2402)[0]

labels = img.imread(
    "/media/Datasets/CHRISS/Hypercubes/Hyperspectral20210830_234812 (ErdeSteinePflanzenreste)/labels.png")
labels = np.asarray(labels * 255).astype(int)
labels = labels.transpose()
labels = labels.reshape(-1, 1)



# Remove labels that have label 0
indices = labels > 0
indices2D = np.tile(indices, 2402)
data = datacube[indices.squeeze(), :]
labels = labels[indices]
labels = labels - 1


# To plot the mean of each label along with its standard deviation.

n = 42000
data_dict = {0: np.zeros([n, 2402]), 1: np.zeros([n, 2402]), 2: np.zeros([n, 2402]), 3: np.zeros([n, 2402]), 4: np.zeros([n, 2402]), 5: np.zeros([n, 2402]), 6: np.zeros([n, 2402])}
count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
std = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
label_dict = {0: "soil", 1: "stone_1", 2: "stone_2", 3: "ear", 4: "branch", 5: "catkin", 6: "leaf"}



for bands, label in zip(data, labels):
    data_dict[label][count[label]] = bands
    count[label] += 1

for key, val in data_dict.items():
    std[key] = np.std(val, axis=0)
    data_dict[key] = sum(val) / count[key]

fig, axs = plt.subplots(2, 1)

for key in range(2):
    val = data_dict[key]
    axs[key].plot(wavelengths, val, color=(0.043, 0.611, 0.192, 1))
    axs[key].fill_between(wavelengths, val-std[key], val+std[key], color=(0.043, 0.611, 0.192, 0.1))
    axs[key].set_title(label_dict[key])
    axs[key].set_ylim([0.0, 0.8])


fig.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 1)

for key in range(2):
    val = data_dict[key + 2]
    axs[key].plot(wavelengths, val, color=(0.043, 0.611, 0.192, 1))
    axs[key].fill_between(wavelengths, val-std[key+2], val+std[key+2], color=(0.043, 0.611, 0.192, 0.1))
    axs[key].set_title(label_dict[key+2])
    axs[key].set_ylim([0.0, 0.8])

fig.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 1)

for key in range(2):
    val = data_dict[key + 4]
    axs[key].plot(wavelengths, val, color=(0.043, 0.611, 0.192, 1))
    axs[key].fill_between(wavelengths, val-std[key+4], val+std[key+4], color=(0.043, 0.611, 0.192, 0.1))
    axs[key].set_title(label_dict[key+4])
    axs[key].set_ylim([0.0, 0.8])

fig.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 1)

for key in range(1):
    val = data_dict[key+6]
    axs[key].plot(wavelengths, val, color=(0.043, 0.611, 0.192, 1))
    axs[key].fill_between(wavelengths, val-std[key+6], val+std[key+6], color=(0.043, 0.611, 0.192, 0.1))
    axs[key].set_title(label_dict[key+6])
    axs[key].set_ylim([0.0, 0.8])

fig.tight_layout()
plt.show()

# Load the dictionary of heatmaps generated from the varianceTest.py

bwa_reduced_heatmaps = np.load('bwareducedhms.npy', allow_pickle=True).item()
gbs_reduced_heatmaps = np.load('reducedhms.npy', allow_pickle=True).item()
heatmaps = np.load("hms.npy", allow_pickle=True).item()
n = 10

# Total average heatmaps for each classifier instances

bwa_reduced_heatmaps = sum(list(bwa_reduced_heatmaps.values())) / len(list(bwa_reduced_heatmaps.keys()))
gbs_reduced_heatmaps = sum(list(gbs_reduced_heatmaps.values())) / len(list(gbs_reduced_heatmaps.keys()))
heatmaps = sum(list(heatmaps.values())) / len(list(heatmaps.keys()))

# Smoothen the heatmap with rolling average with a window size of 10

box_pts = 10
box = np.ones(box_pts)/box_pts

# First section heatmap plot

for hm in gbs_reduced_heatmaps:
    hmhat = np.convolve(hm, box, mode="same")
    plt.plot(wavelengths[0:2196], hm[0:2196])

plt.title("Averaged GBS transformed Guided GradHM of " + str(n) + " classifier instances \n (380.82 nm - 944.65 nm)")
plt.xlabel("Wavelength (nm)")
plt.ylim([0.0, 0.4])
plt.show()

for hm in bwa_reduced_heatmaps:
    hmhat = np.convolve(hm, box, mode="same")
    plt.plot(wavelengths[0:2196], hm[0:2196])

plt.title("Averaged BWA transformed Guided GradHM of " + str(n) + " classifier instances \n (380.82 nm - 944.65 nm)")
plt.xlabel("Wavelength (nm)")
plt.ylim([0.0, 0.4])
plt.show()

for hm in heatmaps:
    hmhat = np.convolve(hm, box, mode="same")
    plt.plot(wavelengths[0:2196], hm[0:2196])

plt.title("Averaged Guided GradHM of " + str(n) + " classifier instances \n (380.82 nm - 944.65 nm)")
plt.xlabel("Wavelength (nm)")
plt.ylim([0.0, 0.4])
plt.show()


# First section Variance plot

plt.plot(wavelengths[0:2196], np.var(heatmaps, axis=0)[0:2196], label="Trained with whole spectral")
plt.plot(wavelengths[0:2196], np.var(gbs_reduced_heatmaps, axis=0)[0:2196], label="Trained with GBS feature reduced bands")
plt.plot(wavelengths[0:2196], np.var(bwa_reduced_heatmaps, axis=0)[0:2196], label="Trained with BWA feature reduced bands")
plt.title("Variance of the averaged Guided GradHM \n (380.82 nm - 944.65 nm)")
plt.xlabel("Wavelength (nm)")
plt.legend(loc='best')
plt.show()

# second section Heatmaps plot

for hm in bwa_reduced_heatmaps:
    hmhat = np.convolve(hm, box, mode="same")
    plt.plot(wavelengths[2196:2401], hm[2196:2401])

plt.title("Averaged BWA transformed Guided GradHM of " + str(n) + " classifier instances \n (944.65 nm - 1610.78 nm)")
plt.xlabel("Wavelength (nm)")
plt.ylim([0.0, 0.5])
plt.show()

for hm in gbs_reduced_heatmaps:
    hmhat = np.convolve(hm, box, mode="same")
    plt.plot(wavelengths[2196:2401], hm[2196:2401])

plt.title("Averaged GBS transformed Guided GradHM of " + str(n) + " classifier instances \n (944.65 nm - 1610.78 nm)")
plt.xlabel("Wavelength (nm)")
plt.ylim([0.0, 0.5])
plt.show()

for hm in heatmaps:
    hmhat = np.convolve(hm, box, mode="same")
    plt.plot(wavelengths[2196:2401], hm[2196:2401])

plt.title("Averaged Guided GradHM of " + str(n) + " classifier instances \n (944.65 nm - 1610.78 nm)")
plt.xlabel("Wavelength (nm)")
plt.ylim([0.0, 0.5])
plt.show()

# Second section Variance plot

plt.plot(wavelengths[2196:2401], np.var(heatmaps, axis=0)[2196:2401], label="Trained with whole spectral")
plt.plot(wavelengths[2196:2401], np.var(gbs_reduced_heatmaps, axis=0)[2196:2401], label="Trained with GBS feature reduced bands")
plt.plot(wavelengths[2196:2401], np.var(bwa_reduced_heatmaps, axis=0)[2196:2401], label="Trained with BWA feature reduced bands")
plt.title("Variance of the averaged Guided GradHM \n (944.65 nm - 1610.78 nm)")
plt.xlabel("Wavelength (nm)")
plt.legend(loc='best')
plt.show()

# Sum all the variance and compare the values

print(f"{sum(np.var(heatmaps, axis=0)) = }")
print(f"{sum(np.var(gbs_reduced_heatmaps, axis=0)) = }")
print(f"{sum(np.var(bwa_reduced_heatmaps, axis=0)) = }")
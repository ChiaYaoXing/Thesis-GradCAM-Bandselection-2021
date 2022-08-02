import collections
import sys

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


import Band_Wise_Averaging
import Gaussian_Based_Smoothing
import gradcam
import guided_gradcam
from Data_Cube import DataCube
from Guided_BackProp import GuidedBackProp
from gradcam import GradCam as gc
from Network import CNN

print("### Process Running at snyachia@134.91.77.152:4222 ###")

# reduced_heatmaps = np.load('reducedhms.npy', allow_pickle=True).item()
# heatmaps = np.load("hms.npy", allow_pickle=True).item()
# n = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file = (h5py.File(
    "/media/Datasets/CHRISS/Hypercubes/Hyperspectral20210830_234812 ("
    "ErdeSteinePflanzenreste)/Hyperspectral20210830_234812_Reflectance.mat"))


datacube = np.asarray(file['reflectance'])
datacube = datacube.reshape(2402, -1)
datacube = datacube.transpose()

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

print(data.shape)

label_dict = {0: "soil", 1: "stone_1", 2: "stone_2", 3: "ear", 4: "branch", 5: "catkin", 6: "leaf"}

# Hyperparameter
num_epochs = 25
batch_size = 64
learning_rate = 0.002

bwa = Band_Wise_Averaging.BandwiseAveraging(800)
gbs = Gaussian_Based_Smoothing.GaussianBasedSmoothing(800)

# 80% Train, 20% Test
xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.2)

# Prepare the Dataset

bwa_reduced_train_data = DataCube(xtrain, ytrain, bwa)
bwa_reduced_test_data = DataCube(xtest, ytest, bwa)
gbs_reduced_train_data = DataCube(xtrain, ytrain, gbs)
gbs_reduced_test_data = DataCube(xtest, ytest, gbs)
train_data = DataCube(xtrain, ytrain, None)
test_data = DataCube(xtest, ytest, None)

train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
grad_test_loader = DataLoader(test_data, batch_size=1)

gbs_reduced_train_loader = DataLoader(gbs_reduced_train_data, batch_size=batch_size)
gbs_reduced_test_loader = DataLoader(gbs_reduced_test_data, batch_size=batch_size)
gbs_reduced_grad_test_loader = DataLoader(gbs_reduced_test_data, batch_size=1)

bwa_reduced_train_loader = DataLoader(bwa_reduced_train_data, batch_size=batch_size)
bwa_reduced_test_loader = DataLoader(bwa_reduced_test_data, batch_size=batch_size)
bwa_reduced_grad_test_loader = DataLoader(bwa_reduced_test_data, batch_size=1)

# Dictionaries to store the averaged class-wise heatmaps of the classifier instances.

n = 10
bwa_reduced_heatmaps = {0: np.zeros([n, 2402]), 1: np.zeros([n, 2402]), 2: np.zeros([n, 2402]), 3: np.zeros([n, 2402]), 4: np.zeros([n, 2402]), 5: np.zeros([n, 2402]), 6: np.zeros([n, 2402])}
gbs_reduced_heatmaps = {0: np.zeros([n, 2402]), 1: np.zeros([n, 2402]), 2: np.zeros([n, 2402]), 3: np.zeros([n, 2402]), 4: np.zeros([n, 2402]), 5: np.zeros([n, 2402]), 6: np.zeros([n, 2402])}
heatmaps = {0: np.zeros([n, 2402]), 1: np.zeros([n, 2402]), 2: np.zeros([n, 2402]), 3: np.zeros([n, 2402]), 4: np.zeros([n, 2402]), 5: np.zeros([n, 2402]), 6: np.zeros([n, 2402])}

# Training and heatmap generation.

for iteration in range(n):

    print(f"\n\nCurrent Iteration: {iteration}\n\n")

    # Every iteration the classifier instances are initialised with random weights

    bwa_reduced_classifier_instance = CNN().to(device)
    gbs_reduced_classifier_instance = CNN().to(device)
    classifier_instance = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    gbs_reduced_optimizer = torch.optim.Adam(gbs_reduced_classifier_instance.parameters(), lr=learning_rate)
    bwa_reduced_optimizer = torch.optim.Adam(bwa_reduced_classifier_instance.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(classifier_instance.parameters(), lr=learning_rate)


    for epoch in range(1, num_epochs + 1):

        train_loss = 0
        running_loss = 0
        correct = 0
        total = 0

        bwa_reduced_classifier_instance.train()
        for data, label in bwa_reduced_train_loader:

            # forward pass

            data = torch.unsqueeze(data, 1)
            data = data.to(device)
            label = label.to(device)
            outputs = bwa_reduced_classifier_instance(data)
            loss = criterion(outputs, label)

            # Backward pass

            bwa_reduced_optimizer.zero_grad()
            loss.backward()
            bwa_reduced_optimizer.step()

            running_loss += loss.item()

            _, pred = outputs.max(1)
            total += label.size(0)
            correct += pred.eq(label).sum().item()

        train_loss = running_loss / len(bwa_reduced_train_loader)
        train_accu = 100. * correct / total

        test_loss = 0
        running_loss = 0
        correct = 0
        total = 0

        # Testing phase

        bwa_reduced_classifier_instance.eval()
        with torch.no_grad():
            for data, label in bwa_reduced_test_loader:
                data = torch.unsqueeze(data, 1)
                data, label = data.to(device), label.to(device)
                outputs = bwa_reduced_classifier_instance(data)
                loss = criterion(outputs, label)

                running_loss += loss.item()

                _, pred = outputs.max(1)
                total += label.size(0)
                correct += pred.eq(label).sum().item()

            test_loss = running_loss / len(bwa_reduced_test_loader)
            test_accu = 100. * correct / total

        print(f"epoch : {epoch}/{num_epochs} Train Loss: {train_loss:.3f} | Train Accuracy: {train_accu:.3f}% | "
              f"Test Loss: {test_loss:.3f} | Test Accuracy: {test_accu:.3f}%  --- BWA Reduced\n ")
        # torch.save(reduced_classifier_instance.state_dict(), "reduced_model_weights.pth")

    print("\n\n", "#" * 30, "\n\n")

    for epoch in range(1, num_epochs + 1):

        train_loss = 0
        running_loss = 0
        correct = 0
        total = 0

        gbs_reduced_classifier_instance.train()
        for data, label in gbs_reduced_train_loader:

            #forward pass

            data = torch.unsqueeze(data, 1)
            data = data.to(device)
            label = label.to(device)
            outputs = gbs_reduced_classifier_instance(data)
            loss = criterion(outputs, label)

            # backward pass

            gbs_reduced_optimizer.zero_grad()
            loss.backward()
            gbs_reduced_optimizer.step()

            running_loss += loss.item()

            _, pred = outputs.max(1)
            total += label.size(0)
            correct += pred.eq(label).sum().item()

        train_loss = running_loss / len(gbs_reduced_train_loader)
        train_accu = 100. * correct / total

        test_loss = 0
        running_loss = 0
        correct = 0
        total = 0

        # Testing phase

        gbs_reduced_classifier_instance.eval()
        with torch.no_grad():
            for data, label in gbs_reduced_test_loader:
                data = torch.unsqueeze(data, 1)
                data, label = data.to(device), label.to(device)
                outputs = gbs_reduced_classifier_instance(data)
                loss = criterion(outputs, label)

                running_loss += loss.item()

                _, pred = outputs.max(1)
                total += label.size(0)
                correct += pred.eq(label).sum().item()

            test_loss = running_loss / len(gbs_reduced_test_loader)
            test_accu = 100. * correct / total

        print(f"epoch : {epoch}/{num_epochs} Train Loss: {train_loss:.3f} | Train Accuracy: {train_accu:.3f}% | "
            f"Test Loss: {test_loss:.3f} | Test Accuracy: {test_accu:.3f}%  --- GBS Reduced\n ")
        # torch.save(reduced_classifier_instance.state_dict(), "reduced_model_weights.pth")

    print("\n\n", "#" * 30, "\n\n")

    for epoch in range(1, num_epochs + 1):

        train_loss = 0
        running_loss = 0
        correct = 0
        total = 0

        classifier_instance.train()
        for data, label in train_loader:

            # Forward pass

            data = torch.unsqueeze(data, 1)
            data = data.to(device)
            label = label.to(device)
            outputs = classifier_instance(data)
            loss = criterion(outputs, label)

            # Backward pass

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, pred = outputs.max(1)
            total += label.size(0)
            correct += pred.eq(label).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accu = 100. * correct / total


        test_loss = 0
        running_loss = 0
        correct = 0
        total = 0

        # Testing phase

        classifier_instance.eval()
        with torch.no_grad():
            for data, label in test_loader:
                data = torch.unsqueeze(data, 1)
                data, label = data.to(device), label.to(device)
                outputs = classifier_instance(data)
                loss = criterion(outputs, label)

                running_loss += loss.item()

                _, pred = outputs.max(1)
                total += label.size(0)
                correct += pred.eq(label).sum().item()

            test_loss = running_loss / len(test_loader)
            test_accu = 100. * correct / total

        print(f"epoch : {epoch}/{num_epochs} Train Loss: {train_loss:.3f} | Train Accuracy: {train_accu:.3f}% | "
              f"Test Loss: {test_loss:.3f} | Test Accuracy: {test_accu:.3f}% ---Complete\n ")
        # torch.save(classifier_instance.state_dict(), "model_weights.pth")


    # Dictionaries for storing the averaged class-wise heatmaps and its count

    bwa_reduced_hms = {0: np.zeros(2402), 1: np.zeros(2402), 2: np.zeros(2402), 3: np.zeros(2402), 4: np.zeros(2402), 5: np.zeros(2402), 6: np.zeros(2402)}
    gbs_reduced_hms = {0: np.zeros(2402), 1: np.zeros(2402), 2: np.zeros(2402), 3: np.zeros(2402), 4: np.zeros(2402), 5: np.zeros(2402), 6: np.zeros(2402)}
    hms = {0: np.zeros(2402), 1: np.zeros(2402), 2: np.zeros(2402), 3: np.zeros(2402), 4: np.zeros(2402), 5: np.zeros(2402), 6: np.zeros(2402)}
    bwa_reduced_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    gbs_reduced_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

    # Guided GradCAM of whole spectral

    print("Guided GradCAM")
    ggcam = guided_gradcam.GuidedGradCam(classifier_instance)
    for data, label in grad_test_loader:
        gghm = ggcam(data.to(device))
        hms[label.item()] += gghm
        count[label.item()] += 1

    # Gaussian based smoothing feature reduced Guided GradCAM

    print("GBS Reduced Guided GradCAM")
    ggcam = guided_gradcam.GuidedGradCam(gbs_reduced_classifier_instance)
    for data, label in gbs_reduced_grad_test_loader:

        gghm = ggcam(data.to(device), gbs.inv)
        gbs_reduced_hms[label.item()] += gghm
        gbs_reduced_count[label.item()] += 1

    # Band wise averaging feature reduced Guided GradCAM

    print("BWA Reduced Guided GradCAM")
    ggcam = guided_gradcam.GuidedGradCam(bwa_reduced_classifier_instance)
    for data, label in gbs_reduced_grad_test_loader:

        gghm = ggcam(data.to(device), bwa.inv)
        bwa_reduced_hms[label.item()] += gghm
        bwa_reduced_count[label.item()] += 1

    # Normalise and average the class-wise heatmaps

    for i in gbs_reduced_hms.keys():
        bwa_reduced_hms[i] /= bwa_reduced_count[i]
        bwa_reduced_hms[i] -= min(bwa_reduced_hms[i])
        bwa_reduced_hms[i] /= max(bwa_reduced_hms[i])
        bwa_reduced_heatmaps[i][iteration] = bwa_reduced_hms[i]
        gbs_reduced_hms[i] /= gbs_reduced_count[i]
        gbs_reduced_hms[i] -= min(gbs_reduced_hms[i])
        gbs_reduced_hms[i] /= max(gbs_reduced_hms[i])
        gbs_reduced_heatmaps[i][iteration] = gbs_reduced_hms[i]
        hms[i] /= count[i]
        hms[i] -= min(hms[i])
        hms[i] /= max(hms[i])
        heatmaps[i][iteration] = hms[i]

# Save the dictionaries for later use in band_select_test.py and main.py

np.save("bwareducedhms.npy", bwa_reduced_heatmaps)
np.save("gbsreducedhms.npy", gbs_reduced_heatmaps)
np.save("completehms.npy", heatmaps)

print("GBS: \n", sum(list(gbs_reduced_heatmaps.values())))
print("BWA: \n", sum(list(bwa_reduced_heatmaps.values())))
print("Complete: \n", sum(list(heatmaps.values())))


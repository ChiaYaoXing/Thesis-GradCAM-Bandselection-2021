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
import Bandselect
import Gaussian_Based_Smoothing
import bandselection
import gradcam
import guided_gradcam
from Data_Cube import DataCube
from Guided_BackProp import GuidedBackProp
from gradcam import GradCam as gc
from Network import CNN

print("### Process Running at snyachia@134.91.77.152:4222 ###")

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

# Dictionary of id to label
label_dict = {0: "soil", 1: "stone_1", 2: "stone_2", 3: "ear", 4: "branch", 5: "catkin", 6: "leaf"}

# Hyperparameter
num_epochs = 25
batch_size = 64
learning_rate = 0.002

bwa = Band_Wise_Averaging.BandwiseAveraging(800)
gbs = Gaussian_Based_Smoothing.GaussianBasedSmoothing(800)

# transform = transforms.Compose([bwa])
transform = transforms.Compose([bwa])
# transform = None

# 80% Train, 20% Test
xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.15)

# Data preparation

reduced_train_data_bwa = DataCube(xtrain, ytrain, bwa)
reduced_test_data_bwa = DataCube(xtest, ytest, bwa)
reduced_train_data_gbs = DataCube(xtrain, ytrain, gbs)
reduced_test_data_gbs = DataCube(xtest, ytest, gbs)
train_data = DataCube(xtrain, ytrain, None)
test_data = DataCube(xtest, ytest, None)

train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
grad_test_loader = DataLoader(test_data, batch_size=1)

reduced_train_loader_bwa = DataLoader(reduced_train_data_bwa, batch_size=batch_size)
reduced_test_loader_bwa = DataLoader(reduced_test_data_bwa, batch_size=batch_size)
reduced_grad_test_loader_bwa = DataLoader(reduced_test_data_bwa, batch_size=1)

reduced_train_loader_gbs = DataLoader(reduced_train_data_gbs, batch_size=batch_size)
reduced_test_loader_gbs = DataLoader(reduced_test_data_gbs, batch_size=batch_size)
reduced_grad_test_loader_gbs = DataLoader(reduced_test_data_gbs, batch_size=1)


accuracy_whole = []
accuracy_reduced_bwa = []
accuracy_reduced_gbs = []

# Training of 10 iterations

n = 10
for iteration in range(n):

    reduced_classifier_instance_bwa = CNN().to(device)
    reduced_classifier_instance_gbs = CNN().to(device)
    classifier_instance = CNN().to(device)


    criterion = nn.CrossEntropyLoss()
    reduced_optimizer = torch.optim.Adam(reduced_classifier_instance_bwa.parameters(), lr=learning_rate)
    reduced_optimizer_gbs = torch.optim.Adam(reduced_classifier_instance_gbs.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(classifier_instance.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):

        train_loss = 0
        running_loss = 0
        correct = 0
        total = 0

        reduced_classifier_instance_bwa.train()
        for data, label in reduced_train_loader_bwa:
            data = torch.unsqueeze(data, 1)
            data = data.to(device)
            label = label.to(device)
            outputs = reduced_classifier_instance_bwa(data)
            loss = criterion(outputs, label)

            reduced_optimizer.zero_grad()
            loss.backward()
            reduced_optimizer.step()

            running_loss += loss.item()

            _, pred = outputs.max(1)
            total += label.size(0)
            correct += pred.eq(label).sum().item()

        train_loss = running_loss / len(reduced_train_loader_bwa)
        train_accu = 100. * correct / total

        test_loss = 0
        running_loss = 0
        correct = 0
        total = 0

        reduced_classifier_instance_bwa.eval()
        with torch.no_grad():
            for data, label in reduced_test_loader_bwa:
                data = torch.unsqueeze(data, 1)
                data, label = data.to(device), label.to(device)
                outputs = reduced_classifier_instance_bwa(data)
                loss = criterion(outputs, label)

                running_loss += loss.item()

                _, pred = outputs.max(1)
                total += label.size(0)
                correct += pred.eq(label).sum().item()

            test_loss = running_loss / len(reduced_test_loader_bwa)
            test_accu = 100. * correct / total

        print(f"epoch : {epoch}/{num_epochs} Train Loss: {train_loss:.3f} | Train Accuracy: {train_accu:.3f}% | "
                  f"Test Loss: {test_loss:.3f} | Test Accuracy: {test_accu:.3f}%  --- Reduced\n ")
        accuracy_reduced_bwa.append(test_accu)

    print("\n\n", "#" * 30, "\n\n")


    for epoch in range(1, num_epochs + 1):

        train_loss = 0
        running_loss = 0
        correct = 0
        total = 0

        reduced_classifier_instance_gbs.train()
        for data, label in reduced_train_loader_gbs:
            data = torch.unsqueeze(data, 1)
            data = data.to(device)
            label = label.to(device)
            outputs = reduced_classifier_instance_gbs(data)
            loss = criterion(outputs, label)

            reduced_optimizer_gbs.zero_grad()
            loss.backward()
            reduced_optimizer_gbs.step()

            running_loss += loss.item()

            _, pred = outputs.max(1)
            total += label.size(0)
            correct += pred.eq(label).sum().item()

        train_loss = running_loss / len(reduced_train_loader_gbs)
        train_accu = 100. * correct / total

        test_loss = 0
        running_loss = 0
        correct = 0
        total = 0

        reduced_classifier_instance_gbs.eval()
        with torch.no_grad():
            for data, label in reduced_test_loader_gbs:
                data = torch.unsqueeze(data, 1)
                data, label = data.to(device), label.to(device)
                outputs = reduced_classifier_instance_gbs(data)
                loss = criterion(outputs, label)

                running_loss += loss.item()

                _, pred = outputs.max(1)
                total += label.size(0)
                correct += pred.eq(label).sum().item()

            test_loss = running_loss / len(reduced_test_loader_gbs)
            test_accu = 100. * correct / total

        print(f"epoch : {epoch}/{num_epochs} Train Loss: {train_loss:.3f} | Train Accuracy: {train_accu:.3f}% | "
              f"Test Loss: {test_loss:.3f} | Test Accuracy: {test_accu:.3f}%  --- Reduced GBS\n ")
        accuracy_reduced_gbs.append(test_accu)

    print("\n\n", "#" * 30, "\n\n")

    for epoch in range(1, num_epochs + 1):

        train_loss = 0
        running_loss = 0
        correct = 0
        total = 0

        classifier_instance.train()
        for data, label in train_loader:
            data = torch.unsqueeze(data, 1)
            data = data.to(device)
            label = label.to(device)
            outputs = classifier_instance(data)
            loss = criterion(outputs, label)

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
        accuracy_whole.append(test_accu)

    # Load the results of Guided GradHMs from varianceTest.py

    bwa_reduced_hms = np.load('bwareducedhms.npy', allow_pickle=True).item()
    reduced_hms = np.load('reducedhms.npy', allow_pickle=True).item()
    hms = np.load("hms.npy", allow_pickle=True).item()

    bwa_reduced_heatmaps = np.zeros([7, 2402])
    reduced_heatmaps = np.zeros([7, 2402])
    heatmaps = np.zeros([7, 2402])

    # Generate a matrix that contains class-wise classifier instances averaged guided gradhms

    for key, val in bwa_reduced_hms.items():
        bwa_reduced_heatmaps[key] = sum(val) / len(val)

    for key, val in reduced_hms.items():
        reduced_heatmaps[key] = sum(val) / len(val)

    for key, val in hms.items():
        heatmaps[key] = sum(val) / len(val)

    # Select the bands based on the heatmaps

    selected_reduced_bands = bandselection.select(reduced_heatmaps, 50)
    selected_bwa_reduced_bands = bandselection.select(bwa_reduced_heatmaps, 50)
    selected_bands = bandselection.select(heatmaps, 50)

    print("Selected reduced bands", selected_reduced_bands)
    print("Selected reduced bands", selected_bwa_reduced_bands)
    print("Selected bands", selected_bands)

# Transform for band selections

bs = Bandselect.BandSelect(np.array(selected_bands))
reduced_bs = Bandselect.BandSelect(np.array(selected_reduced_bands))
bwa_reduced_bs = Bandselect.BandSelect(np.array(selected_bwa_reduced_bands))

transform = transforms.Compose([bs])
reduced_transform = transforms.Compose([reduced_bs])
bwa_reduced_transform = transforms.Compose([bwa_reduced_bs])

# 80% Train, 20% Test
xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.15)

reduced_train_data_bwa = DataCube(xtrain, ytrain, reduced_transform)
reduced_test_data_bwa = DataCube(xtest, ytest, reduced_transform)
bwa_reduced_train_data = DataCube(xtrain, ytrain, bwa_reduced_transform)
bwa_reduced_test_data = DataCube(xtest, ytest, bwa_reduced_transform)
train_data = DataCube(xtrain, ytrain, transform)
test_data = DataCube(xtest, ytest, transform)

train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
grad_test_loader = DataLoader(test_data, batch_size=1)

reduced_train_loader_bwa = DataLoader(reduced_train_data_bwa, batch_size=batch_size)
reduced_test_loader_bwa = DataLoader(reduced_test_data_bwa, batch_size=batch_size)
reduced_grad_test_loader_bwa = DataLoader(reduced_test_data_bwa, batch_size=1)

bwa_reduced_train_loader = DataLoader(bwa_reduced_train_data, batch_size=batch_size)
bwa_reduced_test_loader = DataLoader(bwa_reduced_test_data, batch_size=batch_size)
bwa_reduced_grad_test_loader = DataLoader(bwa_reduced_test_data, batch_size=1)

accuracy_band_selected_whole = []
accuracy_band_Selected_bwa = []
accuracy_band_selected_gbs = []

n = 10
for iteration in range(n):

    bwa_reduced_classifier_instance = CNN().to(device)
    reduced_classifier_instance_bwa = CNN().to(device)
    classifier_instance = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    bwa_reduced_optimizer = torch.optim.Adam(bwa_reduced_classifier_instance.parameters(), lr=learning_rate)
    reduced_optimizer = torch.optim.Adam(reduced_classifier_instance_bwa.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(classifier_instance.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):

        train_loss = 0
        running_loss = 0
        correct = 0
        total = 0

        bwa_reduced_classifier_instance.train()
        for data, label in bwa_reduced_train_loader:
            data = torch.unsqueeze(data, 1)
            data = data.to(device)
            label = label.to(device)
            outputs = bwa_reduced_classifier_instance(data)
            loss = criterion(outputs, label)

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
              f"Test Loss: {test_loss:.3f} | Test Accuracy: {test_accu:.3f}%  --- Reduced\n ")
        accuracy_band_Selected_bwa.append(test_accu)

    print("\n\n", "#" * 30, "\n\n")

    for epoch in range(1, num_epochs + 1):

        train_loss = 0
        running_loss = 0
        correct = 0
        total = 0

        reduced_classifier_instance_bwa.train()
        for data, label in reduced_train_loader_bwa:
            data = torch.unsqueeze(data, 1)
            data = data.to(device)
            label = label.to(device)
            outputs = reduced_classifier_instance_bwa(data)
            loss = criterion(outputs, label)

            reduced_optimizer.zero_grad()
            loss.backward()
            reduced_optimizer.step()

            running_loss += loss.item()

            _, pred = outputs.max(1)
            total += label.size(0)
            correct += pred.eq(label).sum().item()

        train_loss = running_loss / len(reduced_train_loader_bwa)
        train_accu = 100. * correct / total

        test_loss = 0
        running_loss = 0
        correct = 0
        total = 0

        reduced_classifier_instance_bwa.eval()
        with torch.no_grad():
            for data, label in reduced_test_loader_bwa:
                data = torch.unsqueeze(data, 1)
                data, label = data.to(device), label.to(device)
                outputs = reduced_classifier_instance_bwa(data)
                loss = criterion(outputs, label)

                running_loss += loss.item()

                _, pred = outputs.max(1)
                total += label.size(0)
                correct += pred.eq(label).sum().item()

            test_loss = running_loss / len(reduced_test_loader_bwa)
            test_accu = 100. * correct / total

        print(f"epoch : {epoch}/{num_epochs} Train Loss: {train_loss:.3f} | Train Accuracy: {train_accu:.3f}% | "
              f"Test Loss: {test_loss:.3f} | Test Accuracy: {test_accu:.3f}%  --- Reduced\n ")
        accuracy_band_selected_gbs.append(test_accu)

    print("\n\n", "#" * 30, "\n\n")

    for epoch in range(1, num_epochs + 1):

        train_loss = 0
        running_loss = 0
        correct = 0
        total = 0

        classifier_instance.train()
        for data, label in train_loader:
            data = torch.unsqueeze(data, 1)
            data = data.to(device)
            label = label.to(device)
            outputs = classifier_instance(data)
            loss = criterion(outputs, label)

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

    print(
        f"epoch : {epoch}/{num_epochs} Train Loss: {train_loss:.3f} | Train Accuracy: {train_accu:.3f}% | "f"Test Loss: "
        f"{test_loss:.3f} | Test Accuracy: {test_accu:.3f}% ---Complete\n ")
    accuracy_band_selected_whole.append(test_accu)


print("Final Results: \n")
print(f"Accuracy of {accuracy_whole = }")
print(f"Accuracy of {accuracy_reduced_bwa = }")
print(f"Accuracy of {accuracy_reduced_gbs = }")
print(f"Accuracy of {accuracy_band_selected_whole = }")
print(f"Accuracy of {accuracy_band_Selected_bwa = }")
print(f"Accuracy of {accuracy_band_selected_gbs = }")



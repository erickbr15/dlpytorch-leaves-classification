# Custom ResNet18 Neural Network for Leaf Classification and Health Assessment

## Project Methodology Overview

The core of this project revolves around the development of specialized neural network models for two key tasks: species classification and health status assessment of plant leaves.

- **Species Classification Model:** A robust model was designed to identify the species of a plant from its leaf image. This model can classify leaves into various species, providing a fundamental layer of analysis.
- **Health Status Classification Models:** For each species with sufficient data, an individual model was trained to determine the health status of the leaf. These models are capable of differentiating at least two classes, offering a deeper insight into the plant's condition.

## Repository and Structure

The repository is structured as follows to facilitate navigation and understanding:

- **`/models`**: Stores the trained model files, which can be used for direct predictions without the need for retraining.

  | Model                        | File                        |
  |------------------------------|-----------------------------|
  | Leaf Specie Classification   | `leaf_specie_clmodel`       |
  | Apple Leaf Health Classification | `apple_model`            |
  | Cherry Leaf Health Classification | `cherry_model`          |
  | Corn Leaf Health Classification | `corn_model`              |
  | Grape Leaf Health Classification | `grape_model`            |
  | Peach Leaf Health Classification | `peach_model`            |
  | Pepper Bell Leaf Health Classification | `pepperbell_model`  |
  | Potato Leaf Health Classification | `potato_model`           |
  | Strawberry Leaf Health Classification | `strawberry_model`   |
  | Tomato Leaf Health Classification | `tomato_model`          |

- **`/notebooks`**: Jupyter notebooks demonstrating the model training process and prediction examples are located here.

  | Model                   | Classes      | Notebook                                  |
  |-------------------------|--------------|-------------------------------------------|
  | Species                 | Multi-Class  | `LeafSpecies_Classification.ipynb`        |
  | Tomato health           | Multi-Class  | `TomatoLeafHealth_Classification.ipynb`   |
  | Strawberry health       | Binary       | `StrawberryLeafHealth_Classification.ipynb`|
  | Potato health           | Multi-Class  | `PotatoLeafHealth_Classification.ipynb`   |
  | Pepper-bell health      | Binary       | `PeperBellLeafHealth_Classification.ipynb`|
  | Peach health            | Binary       | `PeachLeafHealth_Classification.ipynb`    |
  | Grape health            | Multi-Class  | `GrapeLeafHealth_Classification.ipynb`    |
  | Corn health             | Multi-Class  | `CornLeafHealth_Classification.ipynb`     |
  | Cherry health           | Binary       | `CherryLeafHealth_Classification.ipynb`   |
  | Apple health            | Multi-Class  | `AppleLeafHealth_Classification.ipynb`    |
  | Predictions             | -            | `Leaves_And_Health_Predictions.ipynb`     |


## Problem Context

Crop diseases pose a significant threat to food security, yet their rapid identification remains a challenge in many parts of the world due to infrastructure limitations. The widespread increase in smartphone usage globally and recent advances in computer vision through deep learning have paved the way for smartphone-assisted disease diagnosis. Generally, the approach of utilizing deep learning on increasingly large and publicly available image datasets represents a clear path toward smartphone-assisted crop disease diagnosis at a massive global scale.

## Overview

This documentation outlines the development of a custom neural network based on the ResNet18 architecture for the classification of leaves. The primary goal is twofold: firstly, to identify the species of the leaf, and secondly, to determine its health status. The project is structured as the first phase in a larger series of experiments.

## Objective

The objective of this activity is to develop a convolutional neural network-based model for the identification of plant species and their health status from leaf images. For training and validating the model, a public dataset of 54,306 images of healthy and diseased plant leaves, collected under controlled conditions, is available. This dataset includes 14 crop species and 26 diseases (or absence thereof).

## Dataset Description

Upon accessing the dataset, you will find the following file organization:

```
dataset
|------- color     RGB images, 256x256
|------- grayscale Grayscale images, 256x256
|------- segmented Segmented images, 256x256
```

In each folder, images are organized according to the class they belong to and their health status (indicated by the folder name). Here are some examples:

| Folder Name              | Leaf Type    | Health Status |
|--------------------------|--------------|---------------|
| apple___apple_scab       | apple        | not healthy   |
| apple___healthy          | apple        | healthy       |
| blueberry___healthy      | blueberry    | healthy       |
| Strawberry___Leaf_scorch | Strawberry   | not healthy   |

## Implementation Details

### Species Classification Model

#### Notebook
| Species     | Notebook       | Optimizer | Loss Function                      |
|-------------|----------------|-----------|------------------------------------|
| Multi-Class | LeafSpecies_Classification.ipynb   | Adam      | Cross Entropy |

#### Custom Dataset Creation
- A custom dataset was developed to read the given folder structure and transform images into tensors of shape 3x224x224.

```
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from collections import Counter
from random import sample

class LeafSpeciesDataset(Dataset):
    def __init__(self, root_dir, transform=None, subset=None):
        self.root_dir = root_dir
        self.transform = transform
        self.subset = subset
        self.samples, self.category_counts = self._load_samples()
        self.species_to_idx = {species: idx for idx, species in enumerate(self.category_counts)}

    def _load_samples(self):
      all_samples = {}
      category_counts = Counter()

      for folder_name in os.listdir(self.root_dir):
          folder_path = os.path.join(self.root_dir, folder_name)
          leaf_specie = folder_name.split('___')[0].lower()
          for img_name in os.listdir(folder_path):
              img_path = os.path.join(folder_path, img_name)
              if leaf_specie not in all_samples:
                  all_samples[leaf_specie] = []
              all_samples[leaf_specie].append((img_path, leaf_specie))

      samples = []
      for specie, specie_samples in all_samples.items():
          if self.subset and len(specie_samples) > self.subset:
              sampled_images = sample(specie_samples, self.subset)
          else:
              sampled_images = specie_samples
          samples.extend(sampled_images)
          category_counts[specie] = len(sampled_images)

      return samples, category_counts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, species = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        one_hot = torch.zeros(len(self.category_counts))
        one_hot[self.species_to_idx[species]] = 1

        return img, species, one_hot

    def get_category_counts(self):
        return self.category_counts

    def get_species_to_idx(self):
      return self.species_to_idx
```

#### Class Balance Analysis
- A bar chart was generated to check for class balance.
- Classes requiring synthetic data augmentation were identified.

#### Synthetic Data Generation
- A data augmentation generator was coded with customizable image transformations.
- Classes were balanced using these synthetic transformations.

#### DataLoader Setup
- DataLoaders for training, validation, and testing were created.
- Data distribution: 80% for training, 2% for validation, and the remaining 18% for testing.

#### Network Customization
- ResNet18 was customized to handle 3x224x224 tensors and output for 14 classes.

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class MyResNet18_LeafSpecieClassification(nn.Module):
    def __init__(self, block, layers, num_classes=14):
        super(MyResNet18_LeafSpecieClassification, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
```

#### Training Method
- A training method was implemented combining training and validation in each batch.
- Persistent model checkpoints were created upon each improvement.

```
def fit(model, train_loader, val_loader, epochs, optimizer, loss_fn, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, _, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}')

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, _, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += loss_fn(outputs, labels).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Model improved and saved to {checkpoint_path}')
```

#### Training Process
- The optimizer used was Adam with a learning rate of 0.001, and Cross Entropy Loss as the loss function.
- The model was trained for 10 epochs using a minimal percentage of the total data for experimental and validation purposes.

#### Note on Training Infrastructure
- Training and execution were conducted using CUDA/GPUs on Colab.
- It is emphasized that there should be an iteration of the entire training process using 100% of the dataset on more capable infrastructure resources.

#### Performance Measurement
- The model's performance was measured using the test dataset.
- An example method was coded to demonstrate how to use the persisted model for making inferences from an image file.

### Health State Classification Model

The process for the health state classification was conducted similarly to the species classification, with the distinction that it was carried out for each species whose data supported the classification of health state. Out of the 14 species classes, the following have data for health state classification: tomato, strawberry, potato, pepper-bell, peach, grape, corn, cherry, and apple. Of these, the following allowed only for binary classification: strawberry, pepper-bell, peach, and cherry. The loss function used for binary classification was `binary_cross_entropy_with_logits`.

Here's a table summarizing the details for each species:

| Species     | Health States  | Notebook       | Optimizer | Loss Function                      |
|-------------|----------------|----------------|-----------|------------------------------------|
| Tomato      | Multi-Class    | TomatoLeafHealth_Classification.ipynb   | Adam      | Cross Entropy                      |
| Strawberry  | Binary         | StrawberryLeafHealth_Classification.ipynb | Adam      | Binary Cross Entropy with Logits   |
| Potato      | Multi-Class    | PotatoLeafHealth_Classification.ipynb   | Adam      | Cross Entropy                      |
| Pepper-Bell | Binary         | PeperBellLeafHealth_Classification.ipynb   | Adam      | Binary Cross Entropy with Logits   |
| Peach       | Binary         | PeachLeafHealth_Classification.ipynb   | Adam      | Binary Cross Entropy with Logits   |
| Grape       | Multi-Class    | GrapeLeafHealth_Classification.ipynb   | Adam      | Cross Entropy                      |
| Corn        | Multi-Class    | CornLeafHealth_Classification.ipynb   | Adam      | Cross Entropy                      |
| Cherry      | Binary         | CherryLeafHealth_Classification.ipynb   | Adam      | Binary Cross Entropy with Logits   |
| Apple       | Multi-Class    | AppleLeafHealth_Classification.ipynb   | Adam      | Cross Entropy                      |

*Note: The notebook column represents the specific Jupyter Notebook used for each species. *

---

## Model Evaluation Results

| Model                                | Total Images | Number of Classes | Test Loss | Test Accuracy |
|--------------------------------------|--------------|-------------------|-----------|---------------|
| Leaf Specie Classification           | 5180         | 14                | 1.1782    | 60%           |
| Apple Leaf Health Classification     | 880          | 4                 | 0.4333    | 67%           |
| Cherry Leaf Health Classification    | 1600         | 2                 | 0.0025    | 100%          |
| Corn Leaf Health Classification      | 1760         | 4                 | 0.1111    | 100%          |
| Grape Leaf Health Classification     | 1600         | 4                 | 0.2614    | 100%          |
| Peach Leaf Health Classification     | 700          | 2                 | 0.0004    | 100%          |
| Pepper Bell Leaf Health Classification | 1800       | 2                 | 0.0059    | 100%          |
| Potato Leaf Health Classification    | 450          | 3                 | 0.0063    | 100%          |
| Strawberry Leaf Health Classification | 800         | 2                 | 0.0051    | 100%          |
| Tomato Leaf Health Classification    | 2400         | 10                | 1.6017    | 56%           |

**Note:** The data distribution for training, validation, and testing follows the rule of 80%, 18%, and 2%, respectively, of the total data provided for each exercise.

## Prediction Exercise

In this section, we delve into the prediction exercise that was performed using our trained models. The goal was to predict both the species and the health status of various leaf images.

### Notebook
Leafs_And_Health_Classification.ipynb

### Data Preparation

A comprehensive dataset was prepared, containing image paths, actual and predicted species, actual and predicted health status, and the correctness of each prediction.

### Prediction Methodology

The prediction process involved the following steps:

1. **Model Loading:** The trained neural networks were loaded along with the corresponding category dictionaries.

2. **Prediction Function:** A prediction function was defined to process each image and return species and health status predictions.

3. **Data Processing:** For each image in our dataset, the prediction function was called, and its output was stored.

```
def classify_leaf_specie(image):
    
    leaf_species_model.eval()
    with torch.no_grad():
        output = leaf_species_model(image)

    probabilities = torch.nn.functional.softmax(output, dim=1)
    _, pred_idx = torch.max(probabilities, 1)
    predicted_specie = species_dictionary[pred_idx.item()]

    return predicted_specie

def classify_leaf_health(image, model, classes):    
    model.eval()
    with torch.no_grad():
        output = model(image)

    probabilities = torch.nn.functional.softmax(output, dim=1)
    _, pred_idx = torch.max(probabilities, 1)

    predicted_health = classes[pred_idx.item()]

    return predicted_health

def classify_leaf_specie_and_health(image_path):

    transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert('RGB')
    transformed_image = transformations(image).unsqueeze(0).to('cuda')

    predicted_specie = classify_leaf_specie(transformed_image)
    predicted_health = ''

    health_classification_model = get_health_classification_model(predicted_specie)

    if health_classification_model != None:
      health_classes_dictionary = get_health_classes_dictionary(predicted_specie)
      predicted_health = classify_leaf_health(transformed_image, health_classification_model, health_classes_dictionary)
    
    return predicted_specie, predicted_health
```

### Results

The prediction results were tabulated, and each entry included:

- Image path
- Real species
- Predicted species
- Correctness of species prediction
- Real health status
- Predicted health status
- Correctness of health status prediction

|index|Image\_Path|Real\_Specie|Predicted\_Specie|Specie\_Correct|Real\_Health|Predicted\_Health|Health\_Correct|
|---|---|---|---|---|---|---|---|
|0|apple\_\_\_apple\_scab/97d9b70c-15fc-4463-a7cd-129e7bccef1c\_\_\_FREC\_Scab 2910\.JPG|apple|squash|FAIL|apple\_scab||FAIL|
|1|apple\_\_\_black\_rot/f15c02d9-9aa6-45b5-84fc-97e22f7fabaa\_\_\_JR\_FrgE\.S 2831\.JPG|apple|pepper,\_bell|FAIL|black\_rot||FAIL|
|2|apple\_\_\_healthy/bdcce69f-7520-48cc-b22b-c3ad33d5be71\_\_\_RS\_HL 7878\.JPG|apple|apple|SUCCESS|healthy|healthy|SUCCESS|
|3|blueberry\_\_\_healthy/d2dec908-fb8f-464b-b427-8f112340d649\_\_\_RS\_HL 0570\.JPG|blueberry|blueberry|SUCCESS|healthy||FAIL|
|4|grape\_\_\_black\_rot/e2dca6ce-4b58-4850-b6c0-1f15189fe944\_\_\_FAM\_B\.Rot 0431\.JPG|grape|grape|SUCCESS|black\_rot|leaf\_blight\_\(isariopsis\_leaf\_spot\)|FAIL|
|5|grape\_\_\_healthy/2b4a92e2-a039-45eb-ad1a-ee9914dfeefe\_\_\_Mt\.N\.V\_HL 6177\.JPG|grape|grape|SUCCESS|healthy|leaf\_blight\_\(isariopsis\_leaf\_spot\)|FAIL|
|6|peach\_\_\_healthy/95f38d4e-5e8d-4b72-ac83-bd7e794a3093\_\_\_Rutg\.\_HL 2498\.JPG|peach|peach|SUCCESS|healthy|healthy|SUCCESS|
|7|potato\_\_\_healthy/ff700844-68ad-4e99-8427-58a39c07f817\_\_\_RS\_HL 1860\.JPG|potato|potato|SUCCESS|healthy|healthy|SUCCESS|
|8|potato\_\_\_late\_blight/1789b0d1-d850-4e04-8a20-bc3901d4ab0a\_\_\_RS\_LB 5251\.JPG|potato|cherry\_\(including\_sour\)|FAIL|late\_blight|powdery\_mildew|FAIL|
|9|raspberry\_\_\_healthy/1bf5e5e4-d00c-442c-b153-497aa3ed9278\_\_\_Mary\_HL 6240\.JPG|raspberry|grape|FAIL|healthy|healthy|SUCCESS|
|10|soybean\_\_\_healthy/51720e52-cf2a-4818-889f-a7426e764421\_\_\_RS\_HL 2841\.JPG|soybean|soybean|SUCCESS|healthy||FAIL|
|11|tomato\_\_\_healthy/67549f03-2c86-44b7-9730-32d4a23287f3\_\_\_GH\_HL Leaf 495\.2\.JPG|tomato|tomato|SUCCESS|healthy|healthy|SUCCESS|
|12|tomato\_\_\_leaf\_mold/59be5fe5-77f9-4538-a3c9-8c3be20f662d\_\_\_Crnl\_L\.Mold 8938\.JPG|tomato|tomato|SUCCESS|leaf\_mold|leaf\_mold|SUCCESS|
|13|apple\_\_\_cedar\_apple\_rust/9ace7aaf-8950-43b5-bafb-2c63f8839c20\_\_\_FREC\_C\.Rust 9867\.JPG|apple|peach|FAIL|cedar\_apple\_rust|bacterial\_spot|FAIL|
|14|cherry\_\(including\_sour\)\_\_\_healthy/f9236ca5-ea7c-4a87-bac0-396883f95ba6\_\_\_JR\_HL 9838\.JPG|cherry\_\(including\_sour\)|cherry\_\(including\_sour\)|SUCCESS|healthy|healthy|SUCCESS|
|15|cherry\_\(including\_sour\)\_\_\_powdery\_mildew/bda9d7d5-617a-4159-92f2-a30a05396091\_\_\_FREC\_Pwd\.M 4924\.JPG|cherry\_\(including\_sour\)|cherry\_\(including\_sour\)|SUCCESS|powdery\_mildew|powdery\_mildew|SUCCESS|
|16|corn\_\(maize\)\_\_\_cercospora\_leaf\_spot gray\_leaf\_spot/9103b8e5-919c-4d08-a282-25176874769c\_\_\_RS\_GLSp 4653\.JPG|corn\_\(maize\)|corn\_\(maize\)|SUCCESS|cercospora\_leaf\_spot gray\_leaf\_spot|cercospora\_leaf\_spot gray\_leaf\_spot|SUCCESS|
|17|corn\_\(maize\)\_\_\_common\_rust\_/RS\_Rust 1591\.JPG|corn\_\(maize\)|corn\_\(maize\)|SUCCESS|common\_rust\_|common\_rust\_|SUCCESS|
|18|corn\_\(maize\)\_\_\_healthy/b914d1d7-db9b-4db3-9360-1ba44beef18b\_\_\_R\.S\_HL 8176 copy 2\.jpg|corn\_\(maize\)|corn\_\(maize\)|SUCCESS|healthy|healthy|SUCCESS|
|19|corn\_\(maize\)\_\_\_northern\_leaf\_blight/7220bfaa-b955-46c6-994b-00810d0e65b3\_\_\_RS\_NLB 3964\.JPG|corn\_\(maize\)|corn\_\(maize\)|SUCCESS|northern\_leaf\_blight|northern\_leaf\_blight|SUCCESS|
|20|grape\_\_\_esca\_\(black\_measles\)/c1015ba9-4629-43eb-90a2-accfed27b787\_\_\_FAM\_B\.Msls 1830\.JPG|grape|grape|SUCCESS|esca\_\(black\_measles\)|esca\_\(black\_measles\)|SUCCESS|
|21|grape\_\_\_leaf\_blight\_\(isariopsis\_leaf\_spot\)/a77906c9-3b5d-4612-bbd7-0ce3f223882c\_\_\_FAM\_L\.Blight 4876\.JPG|grape|grape|SUCCESS|leaf\_blight\_\(isariopsis\_leaf\_spot\)|leaf\_blight\_\(isariopsis\_leaf\_spot\)|SUCCESS|
|22|orange\_\_\_haunglongbing\_\(citrus\_greening\)/28cc3cb0-6bf8-4476-a6fb-59c5af49ff89\_\_\_UF\.Citrus\_HLB\_Lab 1456\.JPG|orange|pepper,\_bell|FAIL|haunglongbing\_\(citrus\_greening\)||FAIL|
|23|peach\_\_\_bacterial\_spot/a0b7956c-841a-4f49-a163-c627c47fe43a\_\_\_Rutg\.\_Bact\.S 1853\.JPG|peach|peach|SUCCESS|bacterial\_spot|bacterial\_spot|SUCCESS|
|24|pepper,\_bell\_\_\_bacterial\_spot/a72dbf23-65d1-40c6-a7bc-82caed00c6d3\_\_\_JR\_B\.Spot 3333\.JPG|pepper,\_bell|pepper,\_bell|SUCCESS|bacterial\_spot||FAIL|
|25|pepper,\_bell\_\_\_healthy/53c14233-6e5f-4775-acfe-212000c81ffa\_\_\_JR\_HL 6009\.JPG|pepper,\_bell|cherry\_\(including\_sour\)|FAIL|healthy|healthy|SUCCESS|
|26|potato\_\_\_early\_blight/b76550de-8e3a-46f1-b06f-6bd4ed3dc8a5\_\_\_RS\_Early\.B 8456\.JPG|potato|potato|SUCCESS|early\_blight|late\_blight|FAIL|
|27|squash\_\_\_powdery\_mildew/df266ee0-67e4-439f-b3d0-d00d8257aa55\_\_\_MD\_Powd\.M 0865\.JPG|squash|squash|SUCCESS|powdery\_mildew||FAIL|
|28|strawberry\_\_\_healthy/4005fb13-0d7c-4a30-9ee3-73e9e4cee05e\_\_\_RS\_HL 1688\.JPG|strawberry|grape|FAIL|healthy|healthy|SUCCESS|
|29|strawberry\_\_\_leaf\_scorch/f7d50599-f99b-4b36-ada8-712428030a2e\_\_\_RS\_L\.Scorch 0945\.JPG|strawberry|strawberry|SUCCESS|leaf\_scorch|leaf\_scorch|SUCCESS|
|30|tomato\_\_\_bacterial\_spot/144352ee-0f8d-44cc-9db1-c4f27eb5a00a\_\_\_GCREC\_Bact\.Sp 3284\.JPG|tomato|tomato|SUCCESS|bacterial\_spot|bacterial\_spot|SUCCESS|
|31|tomato\_\_\_early\_blight/d53bdcff-1ea7-46c7-9d46-701d104bf130\_\_\_RS\_Erly\.B 8353\.JPG|tomato|peach|FAIL|early\_blight|bacterial\_spot|FAIL|
|32|tomato\_\_\_late\_blight/01a68044-9c5b-4658-a944-6108c6862ce7\_\_\_GHLB Leaf 2\.1 Day 16\.JPG|tomato|peach|FAIL|late\_blight|healthy|FAIL|
|33|tomato\_\_\_septoria\_leaf\_spot/efb0fe31-821d-4259-8daf-495b04f0a0d1\_\_\_JR\_Sept\.L\.S 8503\.JPG|tomato|peach|FAIL|septoria\_leaf\_spot|bacterial\_spot|FAIL|
|34|tomato\_\_\_spider\_mites two-spotted\_spider\_mite/a67b81b4-ba9a-48eb-99cd-be36350a787e\_\_\_Com\.G\_SpM\_FL 8906\.JPG|tomato|tomato|SUCCESS|spider\_mites two-spotted\_spider\_mite|spider\_mites two-spotted\_spider\_mite|SUCCESS|
|35|tomato\_\_\_target\_spot/40d81563-afde-4956-a67d-695d22449170\_\_\_Com\.G\_TgS\_FL 7993\.JPG|tomato|tomato|SUCCESS|target\_spot|tomato\_mosaic\_virus|FAIL|
|36|tomato\_\_\_tomato\_mosaic\_virus/57b10f20-7819-40b1-924c-b484c42c515f\_\_\_PSU\_CG 2356\.JPG|tomato|tomato|SUCCESS|tomato\_mosaic\_virus|tomato\_mosaic\_virus|SUCCESS|
|37|tomato\_\_\_tomato\_yellow\_leaf\_curl\_virus/a52cd19b-99e6-495e-bd59-c8695a0b3d17\_\_\_YLCV\_GCREC 5256\.JPG|tomato|pepper,\_bell|FAIL|tomato\_yellow\_leaf\_curl\_virus||FAIL|


### Conclusions

The approach of deploying a multi-class classification model for species identification and individual models for assessing leaf health has proven to be a viable and effective solution. This approach not only facilitates accuracy in predictions but also offers remarkable flexibility, allowing for adjustments and enhancements to the models as needed, particularly when it involves adding or modifying datasets in the model training phase.

Furthermore, I conclude that there is significant scope for further enhancing the accuracy and efficiency of these models. Training them with the full extent of the available data, using more specialized infrastructure tailored for deep learning model training, could lead to substantial improvements in performance. This step is crucial for advancing towards more precise and efficient diagnostics in disease identification and overall plant health assessment, holding considerable potential for practical applications in agriculture and botany.

I invite others to use this work as a foundation for their own implementations, encouraging further exploration and innovation in this promising field.

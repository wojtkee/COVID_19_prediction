# Dataset
## Link to dataset:
## https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset
The project utilized a publicly available dataset - the Covid-19 Image Dataset, containing diverse X-ray images categorized into three classes. The dataset owner provided a pre-defined split between training and testing sets, eliminating the need for manual data splitting. This stable division ensures objectivity in model comparisons and facilitates the comparability of experimental results.

The images in the dataset exhibit various file types, primarily in jpg and png formats, as well as different resolutions. To standardize the data, the images were scaled to a size of 432x432 px in the dataloader. Additionally, some images required conversion from RGB to grayscale. The final step in image preparation involved normalization, setting mean=0.5 and std=0.5. Normalized images were then transformed into tensors by the requirements of the PyTorch library.

# Modele Sieci Neuronowych

| Model                         | Epoki | Accuracy | Precision | Recall | F1 Score |
|-------------------------------|-------|----------|-----------|--------|----------|
| CNN1                          | 20    | 96.97%   | 97.11%    | 96.97% | 96.95%   |
| CNN2                          | 20    | 100.0%   | 100.0%    | 100.0% | 100.0%   |
| U-Net                         | 30    | 100.0%   | 100.0%    | 100.0% | 100.0%   |
| ResNet (Przetrenowany)        | 15    | 100.0%   | 100.0%    | 100.0% | 100.0%   |
| ResNet (Nieprzetrenowany)     | 20    | 96.97%   | 97.11%    | 96.97% | 96.95%   |


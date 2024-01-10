
import torch
import torchvision.transforms as transforms
import numpy as np
import os
import sys
from PIL import Image
import pickle

def load_model(model_path):
    # Load the pre-trained model
    model = torch.load(model_path)
    model.eval()
    return model

def load_images_from_folder(folder_path):
    # Load and transform images from a folder
    images = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Example resize, adjust as needed
        transforms.ToTensor(),
    ])
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img = transform(img)
                images.append(img)
    return torch.stack(images)

def load_images_from_file(file_path):
    # Load images from a numpy or pickle file, or a single image file
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to 28x28 for MNIST
        transforms.Grayscale(),       # Convert to grayscale
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the image
    ])

    if file_path.endswith('.npy'):
        images = np.load(file_path)
        return torch.from_numpy(images)
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            images = pickle.load(f)
        return torch.from_numpy(images)
    elif file_path.endswith('.jpg') or file_path.endswith('.png'):
        with Image.open(file_path) as img:
            img = img.convert('RGB')
            img = transform(img)
            return img.unsqueeze(0)  # Add batch dimension
    else:
        raise ValueError("Unsupported file format")

def predict(model, images):
    # Make predictions with the model
    with torch.no_grad():
        predictions = model(images)
    return predictions

def main():
    if len(sys.argv) != 3:
        print("Usage: python predict_model.py <model_path> <data_path>")
        sys.exit(1)

    model_path, data_path = sys.argv[1], sys.argv[2]

    # Load the model
    model = load_model(model_path)

    # Load the images
    if os.path.isdir(data_path):
        images = load_images_from_folder(data_path)
    elif os.path.isfile(data_path):
        images = load_images_from_file(data_path)
    else:
        raise ValueError("Data path must be a folder or file")

    # Make predictions
    predictions = predict(model, images)
    print(predictions)

if __name__ == "__main__":
    main()
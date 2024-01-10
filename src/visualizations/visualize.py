import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import resnet18  # Example model, replace with your own

def load_model():
    # Load the pre-trained model
    model = resnet18(pretrained=True)
    model.eval()
    return model

def get_intermediate_layer(model, layer_name):
    # Function to fetch an intermediate layer
    def hook(model, input, output):
        return output

    layer = getattr(model, layer_name)
    handle = layer.register_forward_hook(hook)
    return handle

def extract_features(model, loader, layer_handle):
    # Extract features from the intermediate layer
    features = []
    with torch.no_grad():
        for inputs, _ in loader:
            _ = model(inputs)
            features.append(layer_handle.output.cpu().numpy())
    return features

def visualize_with_tsne(features, output_file):
    # Use t-SNE for dimensionality reduction and visualize
    tsne = TSNE(n_components=2, random_state=0)
    reduced_features = tsne.fit_transform(features)
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1])
    plt.savefig(output_file)

def main():
    model = load_model()
    layer_handle = get_intermediate_layer(model, 'layer4')  # Replace 'layer4' as needed

    # Prepare your dataset
    transform = transforms.Compose([transforms.Resize(256), 
                                    transforms.CenterCrop(224), 
                                    transforms.ToTensor()])
    dataset = datasets.ImageFolder('path/to/your/dataset', transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    features = extract_features(model, loader, layer_handle)
    visualize_with_tsne(features, 'reports/figures/tsne_visualization.png')

if __name__ == "__main__":
    main()

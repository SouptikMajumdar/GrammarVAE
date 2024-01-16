import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.decomposition import PCA 
import torch
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)

# GPU operations have a separate seed we also want to set
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Additionally, some operations on a GPU are implemented stochastic for efficiency
# We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def visualize_latent_space_Eqn(model,data_loader,vae=False):
    model.eval()
    # Extract latent vectors
    latent_vectors = []
    for batch in data_loader:
        # Assuming batch is your input tensor
        with torch.no_grad():
            batch = batch.float().to(device)
            if vae:
                mean, log_var = model.encode(batch)
                z = model.sampler(mean,log_var)
            else:
                z = model.encode(batch)
            latent_vectors.append(z)

    latent_vectors = torch.cat(latent_vectors, dim=0).numpy()

    # Dimensionality reduction using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_vectors_2d = tsne.fit_transform(latent_vectors)

    # Visualization
    plt.scatter(latent_vectors_2d[:, 0], latent_vectors_2d[:, 1])
    plt.colorbar()
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.title('Latent Space Visualization')
    plt.show()


def visualize_img(original,recon):
    # Visualize the input and decoded images side by side
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(original, cmap='gray')  # Assuming it's a grayscale image
    axes[0].axis('off')
    axes[0].set_title('Original Image')

    axes[1].imshow(recon, cmap='gray')  # Assuming it's a grayscale image
    axes[1].axis('off')
    axes[1].set_title('Generated Image')

    plt.show()

def visualize_latent_space(train_loader,model):
    encoded_data_list = []
    labels_list = []
    for sample_batch in train_loader:
        data, labels = sample_batch[0].to(device), sample_batch[1].to(device)

        # Assuming 'AEmodel' is your autoencoder model
        encoded_data = model.encoder(data)
        encoded_data_list.append(encoded_data.cpu().detach().numpy())
        labels_list.append(labels.cpu().detach().numpy())

    # Concatenate the lists into arrays
    all_encoded_data = np.concatenate(encoded_data_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    # Use PCA for dimensionality reduction to 2D for visualization
    pca = PCA(n_components=2)
    encoded_data_pca = pca.fit_transform(all_encoded_data)

    # Visualize the entire encoded data in 2D space with color-coded labels
    plt.scatter(encoded_data_pca[:, 0], encoded_data_pca[:, 1], cmap='hot', c=all_labels, alpha=0.5)
    plt.colorbar()
    plt.title('Encoded Data Space Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
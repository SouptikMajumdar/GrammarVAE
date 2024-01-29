import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set()
from sklearn.decomposition import PCA 
import torch
import numpy as np
from sklearn.manifold import TSNE
import mlflow
from ac_dll_grammar_vae.data.transforms import MathTokenEmbedding, RuleTokenEmbedding, OneHotEncode 
from ac_dll_grammar_vae.data.alphabet import alphabet
from sklearn.decomposition import PCA
import random

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = "cpu"
print("Device", device)

# GPU operations have a separate seed we also want to set
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Additionally, some operations on a GPU are implemented stochastic for efficiency
# We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#device = "cpu"

def visualize_latent_space_EqnGVAE(model,data_loader,max_num_rules=16,cfg=None,vae=True):
    model.eval()
    # Extract latent vectors
    latent_vectors = []
    for batch in data_loader:
        with torch.no_grad():
            batch = batch.float().to(device)
            if vae:
                mean, log_var = model.encode(batch)
                z = model.sampler(mean,log_var)
            else:
                z = model.encode(batch)
            latent_vectors.append(z)
    
    labels = []
    emb = RuleTokenEmbedding(cfg,max_num_rules=max_num_rules,one_hot_encode=True)
    for sample in data_loader:
        sample = sample.float()
        for idx,ele in enumerate(sample):
            eqn = emb.decode(torch.argmax(sample[idx], dim=1))
            eqn = ''.join(eqn)
            labels.append(eqn)
        break

    latent_vectors = torch.cat(latent_vectors, dim=0).cpu().numpy()
    print(latent_vectors.shape)
    # Dimensionality reduction using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_vectors_2d = tsne.fit_transform(latent_vectors)

    ann_latent_vectors_2d = latent_vectors_2d[:10,:]
    try:
        plt.clf()
    except:
        pass
    # Visualization
    plt.scatter(latent_vectors_2d[:, 0], latent_vectors_2d[:, 1])
    for i, label in enumerate(labels[:10]):
        plt.scatter(ann_latent_vectors_2d[i, 0], ann_latent_vectors_2d[i, 1], label=label)
        plt.text(ann_latent_vectors_2d[i, 0], ann_latent_vectors_2d[i, 1], label)

    plt.colorbar()
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.title('Latent Space Visualization')
    #plt.show()
    plot_filename = "LatentSpace_plot.png"
    plt.savefig(f'./plots/{plot_filename}')
    plt.close()

def visualize_latent_space_Eqn_PCA(model,data_loader,vae=False,gvae=False,cfg=None,n=20,seed=42):
    model.eval()
    # Extract latent vectors
    latent_vectors = []
    for batch in data_loader:
        with torch.no_grad():
            batch = batch.float().to(device)
            if vae:
                mean, log_var = model.encode(batch)
                z = model.sampler(mean,log_var)
            else:
                z = model.encode(batch)
            latent_vectors.append(z)
    
    #Extract labels for first 10 equations:
    labels = []
    if not gvae:
        emb = MathTokenEmbedding(alphabet=alphabet)
        for sample in data_loader:
            sample = sample.float()
            for idx,ele in enumerate(sample):
                eqn = emb.decode(torch.argmax(sample[idx], dim=1))
                eqn = ''.join(eqn)
                labels.append(eqn)
    else:
        emb = RuleTokenEmbedding(cfg=cfg,one_hot_encode=True)
        for sample in data_loader:
            sample = sample.float()
            for idx,ele in enumerate(sample):
                eqn = emb.decode(sample[idx].numpy())
                eqn = ''.join(eqn)
                labels.append(eqn)
            

    latent_vectors = torch.cat(latent_vectors, dim=0).cpu().numpy()
    print(latent_vectors.shape)
    # Dimensionality reduction using PCA
    pca = PCA(n_components=2)
    latent_vectors_2d = pca.fit_transform(latent_vectors)
    random.seed(seed)
    ind = random.sample(range(len(latent_vectors_2d)), n)
    print(ind)
    try:
        plt.clf()
    except:
        pass
    # Visualization
    plt.scatter(latent_vectors_2d[:, 0], latent_vectors_2d[:, 1])
    for i in ind:
        plt.scatter(latent_vectors_2d[i, 0], latent_vectors_2d[i, 1], label=labels[i])
        
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
    #plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    #plt.show()

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Latent Space Visualization using PCA')
    plot_filename = "PCA_LatentSpace_plot.png"
    plt.savefig(f'./plots/{plot_filename}')
    plt.show()
    plt.close()

def vis_interpolation(model,data_loader,vae=False):
    def interpolate(start, end, steps):
        return np.array([start + i*(end-start)/(steps-1) for i in range(steps)])

    def decode_latent_points(emb, vectors, model,vae):
        if vae:
            decoded ,_ ,_ = model(vectors)
        else:
            decoded = model(vectors)
        labels = []
        for sample in decoded:
            sample = sample.float()
            for idx,ele in enumerate(sample):
                eqn = emb.decode(torch.argmax(sample[idx], dim=1))
                eqn = ''.join(eqn)
                labels.append(eqn)

        return labels

    model.eval()
    # Extract latent vectors
    latent_vectors = []
    for batch in data_loader:
        with torch.no_grad():
            batch = batch.float().to(device)
            if vae:
                mean, log_var = model.encode(batch)
                z = model.sampler(mean,log_var)
            else:
                z = model.encode(batch)
            latent_vectors.append(z)
    
    #Extract labels for first 10 equations:
    labels = []
    emb = MathTokenEmbedding(alphabet=alphabet)
    for sample in data_loader:
        sample = sample.float()
        for idx,ele in enumerate(sample):
            eqn = emb.decode(torch.argmax(sample[idx], dim=1))
            eqn = ''.join(eqn)
            labels.append(eqn)
        break

    latent_vectors = torch.cat(latent_vectors, dim=0).cpu().numpy()

    pca = PCA(n_components=2)
    latent_vectors_2d = pca.fit_transform(latent_vectors)

    steps = 3  # Number of interpolation steps
    interpolated_vectors_2d = interpolate(latent_vectors_2d[0], latent_vectors_2d[1], steps)

    # Reverse PCA
    interpolated_vectors = pca.inverse_transform(interpolated_vectors_2d)
    interpolated_vectors_tensor = torch.Tensor(interpolated_vectors)
    # Pad each vector in the tensor individually
    padded_vectors = []
    for i, ele in enumerate(interpolated_vectors_tensor):
        current_vector = ele
        pad_size = 21 - current_vector.shape[0]
        if pad_size > 0:
            padded_vector = torch.nn.functional.pad(current_vector, (0, 0, 0, pad_size), 'constant', 0)
            padded_vectors.append(padded_vector)
        else:
            padded_vectors.append(current_vector)

    interpolated_vectors_padded = torch.stack(padded_vectors)

    interpolated_vectors_padded_tensor = torch.Tensor(interpolated_vectors_padded)
    print(interpolated_vectors_padded_tensor.shape)

    ann_latent_vectors_2d = latent_vectors_2d[:10,:]

    # Decode the interpolated points
    interpolated_labels = decode_latent_points(emb, interpolated_vectors_padded_tensor, model, vae)

    # Visualization
    #plt.scatter(latent_vectors_2d[:, 0], latent_vectors_2d[:, 1])
    for i, label in enumerate(interpolated_labels):
        plt.scatter(interpolated_vectors_2d[i, 0], interpolated_vectors_2d[i, 1], label=label)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Latent Space for Interpolated Points Visualization using PCA')
    #plt.show()
    plot_filename = "PCA_Interpolation_LatentSpace_plot.png"
    plt.savefig(f'./plots/{plot_filename}')
    plt.close()
    


def visualize_latent_space_Eqn(model,data_loader,vae=False):
    model.eval()
    # Extract latent vectors
    latent_vectors = []
    for batch in data_loader:
        with torch.no_grad():
            batch = batch.float().to(device)
            if vae:
                mean, log_var = model.encode(batch)
                z = model.sampler(mean,log_var)
            else:
                z = model.encode(batch)
            latent_vectors.append(z)
    
    #Extract labels for first 10 equations:
    labels = []
    emb = MathTokenEmbedding(alphabet=alphabet)
    for sample in data_loader:
        sample = sample.float()
        for idx,ele in enumerate(sample):
            eqn = emb.decode(torch.argmax(sample[idx], dim=1))
            eqn = ''.join(eqn)
            labels.append(eqn)
        break

    latent_vectors = torch.cat(latent_vectors, dim=0).cpu().numpy()
    print(latent_vectors.shape)
    # Dimensionality reduction using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_vectors_2d = tsne.fit_transform(latent_vectors)

    ann_latent_vectors_2d = latent_vectors_2d[:10,:]
    try:
        plt.clf()
    except:
        pass
    # Visualization
    plt.scatter(latent_vectors_2d[:, 0], latent_vectors_2d[:, 1])
    for i, label in enumerate(labels[:10]):
        plt.scatter(ann_latent_vectors_2d[i, 0], ann_latent_vectors_2d[i, 1], label=label)
        plt.text(ann_latent_vectors_2d[i, 0], ann_latent_vectors_2d[i, 1], label)

    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.title('Latent Space Visualization')
    #plt.show()
    plot_filename = "LatentSpace_plot.png"
    plt.savefig(f'./plots/{plot_filename}')
    plt.close()


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
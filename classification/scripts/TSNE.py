import os, sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

# define the plot function
def plot_2D_scatter(save_path: str,
                    df: pd.DataFrame,
                    filename: str,
                    title: str) -> None:
    """plot 2d scatter from dataframe

    Args:
        save_path (str): the path to save fig
        df (pd.DataFrame): the data
        filename (str): the filename
        title (str): the title

    Return:
        None
    """
    assert os.path.exists(save_path), "path {} does not exist".format(save_path)
    assert len(df.columns) == 3, "the dataframe should have 3 columns"
    fig, ax = plt.subplots(figsize=(10, 8)) 
    ax = sns.scatterplot(data=df, hue=df.columns[-1], x=df.columns[0], y=df.columns[1])
    ax.set_title(title)
    fig.tight_layout()
    # save the fig
    path = os.path.join(save_path, "{}.png".format(filename))
    fig.savefig(path)
    plt.close()

MODEL_NAME = "DenseNet121"
BN_TYPE = "ln"
DATASET = "cifar10"
MODEL_PATH = "NEED TO BE FILLED"

sys.path.insert(1, os.path.join(os.path.dirname(MODEL_PATH), "../src/"))
# import internal libs
from data import prepare_dataset
from model import prepare_model
from utils import set_seed

DATA_PATH = "NEED TO BE FILLED"
SAMPLE_NUM = 1000
SEED = 0

# set the seed
set_seed(SEED)

# load the dataset
_, dataset = prepare_dataset(DATASET, root = DATA_PATH)
inputs, labels = next(iter(DataLoader(dataset, batch_size=len(dataset))))

# sample a set of inputs and labels with equal number for each classies
indices = []
for i, label in enumerate(range(len(dataset.classes))):
    indices_tmp = np.where(labels == label)[0]
    np.random.seed(seed=i)
    indices.extend(np.random.choice(indices_tmp, SAMPLE_NUM // len(dataset.classes), replace=False))
sampled_inputs, sampled_labels = inputs[indices], labels[indices].cpu().numpy()

# make the model
model = prepare_model(MODEL_NAME, DATASET, BN_TYPE)
# load the pretrained model
state_dict = torch.load(MODEL_PATH)
model.load_state_dict(state_dict)

# get the intermediate features
# define the hook
features = []
def hook(module, input, output):
    assert len(input) == 1, "the input should be a tuple containing one tensor"
    features.append(input[0].cpu().detach().numpy())
if MODEL_NAME.startswith("vgg"):
    handle = model.classifier[-1].register_forward_hook(hook)
elif MODEL_NAME.startswith("ResNet"):
    handle = model.linear.register_forward_hook(hook)
elif MODEL_NAME.startswith("DenseNet"):
    handle = model.linear.register_forward_hook(hook)
else:
    raise NotImplementedError("model {} is not implemented".format(MODEL_NAME))

# foward
model.eval()
with torch.no_grad():
    model(sampled_inputs)
features_df = pd.DataFrame(features[0])

# now t-SNE
tsne = TSNE(n_components=2)
features_tsne = tsne.fit_transform(features_df)
data_tsne = np.vstack((features_tsne.T, sampled_labels)).T
tsne_df = pd.DataFrame(data_tsne, columns=['Dim1', 'Dim2', 'class'])
tsne_df['class'] = tsne_df['class'].astype(str)

# make save_path
save_path = os.path.join(os.path.dirname(MODEL_PATH), "TSNE", f"{os.path.basename(MODEL_PATH).split('.')[0]}")
if not os.path.exists(save_path):
    os.makedirs(save_path)

# save the tsne_df
tsne_df.to_csv(os.path.join(save_path, "tsne_df.csv"), index=False)

# plot tsne
plot_2D_scatter(save_path = save_path,
                df = tsne_df,
                filename = "tsne_features_testset",
                title = "t-SNE on features of testset")
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px

def log_attention_heatmap(attn_weights, global_step, layer=0, head=0):
    attn = attn_weights[layer][0, head].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(attn, cmap="viridis", cbar=True, ax=ax)
    ax.set_title(f"Attention Map - Layer {layer} Head {head}")
    wandb.log({f"Attention/Layer{layer}_Head{head}": wandb.Image(fig)}, step=global_step)
    plt.close(fig)

def log_embeddings_plotly(model, vocab, num_tokens=100, step=0):
    emb = model.embeddings.weight.detach().cpu().numpy()
    pca = PCA(n_components=2).fit_transform(emb[:num_tokens])
    df = pd.DataFrame(pca, columns=["x", "y"])
    df["token"] = vocab[:num_tokens]

    fig = px.scatter(df, x="x", y="y", text="token", title="Embedding PCA")
    fig.update_traces(textposition='top center')
    wandb.log({"Embeddings/PCA_plotly": wandb.Plotly(fig)}, step=step)

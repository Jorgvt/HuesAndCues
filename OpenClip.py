from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModelWithProjection
import open_clip

from tqdm.auto import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch

from huesandcues.utils import create_color_images

# Load the model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Load the words
df_words = pd.read_excel("Words_Hues&Clues_Eng.xlsx", index_col=0)

# Generate the color images
color_images = create_color_images('HC_RGB.csv')

results = defaultdict(list)
show_colors = True

# Iterate over all the words
it = 0
for word in tqdm(df_words.palabra.to_list()):
# for word in ["EGG", "LEMON"]:
    # Process the images and text
    inputs = processor(text=[word], images=color_images, return_tensors="pt", padding=True)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    text_embeds = outputs.text_embeds
    image_embeds = outputs.image_embeds

    # Normalize embeddings
    text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
    image_embeds /= image_embeds.norm(dim=-1, keepdim=True)

    # Calculate cosine similarity
    cosine_sim = (text_embeds @ image_embeds.T).squeeze(0)

    # Get top 5 colors
    top5_indices = cosine_sim.argsort(descending=True)[:5]
    top5_distances = cosine_sim.sort(descending=True)[0][:5]

    # print(f"Top 5 colors for the word '{word}':")

    if show_colors: fig, axes = plt.subplots(1,5)
    for j, i in enumerate(top5_indices):
        color_info = df_colors.iloc[i.item()]
        color_info["distance"] = top5_distances[j].item()
        results[word].append(color_info.to_dict())
        # print(f"- Coords: ({color_info['coordenada_x']}, {color_info['coordenada_y']}), RGB: ({color_info['R']}, {color_info['G']}, {color_info['B']})")
        if show_colors:
            axes[j].imshow(color_images[i.item()])
            axes[j].axis("off")
            axes[j].set_title(f"({color_info['coordenada_x']}, {color_info['coordenada_y']})")
    if show_colors:
        plt.suptitle(word)
        plt.savefig(f"{word}.png", dpi=300)
        plt.close()
        # plt.show()
    it += 1
    # if it == 5: break


flat_data = [
    {**record, 'word':word} 
    for word, records in results.items() 
    for record in records
]

# 2. Create the DataFrame
df = pd.DataFrame(flat_data)

# Optional: Reorder columns to put 'Animal' first
cols = ['word'] + [c for c in df.columns if c != 'word']
df = df[cols]

df.head()

df.to_csv("results_handc.csv", index=0)


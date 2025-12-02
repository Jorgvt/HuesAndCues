#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModelWithProjection

model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")


outputs = model(**inputs, output_hidden_states=True)


outputs['image_embeds'].shape


outputs['last_hidden_state'].shape


len(outputs['hidden_states']), outputs['hidden_states'][-1].shape


from transformers import AutoProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image


inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)


outputs = model(**inputs, output_hidden_states=True)


outputs.keys()


outputs['text_embeds'].shape, outputs['image_embeds'].shape


outputs['logits_per_image']


outputs['logits_per_image'].softmax(dim=1)


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image1 = Image.open(requests.get(url, stream=True).raw)
image1


url = "https://www.infobae.com/resizer/v2/https%3A%2F%2Fs3.amazonaws.com%2Farc-wordpress-client-uploads%2Finfobae-wp%2Fwp-content%2Fuploads%2F2017%2F04%2F06155038%2Fperro-beso.jpg?auth=7db092219938909c16f466d602dcf2715cb44547bae1b45714fbfc66be4b16e9&smart=true&width=1200&height=900&quality=85"
image2 = Image.open(requests.get(url, stream=True).raw)
image2


from glob import glob
from matplotlib import pyplot as plt
import cv2
from natsort import natsorted
import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import manhattan_distances
from utils import rmse

images = glob('./imagenes/*.png')
images = natsorted(images)

colores = []
rs = []
gs = []
bs = []
paths = []

for i in images:
  img = Image.open(i)

  i = i.replace(" ", "")
  path = i.split("_")[1]
  paths.append(path)

  img = np.array(img) 
  r = img[0,0,:][0]
  g = img[0,0,:][1]
  b = img[0,0,:][2]
  rs.append(r)
  gs.append(g)
  bs.append(b)
  colores.append(img)



d = {'Path': paths, 'R': rs,'G': gs,'B': bs}
df = pd.DataFrame(data=d)


df.to_csv("HC_RGB.csv")


m = np.ones((480,3))

for j in range(len(colores)):
  col = colores[j][0,0,:]
  m[j] = col


m.resize((16,30,3))


m.shape


plt.imshow(m/255)
plt.axis("off")
plt.savefig("HCPropio.png")
plt.show()


[0,2,12,13,14,16]


textos_mayus = ["LEMON",
"LIME",
"BASIL",
"SEA",
"PUMPKIN",
"BARROW",
"CUCUMBER",
"SALMON",
"AUBERGINE",
"POND",
"GRASS",
"AMATIST",
"SAPPHIRE",
"LAVENDER",
"BLOOD",
"APPLE",
"EGG",
"BANANA",
"TOMATO",
"WHALE",
"BLUSH",
"SAND",
"KIWI",
"SUNFLOWER",
"HAIR",
"SKIN",
"SHAME",
"PEACE",
"RAGE",
"DISGUST",
"BARBIE",
"DANGER",
"FEMINISM",
"TRACTOR"]


len(textos_mayus)


inputs_multiple = processor(text=textos_mayus, images=colores, return_tensors="pt", padding=True)


outputs = model(**inputs_multiple)


outputs.logits_per_text.shape


out = outputs.logits_per_text
for fila in out:
    print(len(fila.detach().numpy()))
    break


out = outputs.logits_per_text
respuestas = []
for fila in out:
    respuesta = (np.argsort(fila.detach().numpy()))
    respuesta = respuesta[-5:]
    respuestas.append(respuesta)




p = [0,2,12,13,14,16]


respuestas[16]


out = outputs.logits_per_text.softmax(dim=1)
probabilidades = []
for fila in out:
    prob = (np.sort(fila.detach().numpy()))
    prob = prob[-5:]
    probabilidades.append(prob)


probabilidades = pd.DataFrame(probabilidades)


probabilidades.to_csv("Probs_Mod_All.csv")


probabilidades.iloc[p,:].to_csv("Probs_Mod.csv")


fila.detach().numpy()[respuesta]


len(respuestas)


respuestas = np.array(respuestas) + 1


respuestas_m = pd.DataFrame(respuestas)
respuestas_m.to_csv("Resp_Mod_All.csv")


letras = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

num2letra = {}
letra2num = {}
for i,letra in enumerate(letras):
    num2letra[i] = letra
    letra2num[letra] = i


datos = pd.read_csv("HandC.csv", sep = ";")


humanos = datos[["PAULA", "MARÍA", "NURIA", "JORGE", "PABLO", "JOSEP"]]

medias = []
stds = []

paulas = []
marias = []
nurias = []
jorges = []
pablos = []
joseps = []

for row in humanos.iterrows():
    row = row[1]
    d = int(re.findall("\d+", row["PAULA"])[0])
    n = letra2num[row["PAULA"][0]] 
    paula = [n+1,d]
    paulas.append(paula)

    d = int(re.findall("\d+", row["MARÍA"])[0])
    n = letra2num[row["MARÍA"][0]] 
    maria = [n+1,d]
    marias.append(maria)

    d = int(re.findall("\d+", row["NURIA"])[0])
    n = letra2num[row["NURIA"][0]] 
    nuria = [n+1,d]
    nurias.append(nuria)

    d = int(re.findall("\d+", row["JORGE"])[0])
    n = letra2num[row["JORGE"][0]] 
    jorge = [n+1,d]
    jorges.append(jorge)

    d = int(re.findall("\d+", row["PABLO"])[0])
    n = letra2num[row["PABLO"][0]]
    pablo = [n+1,d]
    pablos.append(pablo)

    d = int(re.findall("\d+", row["JOSEP"])[0])
    n = letra2num[row["JOSEP"][0]] 
    josep = [n+1,d]
    joseps.append(josep)

    media = np.round(np.mean([paula, maria, nuria, jorge, pablo, josep], axis = 0),0)
    # std = np.round(np.std([paula, maria, nuria, jorge, pablo, josep], axis = 0),0)

    medias.append(media)
    # stds.append(std)




datos["Paula_casillas"] = paulas
datos["Maria_casillas"] = marias
datos["Nuria_casillas"] = nurias
datos["Jorge_casillas"] = jorges
datos["Pablo_casillas"] = pablos
datos["Josep_casillas"] = joseps


datos.head(10)


covarianzas = []
for i,row in datos.iterrows():
    m_cov = np.cov(np.array(row.filter(regex="\w+_casillas").to_list()), rowvar=False)
    covarianzas.append(m_cov)


casillas = []
for m in medias:
    l = num2letra[m[0]]
    casilla = str(l) + str(m[1])
    casillas.append(casilla)


resp_modelo = [["K4", "K6", "K5", "J6", "J7"], #Limon
["P9", "P11", "P12", "P13", "P14"], #Albahaca
["I28", "H30", "H29", "G30", "G29"], #Zafiro
["G20", "G21", "G23", "G22", "H22"], #Lavanda
["A5", "A8", "A6", "A7", "B6" ], #Sangre
["M20", "L17", "L18", "L19", "H20"]] #Huevo


resp_modelo = pd.DataFrame(resp_modelo)


resp_modelo.to_csv("Resp_Mod.csv")


medias = []
stds = []

for row,p in zip(resp_modelo, [0,2,12,13,14,16]):
    raw = row[0]
    d = int(re.findall("\d+", raw)[0]) 
    n = letra2num[raw[0]] + 1
    modelo1 = [n,d]

    raw = row[1]
    d = int(re.findall("\d+", raw)[0]) 
    n = letra2num[raw[0]] + 1
    modelo2 = [n,d]

    raw = row[2]
    d = int(re.findall("\d+", raw)[0]) 
    n = letra2num[raw[0]] + 1
    modelo3 = [n,d]

    raw = row[3]
    d = int(re.findall("\d+", raw)[0]) 
    n = letra2num[raw[0]] + 1
    modelo4 = [n,d]

    raw = row[4]
    d = int(re.findall("\d+", raw)[0]) 
    n = letra2num[raw[0]] + 1
    modelo5 = [n,d]

    media = np.round((np.sum([np.array(modelo1)*probabilidades[p][4],np.array(modelo2)*probabilidades[p][3],np.array(modelo3)*probabilidades[p][2],
                              np.array(modelo4)*probabilidades[p][1],np.array(modelo5)*probabilidades[p][0]], axis = 0))/(np.sum(probabilidades[p])),0)
    # media = np.mean([modelo1,modelo2,modelo3, modelo4,modelo5], axis = 0)
    # std = np.round(np.std([paula, maria, nuria, jorge, pablo, josep], axis = 0),0)

    medias.append(media)

    # stds.append(std)


medias


valores_tri = pd.read_csv("ValoresHandC.csv", sep = ";")


valores_tri


a = valores_tri[valores_tri["C"]=="A1"][["x", "y"]]
np.array(a)





# covarianzas = []
# lista = ["PAULA", "Maria", "Nuria", "Jorge", "Pablo", "Josep"]

# for i,row in datos.iterrows():
#     cass = []
#     for name in lista:
#         c = row[name] 
#         valores_tri[valores_tri["C"]==c][["x", "y"]]
#     #     cass.append(c)
#     # m_cov = np.cov(np.array(cass), rowvar=False)
#     # covarianzas.append(m_cov)


df_words.head()


from tqdm.auto import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch

# Load the colors
df_colors = pd.read_csv('HC_RGB.csv')

# Load the words
df_words = pd.read_excel("Words_Hues&Clues_Eng.xlsx", index_col=0)

# Create PIL images from RGB values
color_images = []
for index, row in df_colors.iterrows():
    rgb = (int(row['R']), int(row['G']), int(row['B']))
    img = Image.new('RGB', (256, 256), color=rgb)
    color_images.append(img)


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


import os
from tqdm.auto import tqdm
from glob import glob
import re
from collections import defaultdict
from glob import glob
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import cv2
from natsort import natsorted
import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import manhattan_distances
import scipy
import numpy as np
from sklearn import datasets
from scipy.stats import f

from perceptualtests.color_matrices import Mng2xyz

words = pd.read_excel("Words_Hues&Clues_Eng.xlsx")
words = words.rename({"palabra": "word"}, axis=1)

palabras = pd.read_excel("Palabras_Hues&Clues.xlsx")

words_palabras = pd.concat([words, palabras], axis=1)
words_palabras = words_palabras[["palabra", "word"]]


humanos = pd.read_csv("HC.csv")


def translate(palabra):
    if palabra in words_palabras.palabra.to_list():
        palabra = words_palabras[words_palabras.palabra == palabra].word.item()
    return palabra


humanos["word"] = humanos["word"].apply(translate)
humanos = humanos.sort_values("word")

respuesta = []
contador = defaultdict(lambda: 0)
for i, row in humanos.iterrows():
    contador[row.word] += 1
    respuesta.append(contador[row.word])
    # break

humanos["respuesta"] = respuesta


humanos = humanos.pivot(index="word", columns=["respuesta"], values=["coordinate"])
humanos.columns = range(1,19)
humanos = humanos.reset_index()

rgb = pd.read_csv("HC_RGB.csv")

def get_xyz(row):
    rgb = row[["R", "G", "B"]].to_numpy()
    rgb = rgb[None,:]
    ng = rgb**(2)
    xyz = ng @ Mng2xyz.T
    xyz = xyz/xyz.sum()
    xyz = xyz[0]
    return {"x": xyz[0], "y": xyz[1], "z": xyz[2]}

valores_tri = pd.concat([rgb, rgb.apply(get_xyz, axis=1, result_type="expand")], axis=1)
valores_tri["C"] = valores_tri.apply(lambda row: f"{row.coordenada_x}{row.coordenada_y}", axis=1)
valores_tri.head()

def casilla2cord(x): 
    if x is not np.nan:
        return valores_tri[valores_tri.C == x][["x", "y"]].to_numpy()[0]
    else: return x

# humanos = humanos["PAULA", "MARÍA", "NURIA", "JORGE", "PABLO", "JOSEP"]
# palabras_id = [0, 2, 12, 13, 14, 16]
humanos_tri = humanos.iloc[:,1:].applymap(casilla2cord)

means, covs = [], []
for i, r in humanos_tri.iterrows():
    r = np.array([_ for _ in r[~pd.isna(r)]])
    mean = r.mean(axis=0)
    cov = np.cov(r, rowvar=False)
    means.append(mean)
    covs.append(cov)
    # break

def confidence_ellipse(mean, cov, ax, facecolor='none', n_stds=1, **kwargs):
    eigval, eigvec = np.linalg.eig(cov)
    theta = np.arccos(eigvec[0,0])*180/np.pi
    width = eigval[0]**(1/2)*n_stds
    height = eigval[1]**(1/2)*n_stds
    ellipse = Ellipse(mean, width=width, height=height, angle=theta,
                      facecolor=facecolor, **kwargs)

    return ax.add_patch(ellipse)

means = np.stack(means)
covs = np.stack(covs)

for model_path in tqdm(glob("Results/*/results.csv")):
    # print(model_path)
    model = pd.read_csv(model_path)
    model = model.sort_values("word")
    model["C"] = model.apply(lambda row: f"{row.coordenada_x}{row.coordenada_y}", axis=1)
    model["distance"] = model.distance.abs()

    model_prob = model.distance.to_numpy().reshape(100,5)

    respuesta = []
    contador = defaultdict(lambda: 0)
    for i, row in model.iterrows():
        contador[row.word] += 1
        respuesta.append(contador[row.word])
        # break


    model["respuesta"] = respuesta

    model = model.pivot(index="word", columns=["respuesta"], values=["C"])
    model.columns = range(1,6)
    model = model.reset_index()

    # humanos = humanos["PAULA", "MARÍA", "NURIA", "JORGE", "PABLO", "JOSEP"]
    # palabras_id = [0, 2, 12, 13, 14, 16]
    model_tri = model.iloc[:,1:].applymap(casilla2cord)

    means_m, covs_m = [], []
    for (i, r), p in zip(model_tri.iterrows(), model_prob):
        r = np.array([_ for _ in r])
        mean = np.average(r, axis=0, weights=p)
        cov = np.cov(r, rowvar=False, aweights=p)
        means_m.append(mean)
        covs_m.append(cov)
        # break

    def TwoSampleT2Test(X, Y, weights_Y=None):
        nx, p = X.shape
        ny, _ = Y.shape
        if weights_Y is None:
            weights_Y = np.ones(shape=(ny))

        delta = np.mean(X, axis=0) - np.average(Y, axis=0, weights=weights_Y)
        Sx = np.cov(X, rowvar=False)
        Sy = np.cov(Y, rowvar=False, aweights=weights_Y)

        S_pooled = ((nx-1)*Sx + (ny-1)*Sy)/(nx+ny-2)
        t_squared = (nx*ny)/(nx+ny) * np.matmul(np.matmul(delta.transpose(), np.linalg.inv(S_pooled)), delta)
        statistic = t_squared * (nx+ny-p-1)/(p*(nx+ny-2))
        F = f(p, nx+ny-p-1)
        p_value = 1 - F.cdf(statistic)
        # print(f"Test statistic: {statistic}\nDegrees of freedom: {p} and {nx+ny-p-1}\np-value: {p_value}")
        return statistic, p_value

    tests = []
    pvs = []
    for (i, m), p, (j, h) in zip(model_tri.iterrows(), model_prob, humanos_tri.iterrows()):
        h = h[~pd.isna(h)]
        h = np.array([_ for _ in h])
        m = np.array([_ for _ in m])
        test, pv = TwoSampleT2Test(h, m, weights_Y=p)
        tests.append(test)
        pvs.append(pv)

    d = {'Words': humanos.word.unique(), "Stat" : tests, "p-value" : pvs}
    a = pd.DataFrame(d)
    results_path = os.path.join("/".join(model_path.split("/")[:-1]), "p_values.csv")
    a.to_csv(results_path)

    # p_lim = 0.003
    # a[a["p-value"]<p_lim]
    # a[a["p-value"]>p_lim]
    # break


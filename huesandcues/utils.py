import pandas as pd
from PIL import Image

def create_color_images(path, return_df=False) -> list:
    # Load the colors
    df_colors = pd.read_csv(path)

    # Create PIL images from RGB values
    color_images = []
    for index, row in df_colors.iterrows():
        rgb = (int(row['R']), int(row['G']), int(row['B']))
        img = Image.new('RGB', (256, 256), color=rgb)
        color_images.append(img)
    if return_df:
        return color_images, df_colors
    else:
        return color_images

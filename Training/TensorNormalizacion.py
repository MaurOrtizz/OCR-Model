import os
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Paths
csv_path = "Data/ImagenTexto_Validado.csv"
image_folder = "Data/Anotaciones"
output_folder = "Data/Procesadas"
os.makedirs(output_folder, exist_ok=True)

# Transformaciones que aplicarás de forma permanente
transform = transforms.Compose([
    transforms.ToTensor(),  # Convierte a tensor [0,1]
    transforms.Normalize((0.5,), (0.5,))  # Normaliza a [-1,1]
])

# Leer CSV
df = pd.read_csv(csv_path)

# Procesar imágenes
for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_path = os.path.join(image_folder, row["Direccion"])
    try:
        img = Image.open(img_path)  # blanco y negro
        img = transform(img)
        save_path = os.path.join(output_folder, row["Direccion"].replace(".jpg", ".pt"))
        torch.save(img, save_path)
    except Exception as e:
        print(f"Error en {img_path}: {e}")


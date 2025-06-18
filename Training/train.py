import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os
#import pandas as pd
from model import OCRDataset, CRNN, custom_collate_fn
from torchvision import transforms
from config import char2idx, idx2char
import torch.nn as nn


# --- Configuración de entrenamiento ---
def main():
    # 1. Inicialización (mover aquí tus imports y configuraciones)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Carga de datos
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = OCRDataset(
        csv_file="../Data/ImagenTexto_Validado.csv",
        image_folder="../Data/Anotaciones",
        transform=transform,
        char2idx=char2idx,
        idx2char=idx2char
    )
    
    # 3. DataLoader optimizado
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,  # Usar 4-8 según núcleos CPU
        pin_memory=True,
        persistent_workers=True,
        collate_fn=custom_collate_fn
    )
    
    # 4. Modelo y optimizador
    model = CRNN(img_height=32, num_channels=1, num_classes=len(char2idx)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 5. Bucle de entrenamiento (igual al que tienes)
    # --- 1. Funciones para checkpoints (AGREGAR ESTO AL INICIO) ---
    def save_checkpoint(epoch, model, optimizer, loss, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path)

    def load_checkpoint(model, optimizer, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']
    
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=custom_collate_fn, num_workers=2, pin_memory=True)

    # --- 2. Configuración inicial (MODIFICAR ESTA PARTE) ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN(img_height=32, num_channels=1, num_classes=len(char2idx)).to(device)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Crear carpeta para checkpoints
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # --- 3. Cargar checkpoint previo si existe (NUEVO) ---
    start_epoch = 0
    checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Reanudando entrenamiento desde epoch {start_epoch + 1}")
    
    # --- 4. Bucle de entrenamiento (MODIFICADO) ---
    num_epochs = 10
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
    
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        
        for images, labels in loop:
            images = images.to(device)
            label_lengths = torch.tensor([len(t) for t in labels], dtype=torch.long)
            targets = torch.cat([t for t in labels]).to(device)
    
            outputs = model(images)
            log_probs = outputs.log_softmax(2)
            input_lengths = torch.full(
                size=(images.size(0),), 
                fill_value=outputs.size(0),
                dtype=torch.long
            ).to(device)
    
            loss = ctc_loss(log_probs, targets, input_lengths, label_lengths)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
    
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
    
        # --- 5. Guardar checkpoint (NUEVO) ---
        save_checkpoint(
            epoch + 1,  # Guardamos el siguiente epoch a entrenar
            model,
            optimizer,
            avg_loss,
            checkpoint_path  # Sobrescribe el último checkpoint
        )
        # Opcional: Guardar también un checkpoint por epoch
        epoch_checkpoint = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
        save_checkpoint(epoch + 1, model, optimizer, avg_loss, epoch_checkpoint)

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()
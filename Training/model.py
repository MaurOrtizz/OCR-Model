import torch
from PIL import Image
import pandas as pd
import os
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_height, num_channels, num_classes, rnn_hidden_size=256):
        super(CRNN, self).__init__()

        # Feature extractor (CNN backbone)
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, 1, 1),  # output: (64, H, W)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                   # output: (64, H/2, W/2)

            nn.Conv2d(64, 128, 3, 1, 1),          # output: (128, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                   # output: (128, H/4, W/4)

            nn.Conv2d(128, 256, 3, 1, 1),         # output: (256, H/4, W/4)
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, 1, 1),         # output: (256, H/4, W/4)
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),         # output: (256, H/8, W/4)

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),         # output: (512, H/16, W/4)

            nn.Conv2d(512, 512, 2, 1, 0),         # output: (512, H/16 -1, W/4 -1)
            nn.ReLU()
        )

        self.lstm1 = nn.LSTM(512, rnn_hidden_size, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(2 * rnn_hidden_size, rnn_hidden_size, bidirectional=True, batch_first=True)

        # Final classifier
        self.fc = nn.Linear(2 * rnn_hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, channels, height, width)
        conv_out = self.cnn(x)  # shape: (B, C, H, W)
        b, c, h, w = conv_out.size()

        assert h == 1, f"Height must be 1 after CNN, got {h}"

        conv_out = conv_out.squeeze(2)  # remove height dim -> (B, C, W)
        conv_out = conv_out.permute(0, 2, 1)  # (B, W, C)

        # RNN modificado
        lstm_out, _ = self.lstm1(conv_out)
        lstm_out, _ = self.lstm2(lstm_out)
        
        # Clasificación
        out = self.fc(lstm_out)  # (B, W, num_classes)

        return out.permute(1, 0, 2)  # (W, B, num_classes) for CTC loss

class OCRDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, image_folder, transform=None, char2idx=None, idx2char=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform
        self.char2idx = char2idx
        self.idx2char = idx2char

    def __len__(self):
        return len(self.data)

    def encode_label(self, text):        
        # Limpieza básica: remueve caracteres no imprimibles
        cleaned_text = "".join(char for char in str(text) if char.isprintable())
        # Mapea cada carácter, usa <unk> si no existe
        return [self.char2idx.get(char, self.char2idx['<unk>']) for char in cleaned_text]

    def decode_label(self, indices):
        return ''.join([self.idx2char[i] for i in indices if i in self.idx2char])

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        #img_path = os.path.join(self.image_folder, row['Direccion'])
    
        label = str(row['Texto']).strip()
        if not label:
            label = "<unk>"
    
        try:
            pt_path = os.path.join(self.image_folder, row['Direccion'].replace('.jpg', '.pt'))
            image = torch.load(pt_path)  
    
            encoded = torch.tensor(self.encode_label(label), dtype=torch.long)
            if len(encoded) == 0:
                encoded = torch.tensor([self.char2idx['<unk>']], dtype=torch.long)
    
            return image, encoded
        except Exception as e:
            # Opcional: imprimir error para debug
            print(f"Error al cargar imagen {pt_path}: {e}")
            # Devuelve imagen vacía + label <unk>
            dummy_image = torch.zeros((1, 32, 100))  # Tamaño acorde a tu modelo
            dummy_label = torch.tensor([self.char2idx['<unk>']], dtype=torch.long)
            return dummy_image, dummy_label

def custom_collate_fn(batch):
    images, labels = zip(*batch)  # images: tuple of tensors; labels: tuple of 1D tensors
    images = torch.stack(images, dim=0)
    return images, labels
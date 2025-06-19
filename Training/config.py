#import string
from collections import Counter
import pandas as pd
import os

df = pd.read_csv("Data/ImagenTexto_Validado.csv")
all_text = "".join(df['Texto'].dropna().astype(str))
char_counts = Counter(all_text)

all_chars = sorted(char_counts.keys())
char2idx = {char: idx + 2 for idx, char in enumerate(all_chars)} 
char2idx['<blank>'] = 0
char2idx['<unk>'] = 1
idx2char = {idx: char for char, idx in char2idx.items()}
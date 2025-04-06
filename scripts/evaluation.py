from torch.utils.data import DataLoader
from tqdm import tqdm
import torch 
from sklearn.metrics import classification_report

def evaluate_model(model, tokenizer, val_loader, device=torch.device('cpu')):
  num_batches = len(val_loader)

  prediction = []
  all_labels = []

  for i in tqdm(range(num_batches)):
    sentences, labels = next(iter(val_loader))
    encodings = tokenizer(sentences, padding=True, truncation=True, max_length=24, return_tensors="pt")

    with torch.no_grad():
      logits = model(**encodings.to(device)).logits

    prediction += logits.argmax(-1).cpu()
    all_labels += labels.cpu()

  print(classification_report(all_labels, prediction, target_names=['formal', 'informal'])) 
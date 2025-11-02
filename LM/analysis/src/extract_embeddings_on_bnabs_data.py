import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import numpy as np
import argparse
import os
from torch.utils.data import DataLoader

# Function to ensure that the directory exists
def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to extract the label (like "label_test") from the embeddings file path
def extract_label_from_path(file_path):
    label = (file_path).split('/')[-1]
    return label


# Function to handle batching correctly
def collate_fn(batch):
    # Simply stack tensors as they're already in tensor format
    input_ids = torch.tensor([item['input_ids'] for item in batch])  # Stack input_ids
    attention_mask = torch.tensor([item['attention_mask'] for item in batch])  # Stack attention masks
    
    # If you have labels in your dataset, stack them too:
    if 'label' in batch[0]:  # Assuming 'label' is in the dataset
        labels = torch.tensor([label_map[(item['label'])] for item in batch])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    
    return {'input_ids': input_ids, 'attention_mask': attention_mask}



# Function to extract embeddings from the CLS token in RoBERTa
def extract_embeddings(model, dataloader, device): # Set model to evaluation mode
    embeddings = []
    labels = []
    
    with torch.no_grad():  # Disable gradient computation
        for batch in dataloader:
            inputs = batch['input_ids']
            #print((batch['input_ids']))  # Should be torch.Tensor, not list
            #print(batch['input_ids'].shape)  # Should have the shape (batch_size, seq_len)
            attention_mask = batch['attention_mask']
            label = batch['labels']
            
            # Get model outputs
            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            
            # Extract the CLS token embeddings (first token in the sequence)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token is at index 0
            
            # Append CLS embeddings and labels to the list
            embeddings.append(cls_embeddings.cpu().numpy())
            labels.append(label.cpu().numpy())
    
    # Convert the lists to numpy arrays
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    return embeddings, labels

if __name__ == "__main__":
    # Parse arguments from the command line
    parser = argparse.ArgumentParser(description='Extract RoBERTa CLS token embeddings from a test set')
    parser.add_argument('--test_file', type=str, required=True, help='Path to the test dataset (e.g., .csv)')
    parser.add_argument('--save_dir', type=str, default='embeddings', help='Directory to save the extracted embeddings and labels')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing the data')
    parser.add_argument('--model_name_or_path', type=str, help='Pre-trained RoBERTa model')
    parser.add_argument('--tokenizer_name_or_path', type=str, help='Pre-trained  tokenizer')
    parser.add_argument('--binary_labels', type=bool, help='')
    parser.add_argument('--binary_labels_broad', type=bool, help='')

    args = parser.parse_args()

    # Ensure the save directory exists
    create_folder(args.save_dir)
    
    # from str label to int
    if args.binary_labels == True:
        label_map = {'BRO': 0, 'NON': 1, 'UNK': 2}
        label_list = ["BRO","NON",'UNK']
        unique_label = ["BRO","NON"]
        labels_categories = [0,1]

    elif args.binary_labels_broad == True:
        label_map = {'HIV': 0, 'NON': 1, 'UNK': 2}
        label_list = ["HIV","NON",'UNK']
        unique_label = ["HIV","NON"]
        labels_categories = [0,1]

    else:
        label_map = {'BRO': 0, 'NON': 1, 'NEU': 2, 'UNK': 3} #define the label map to keep it constant
        label_list = ["BRO","NON","NEU",'UNK']
        unique_label = ["BRO","NON","NEU"]
        labels_categories = [0,1,2]

    # Load the tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path)

    # Move the model to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Load the dataset (assumes a text column and label column, adjust if needed)
    dataset = load_dataset('csv', data_files={'test': args.test_file})['test']
    

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], 
                                  padding='max_length', 
                                  truncation=True, 
                                  max_length=30, 
                                  return_tensors='pt')
    # Apply tokenization
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns = ['text','id'])

    # Create a DataLoader
    test_dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size,collate_fn=collate_fn)

    # Extract embeddings and labels using the CLS token
    embeddings, labels = extract_embeddings(model, test_dataloader, device)
    
    # Extract the label identifier from the test file path
    label_identifier = extract_label_from_path(args.test_file)

    # Save embeddings and labels
    embeddings_path = os.path.join(args.save_dir, f'{label_identifier}_embeddings_cls.npz')

    # Save embeddings and labels as .npz and file
    np.savez(embeddings_path,  embeddings=embeddings,  labels=labels,)

    print(f"Embeddings and labels saved to {embeddings_path}")

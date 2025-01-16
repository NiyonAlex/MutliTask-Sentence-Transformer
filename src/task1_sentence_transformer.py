import torch
from transformers import BertModel, BertTokenizer
import torch.nn as nn


class SentenceTransformerModel(torch.nn.Module):
    def __init__(self, model_name='bert-base-uncased', fixed_length=256):
        """
        Initializes the SentenceTransformerModel using a pre-trained BERT model.
        
        Args:
            model_name (str): The name of the pre-trained model to use (default is 'bert-base-uncased').
            fixed_length (int): The fixed length for sentence embeddings.
        """
        super(SentenceTransformerModel, self).__init__()
        
        # Load a pre-trained BERT model (transformer backbone) from HuggingFace
        self.bert = BertModel.from_pretrained(model_name)
        
        # Choice: Use mean pooling to get fixed-length sentence embeddings.
        self.pooling = 'mean'
        self.fixed_length = fixed_length

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the model to obtain sentence embeddings.
        
        Args:
            input_ids (torch.Tensor): Tokenized input sentences.
            attention_mask (torch.Tensor): Attention mask indicating where the padding is in the input sequence.
        
        Returns:
            torch.Tensor: Sentence embeddings of fixed length.
        """
        # Pass the tokenized sentences through the BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the last hidden states (token embeddings) from the output of BERT
        last_hidden_state = outputs.last_hidden_state
        
        # Apply the chosen pooling method to get fixed-length embeddings from variable-length token sequences
        if self.pooling == 'mean':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            sum_mask = input_mask_expanded.sum(dim=1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask
        else:
            sentence_embeddings = last_hidden_state[:, 0, :]
        
        # Ensure fixed-length embeddings
        if sentence_embeddings.size(1) > self.fixed_length:
            # Truncate embeddings to the desired length
            sentence_embeddings = sentence_embeddings[:, :self.fixed_length]
        elif sentence_embeddings.size(1) < self.fixed_length:
            # Pad embeddings to the desired length
            padding = torch.zeros(sentence_embeddings.size(0), self.fixed_length - sentence_embeddings.size(1))
            sentence_embeddings = torch.cat((sentence_embeddings, padding), dim=1)

        return sentence_embeddings


# Sample test with a few sentences
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = SentenceTransformerModel(fixed_length=256)

    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Transformer-based model change the world quickly.",
        "I love programming in Python.",
        "I love to work with machine learning stuffs.",
        "perhaps no picture ever made has more literally showed that the road to hell is paved with good intentions",
        "The product exceeded my expectations, and I will definitely buy again.",
        "Amazing customer service! The staff were friendly and helpful.",
        "I love the design and performance of this phone, highly recommend it!",
        "The quality of the product was terrible and broke after one use.",
        "Customer service was rude, and they didn’t resolve my issue.",
        "Very disappointed with the experience, I won’t be coming back.",
        "The new iPhone features a faster processor and improved camera.",
        "Artificial Intelligence is transforming industries across the globe.",
        "The football team won their fifth championship title this season.",
        "The swimmer broke the world record in the 100-meter freestyle.",
        "A balanced diet and regular exercise are key to maintaining good health.",
        "Doctors recommend getting at least 7 hours of sleep per night.",
        "The pandemic has raised awareness about the importance of hygiene.",
    ]

    # Tokenize the sentences: Convert sentences into token IDs for BERT input.
    encoded_input = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=128  # Fixed-length size for tokenization
    )

    # Pass tokenized sentences through the model to obtain embeddings (disable gradient computation)
    with torch.no_grad():
        embeddings = model(
            input_ids=encoded_input['input_ids'],
            attention_mask=encoded_input['attention_mask']
        )

    # Print the sentence embeddings and their corresponding sentence
    print("Sentence Embeddings:")
    for idx, embedding in enumerate(embeddings):
        print(f"Sentence {idx+1}: {sentences[idx]}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding vector:\n{embedding}\n")

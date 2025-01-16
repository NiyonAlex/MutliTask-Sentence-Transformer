# Import necessary libraries
import torch
from transformers import BertModel, BertTokenizer
from task1_sentence_transformer import SentenceTransformerModel  # type: ignore

class MultiTaskSentenceTransformerModel(SentenceTransformerModel):
    """
    A model that extends the basic SentenceTransformerModel to handle multi-task learning.
    This model performs two tasks:
    1. Sentence Classification (Task A)
    2. Sentiment Analysis (Task B)
    """
    def __init__(self, model_name='bert-base-uncased', num_classes_task_a=3, num_classes_task_b=2):
        """
        Initializes the multi-task sentence transformer model by extending the base SentenceTransformerModel.
        
        Args:
            model_name (str): The pre-trained BERT model to use (default is 'bert-base-uncased').
            num_classes_task_a (int): Number of classes for sentence classification task A (default is 3).
            num_classes_task_b (int): Number of classes for sentiment analysis task B (default is 2).
        """
        super(MultiTaskSentenceTransformerModel, self).__init__()  # Call the parent class constructor
        # Load a pre-trained BERT model from HuggingFace (transformer backbone)
        self.bert = BertModel.from_pretrained(model_name)
        
        # Set the pooling method to 'mean', which means we'll average token embeddings for the sentence
        self.pooling = 'mean'
        
        # Task A: Sentence Classification
        # A linear layer that maps sentence embeddings to class scores for Task A
        # Example: Task A could classify sentences into categories like 'News', 'Opinion', 'Entertainment'
        self.classifier_task_a = torch.nn.Linear(self.bert.config.hidden_size, num_classes_task_a)
        
        # Task B: Sentiment Analysis
        # A linear layer that maps sentence embeddings to class scores for Task B
        # Example: Task B could classify the sentiment of the sentence (e.g., 'Positive', 'Negative')
        self.classifier_task_b = torch.nn.Linear(self.bert.config.hidden_size, num_classes_task_b)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the model to get predictions for both tasks.
        
        Args:
            input_ids (torch.Tensor): Tokenized input sentences.
            attention_mask (torch.Tensor): Attention mask to distinguish padding tokens from real tokens.
        
        Returns:
            torch.Tensor: logits_task_a (Task A class scores) and logits_task_b (Task B class scores).
        """
        # Pass the input tokens through BERT to obtain hidden states (token embeddings)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the last hidden state from the BERT outputs (the token embeddings)
        last_hidden_state = outputs.last_hidden_state
        
        # Perform pooling to obtain sentence-level embeddings (mean pooling by default)
        if self.pooling == 'mean':
            # Expand the attention mask to match the size of the last hidden state for element-wise multiplication
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            
            # Compute the sum of embeddings, ignoring padding tokens (by applying attention mask)
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            
            # Sum the attention mask (counts real tokens) to later normalize the embeddings
            sum_mask = input_mask_expanded.sum(dim=1)
            
            # Prevent division by zero by clamping the sum_mask to a minimum value
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            
            # Normalize the embeddings by dividing the sum of embeddings by the number of real tokens
            sentence_embeddings = sum_embeddings / sum_mask
        else:
            # Use the [CLS] token's embedding as a representation of the whole sentence (default option)
            sentence_embeddings = last_hidden_state[:, 0, :]
        
        # Task A output: Class scores for sentence classification (based on the sentence embeddings)
        logits_task_a = self.classifier_task_a(sentence_embeddings)
        
        # Task B output: Class scores for sentiment analysis (based on the sentence embeddings)
        logits_task_b = self.classifier_task_b(sentence_embeddings)
        
        # Return the outputs for both tasks
        return logits_task_a, logits_task_b

# Sample test with a few sentences
if __name__ == "__main__":
    # Initialize tokenizer (to convert sentences into tokens) and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Define number of classes for each task
    num_classes_task_a = 3  # e.g., Sentence classification could have 3 classes: 'News', 'Opinion', 'Entertainment'
    num_classes_task_b = 2  # e.g., Sentiment analysis could have 2 classes: 'Positive', 'Negative'
    
    # Create an instance of the multi-task model with the specified number of classes
    model = MultiTaskSentenceTransformerModel(
        num_classes_task_a=num_classes_task_a,
        num_classes_task_b=num_classes_task_b
    )
    
    # Sample sentences for testing
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Transformer-based models change the world quickly.",
        "I love programming in Python.",
        "I love to work with machine learning stuff.",
        "Perhaps no picture ever made has more literally showed that the road to hell is paved with good intentions.",
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
    
    # Tokenize the sentences using BERT tokenizer
    encoded_input = tokenizer(
        sentences,  # List of sentences to tokenize
        padding=True,  # Pad sentences to make them the same length
        truncation=True,  # Truncate sentences that exceed the max token length for BERT
        return_tensors='pt'  # Return as PyTorch tensors for compatibility with the model
    )
    
    # Perform inference with the model (disable gradient computation during inference)
    with torch.no_grad():  # No need to track gradients during inference
        logits_task_a, logits_task_b = model(
            input_ids=encoded_input['input_ids'],  # Tokenized input sentences
            attention_mask=encoded_input['attention_mask']  # Attention mask to ignore padding tokens
        )
    
    # Apply softmax to logits to obtain probabilities for both tasks (optional)
    probabilities_task_a = torch.nn.functional.softmax(logits_task_a, dim=1)  # For classification (Task A)
    probabilities_task_b = torch.nn.functional.softmax(logits_task_b, dim=1)  # For sentiment analysis (Task B)
    
    # Print results for Task A: Sentence Classification
    print("Task A: Sentence Classification")
    for idx, probs in enumerate(probabilities_task_a):
        print(f"Sentence {idx+1}: {sentences[idx]}")
        print(f"Class Probabilities: {probs.numpy()}\n")  # Output the class probabilities for each sentence
    
    # Print results for Task B: Sentiment Analysis
    print("Task B: Sentiment Analysis")
    for idx, probs in enumerate(probabilities_task_b):
        print(f"Sentence {idx+1}: {sentences[idx]}")
        print(f"Sentiment Probabilities: {probs.numpy()}\n")  # Output the sentiment probabilities for each sentence

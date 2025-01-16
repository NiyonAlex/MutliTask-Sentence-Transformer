# Sentence Transformers and Multi-Task Learning

##  📖 Project Overview

This project explores sentence transformers in a multi-task setup. It includes tasks like encoding sentences into fixed-length embeddings, classifying sentences, examining the effects of freezing different model layers during training, and using layer-specific learning rates to improve performance.

---
## 📂 Project Layout

```plaintext
/project-root
├── README.md               # Project documentation
├── Dockerfile              # Docker setup
├── requirements.txt        # Project dependencies
├── /src                    # Source code for each task
│   ├── task1_sentence_transformer.py       # Task 1: Sentence Embeddings
│   ├── task2_multitask_learning_model.py   # Task 2: Multi-Task Model
│   └── task3_discussion.md                 # Task 3: Discussing
```
---
## Project Layout
## ⚙️ Setup Guide

### 🐳 Running with Docker

1. **Build the Docker Image**:
   ```sh
   docker build -t multi_task_transformer .
   ```
2. **Run the Container: To run the default task (Task 2):**:
   ```sh
   docker run --rm multi_task_transformer
   ```
3. **Running a Specific Task: Use the command below to run a specific task (e.g., Task 1):**:
   ```sh
   docker run --rm multi_task_transformer python /app/src/task1_sentence_transformer.py
   ```

### 📝 Manual Setup (without Docker)

1. **Create a Virtual Environment:**:
   ```sh
   python -m venv env
   source env/bin/activate  # Use `env\Scripts\activate` on Windows
   ```
2. **Install Required Packages:**:
   ```sh
   pip install -r requirements.txt
   ```
3. **Running Each Task: Use the following commands to run each task manually:**:
   ```sh
   python src/task1_sentence_transformer.py  # Task 1
   python src/task2_multitask_learning_model.py  # Task 2
   ```
---
## 🧩 Task Descriptions and How to Run Each


### Task 1: Sentence Transformer Implementation

- **Purpose:** Generate contextually aware sentence embeddings for downstream NLP tasks.

- **Command:**
    ```sh
    docker run --rm multi_task_transformer python /app/src/task1_sentence_transformer.py.py
    ```

- **Expected Output:** Contextual embeddings for each sample sentence.

- **Summary:**  
    - This task uses a pretrained transformer model with mean pooling to obtain efficient, fixed-length embeddings.
    - **Model Choice:** `bert-base-uncase` was selected for its balance of speed and accuracy.
    - **Pooling Method:** Mean pooling offers simplicity without sacrificing contextual information, making it suitable for varied NLP tasks.

- **Key Decisions:**
    - **Efficiency:** Mean pooling avoids extra computational layers, aligning with the task’s performance goals.
    - **Clarity:** This approach is straightforward, making it reproducible and accessible for a variety of tasks and models.

- **Output:** Each sentence produces a fixed-length embedding of shape [1, 256]. 

---
# Task 2: Explanation of Changes to Support Multi-Task Learning

## 1. Added Separate Output Layers for Each Task

- **self.classifier_task_a**: A linear layer mapping the sentence embeddings to class scores for **Task A** (Sentence Classification).
- **self.classifier_task_b**: A separate linear layer for **Task B** (Sentiment Analysis).

### Rationale:
By adding task-specific output layers, the model can share the transformer backbone and sentence embeddings while learning to perform different tasks simultaneously. This setup allows the model to learn representations that are beneficial for both tasks.

## 2. Modified the `forward` Method

After obtaining the `sentence_embeddings`, the embeddings are passed through both `classifier_task_a` and `classifier_task_b` to get logits for each task.

### Rationale:
Ensures that during the forward pass, outputs for all tasks are computed, which is essential for multi-task learning.

## 3. Choice of Tasks and Labels

- **Task A (Sentence Classification)**: Classifies sentences into predefined classes such as 'News', 'Opinion', and 'Entertainment'. We set `num_classes_task_a = 3`.
- **Task B (Sentiment Analysis)**: Classifies sentences based on sentiment with labels like 'Positive' and 'Negative'. We set `num_classes_task_b = 2`.

### Rationale:
These tasks are common in NLP and demonstrate how the model can handle multiple objectives.

## 4. Sample Output and Testing

In the test code, we generate sample sentences and obtain the probabilities for each class in both tasks.

- **Softmax Function**: Applied to logits to convert them into probabilities for better interpretability.

### Rationale:
Testing with sample data helps verify that the model outputs are as expected and that the multi-task setup works correctly.

---

## Notes on Multi-Task Learning Implementation

### 1. Shared Backbone and Embeddings

The BERT model and the pooling mechanism are shared across tasks, which allows the model to learn general representations useful for multiple tasks.

#### Benefit:
Reduces the overall number of parameters compared to training separate models for each task and can lead to improved performance due to shared learning.

### 2. Task-Specific Layers

Each task has its own output layer, which allows the model to specialize in each task without interference.

### 3. Training Considerations

- During training, losses from each task would be combined (e.g., summed or weighted) to update the model parameters.
- **Potential for Additional Tasks**: The architecture can be easily extended to include more tasks by adding additional task-specific output layers.

#### Flexibility:
Makes the model versatile for various applications in NLP.

---
## 🔍 Method Selection Rationale

In developing this multi-task transformer model, I explored several different approaches before settling on the final choices. Here’s a rundown of some alternative methods I considered, along with the reasons they weren’t selected:

### 1. **Alternative Pooling Methods (e.g., Max Pooling, Attention Pooling)**
While max pooling and attention pooling can capture specific nuances, they also add complexity and require extra tuning, which wasn’t necessary for this use case. Mean pooling provided a reliable way to get fixed-length embeddings without the extra setup.

### 2. **Separate Encoder Models for Each Task**
Using a shared encoder allowed general features to benefit both tasks while saving resources.

### 3. **Fully Fine-Tuned Model**
Freezing certain layers struck a balance between retaining pre-trained knowledge and adapting to specific tasks, avoiding overfitting.

### 4. **Uniform Learning Rate for All Layers**
Layer-wise learning rates offered flexibility, helping foundational layers retain general knowledge and task-specific layers adapt faster.

---
## 🏁 Final Notes
- **Testing:** Each task has been tested in Docker to make sure everything works as expected.
- **Efficiency:** The model is designed to be efficient, with minimal additional layers and thoughtful use of layer-specific learning rates.
# Automating Zero-Shot Detection Fine-Tuning with SkyCLIP and Agents

This guide provides a complete roadmap for fine-tuning **SkyCLIP** using **6 million satellite images** to achieve **state-of-the-art zero-shot detection**. The process incorporates **agent-based automation** for seamless data generation, training, and system optimization.

---

## **Overview**

### **Objective**
- Build a fine-tuned SkyCLIP model tailored to your proprietary satellite images for detecting **any user-defined category** (e.g., beachbars, vineyards).
- Enable real-time zero-shot search using embeddings stored in **Astra DB Vector Store**.

### **Key Features**
1. **Automated Caption Generation**: Use agents to process satellite images and generate captions with BLIP-2.
2. **Fine-Tuning SkyCLIP**: Train the model on generated image-text pairs using contrastive learning.
3. **Scalable Embedding Storage**: Use Astra DB Vector Store for efficient similarity search.
4. **Zero-Shot Querying**: Provide user-defined queries with robust inference from SkyCLIP.

---

## **Agent-Based Automation Workflow**

### **1. Components of the Agent System**

| Agent Name       | Function                                                   |
|-------------------|-----------------------------------------------------------|
| **CaptionAgent**  | Generates captions for images using BLIP-2.               |
| **DataAgent**     | Manages datasets, creating subsets, and organizing data.  |
| **TrainingAgent** | Fine-tunes SkyCLIP on generated captions and images.      |
| **EmbeddingAgent**| Computes and uploads embeddings to Astra DB.              |
| **QueryAgent**    | Handles user queries and retrieves matching images.       |

---

### **2. Workflow**

#### **Step 1: Image Caption Generation (CaptionAgent)**

The **CaptionAgent** automates the process of captioning the dataset.

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

class CaptionAgent:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def generate_captions(self, image_paths):
        captions = {}
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            output = self.model.generate(**inputs)
            captions[image_path] = self.processor.decode(output[0], skip_special_tokens=True)
        return captions

# Example usage
image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
agent = CaptionAgent()
captions = agent.generate_captions(image_paths)
```

---

#### **Step 2: Data Management (DataAgent)**

The **DataAgent** prepares subsets of data for training and ensures data is correctly labeled and stored.

```python
import datasets

class DataAgent:
    def create_dataset(self, image_paths, captions):
        return datasets.Dataset.from_dict({
            "image": image_paths,
            "text": captions
        })

# Example usage
data_agent = DataAgent()
dataset = data_agent.create_dataset(image_paths, list(captions.values()))
```

---

#### **Step 3: Fine-Tuning SkyCLIP (TrainingAgent)**

The **TrainingAgent** fine-tunes SkyCLIP using the dataset created by the DataAgent.

```python
import torch
from transformers import CLIPProcessor, CLIPModel

class TrainingAgent:
    def __init__(self, model_checkpoint, device="cuda"):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        state_dict = torch.load(model_checkpoint, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

    def fine_tune(self, dataset, num_epochs=5):
        for epoch in range(num_epochs):
            for record in dataset:
                image = record["image"]
                text = record["text"]

                # Process inputs
                image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                text_inputs = self.processor(text=text, return_tensors="pt").to(self.device)

                # Forward pass
                image_features = self.model.get_image_features(**image_inputs)
                text_features = self.model.get_text_features(**text_inputs)

                # Contrastive loss
                loss = self.compute_contrastive_loss(image_features, text_features)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def compute_contrastive_loss(self, image_features, text_features):
        # Implement contrastive loss (e.g., cosine similarity)
        pass
```

---

#### **Step 4: Embedding Storage in Astra DB (EmbeddingAgent)**

The **EmbeddingAgent** automates embedding generation and storage.

```python
from astrapy import DataAPIClient

class EmbeddingAgent:
    def __init__(self, model, processor, collection):
        self.model = model
        self.processor = processor
        self.collection = collection

    def generate_and_store_embeddings(self, dataset):
        for idx, record in enumerate(dataset):
            image = record["image"]
            inputs = self.processor(images=image, return_tensors="pt")
            vector = self.model.get_image_features(**inputs).detach().cpu().numpy().tolist()

            # Store in Astra DB
            self.collection.insert_one({
                "_id": idx,
                "$vector": vector,
                "metadata": {"image_path": record["image"]}
            })

# Astra DB setup
client = DataAPIClient(api_endpoint=ASTRA_DB_API_ENDPOINT, token=ASTRA_DB_APPLICATION_TOKEN)
database = client.get_database(ASTRA_DB_API_ENDPOINT)
collection = database.create_collection(
    "satellite_images",
    dimension=768,
    metric=VectorMetric.COSINE
)

# Example usage
embedding_agent = EmbeddingAgent(model, processor, collection)
embedding_agent.generate_and_store_embeddings(dataset)
```

---

#### **Step 5: Zero-Shot Querying (QueryAgent)**

The **QueryAgent** handles user queries and retrieves matching images from Astra DB.

```python
class QueryAgent:
    def __init__(self, model, processor, collection):
        self.model = model
        self.processor = processor
        self.collection = collection

    def query(self, text_query, limit=5):
        inputs = self.processor(text=text_query, return_tensors="pt")
        query_vector = self.model.get_text_features(**inputs).detach().cpu().numpy().tolist()
        results = self.collection.find(sort={"$vector": query_vector}, limit=limit)
        return results

# Example usage
query_agent = QueryAgent(model, processor, collection)
results = query_agent.query("Find beachbars")
```

---

## **Tools**

1. **SkyCLIP**
   - Pre-trained on millions of satellite image-text pairs.
   - [SkyCLIP GitHub Repository](https://github.com/ml4space/SkyCLIP)

2. **Astra DB Vector Store**
   - Scalable vector database for embedding storage.
   - [Astra DB Documentation](https://www.datastax.com/docs)

3. **BLIP-2**
   - State-of-the-art model for generating image captions.
   - [BLIP-2 on Hugging Face](https://huggingface.co/Salesforce/blip-image-captioning-base)

4. **Hugging Face Transformers**
   - For model loading, fine-tuning, and embedding generation.
   - [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)

---

## **Additional Resources**
- [Contrastive Learning](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html)
- [Zero-Shot Learning](https://paperswithcode.com/task/zero-shot-learning)
- [SkyCLIP Paper](https://github.com/ml4space/SkyCLIP)

---

## **Key Benefits**
- High-quality captions and embeddings tailored to your proprietary dataset.
- Scalability with Astra DB Vector Store for storing and retrieving millions of embeddings.
- Flexibility to add new categories with zero-shot capabilities.
```
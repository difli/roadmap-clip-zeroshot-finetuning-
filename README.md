# Automating Zero-Shot Classification of Land Use from Satellite Images with SkyCLIP and Agents

This guide provides a roadmap for fine-tuning **SkyCLIP** to improve **zero-shot classification** of **land use categories** (e.g., agriculture, streets, vineyards, beachbars) from satellite images. The process incorporates **agent-based automation** to efficiently manage data preparation, fine-tuning, and vector search deployment.

---

## **1. The Problem: Improving Zero-Shot Detection**

**Zero-shot detection** means the model can identify concepts it has never seen during training, based solely on textual descriptions. For satellite imagery, this includes complex or niche queries like "beachbars," "vineyards," "solar farms," etc.

SkyCLIP is already pre-trained on a large dataset of satellite images and paired textual descriptions. However:
1. It may lack **fine-grained understanding** of niche or novel queries.
2. It might underperform on **uncommon or ambiguous concepts** due to a limited understanding of specific land use domains.
3. Query optimization for zero-shot tasks may require additional **prompt engineering** for better context alignment.

The goal of this roadmap is to enhance SkyCLIP’s performance for **land use classification** by:
- Fine-tuning it with domain-specific data.
- Building a system that supports **zero-shot detection** for novel queries.
- Enabling **continuous learning** by incorporating user feedback.

---

## **2. Agent-Based Automation Workflow**

### **A. Components of the Agent System**

| **Agent Name**       | **Function**                                                   |
|-----------------------|---------------------------------------------------------------|
| **CaptionAgent**      | Generates captions for satellite images using BLIP-2.         |
| **DataAgent**         | Prepares datasets, creating subsets, and organizing data.      |
| **TrainingAgent**     | Fine-tunes SkyCLIP using generated image-text pairs.           |
| **EmbeddingAgent**    | Computes and stores embeddings in Astra DB.                   |
| **QueryAgent**        | Handles user queries and retrieves matching images.           |
| **ActiveLearningAgent** | Continuously refines the model by collecting user feedback.  |
| **PromptAgent**       | Generates optimized prompts for user queries for zero-shot tasks. |

---

### **B. Workflow**

#### **Step 1: Image Caption Generation (CaptionAgent)**

The **CaptionAgent** automates the process of generating captions for satellite images. This step provides meaningful text data for fine-tuning SkyCLIP.

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
```

---

#### **Step 2: Data Management (DataAgent)**

The **DataAgent** organizes image-caption pairs into a structured dataset for fine-tuning.

```python
import datasets

class DataAgent:
    def create_dataset(self, image_paths, captions):
        return datasets.Dataset.from_dict({
            "image": image_paths,
            "text": captions
        })
```

---

#### **Step 3: Fine-Tune with Diverse Text Prompts**

This step involves **fine-tuning SkyCLIP** with a set of **diverse text prompts** to improve its ability to handle new and novel land use categories.

**Approach**:
1. Use **text-to-image contrastive learning** to align visual embeddings with textual descriptions.
2. Generate synthetic text examples or use tools like **CLIP-style prompting** to define novel land use categories.

**Example Prompts**:
- "A dense vineyard in the countryside."
- "A small beachbar with colorful umbrellas."
- "A solar farm with rows of solar panels in a desert."

#### **What is a CLIP-Style Prompting Tool?**
CLIP-style prompting is a technique used to define **contextual descriptions** for categories that are new or poorly represented in the training data. While no specific **CLIP-prompting tool** exists as a standalone, this concept can be implemented with libraries like **OpenAI's CLIP** or **Hugging Face Transformers** by:
- Generating textual templates (e.g., "A photo of {query}.")
- Testing multiple variations to find the most effective description for aligning with embeddings.

For a CLIP-style prompting implementation, the **PromptAgent** (described below) can automatically generate these prompts.

---

#### **Step 4: Fine-Tuning SkyCLIP (TrainingAgent)**

The **TrainingAgent** fine-tunes SkyCLIP for improved land use classification using contrastive learning.

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

#### **Step 5: Continuous Learning (ActiveLearningAgent)**

The **ActiveLearningAgent** collects user feedback to improve the model iteratively.

1. **Feedback Collection**: Capture user feedback on retrieved results.
2. **Dataset Expansion**: Add new labeled examples based on feedback.
3. **Model Fine-Tuning**: Update the model using the expanded dataset.

---

#### **Step 6: Prompt Engineering for Zero-Shot Detection (PromptAgent)**

The **PromptAgent** enhances zero-shot inference by generating optimized prompts for user queries.

```python
class PromptAgent:
    def generate_prompt(self, query):
        templates = [
            "Satellite image showing {query}.",
            "Aerial view of {query}."
        ]
        return [template.format(query=query) for template in templates]

# Example usage
agent = PromptAgent()
prompts = agent.generate_prompt("vineyards")
```

---

#### **Step 7: Build and Test a Zero-Shot Inference Pipeline**

Combine the agents into a cohesive pipeline:
1. **Query Processing**: Use the PromptAgent to optimize user queries.
2. **Vector Matching**: Query Astra DB for similar embeddings.
3. **Result Display**: Present results to the user in a UI (e.g., Gradio).

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
```

---

## **Tools**

1. **SkyCLIP**
   - Pre-trained on millions of satellite image-text pairs.
   - [SkyCLIP GitHub Repository](https://github.com/wangzhecheng/SkyScript)

2. **Astra DB Vector Store**
   - Scalable vector database for storing embeddings and querying.
   - [Astra DB Documentation](https://www.datastax.com/docs)

3. **BLIP-2**
   - For generating captions for satellite images.
   - [BLIP-2 on Hugging Face](https://huggingface.co/Salesforce/blip-image-captioning-base)

4. **Prompt Engineering**
   - Generate optimized prompts for better zero-shot classification.
   - Example implementation in the PromptAgent section above.

---

## **Key Benefits**
- Tailored to land use classification for high accuracy in diverse categories.
- Scalable embedding storage and retrieval with Astra DB.
- Continuous learning with user feedback to improve classification.
- Seamless zero-shot classification for new land use categories.

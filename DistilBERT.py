import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd
from transformers import get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_excel(r"H:\DTST\train_dataset.xlsx", usecols=["Tokenized_Text", "Label"])

# Split the dataset into training and testing
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Load tokenizer and model from Hugging Face
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=df['Label'].nunique())

class TextDataset(Dataset):
    def __init__(self, dataframe):
        self.labels = dataframe['Label'].to_numpy()
        self.texts = dataframe['Tokenized_Text'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoded = tokenizer(self.texts[idx], return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        item = {key: val.squeeze() for key, val in encoded.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = TextDataset(train_df)
test_dataset = TextDataset(test_df)

# Adjusting hyperparameters
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,  # Increased epochs
    per_device_train_batch_size=16,  # Increased batch size
    per_device_eval_batch_size=32,  # Increased eval batch size for faster evaluation
    warmup_steps=1000,  # Increased warmup steps
    weight_decay=0.05,  # Increased regularization
    logging_dir='./logs',
    learning_rate=3e-5,  # Adjusted learning rate
    lr_scheduler_type='linear',  # Adding a learning rate scheduler
    evaluation_strategy="steps",  # Evaluate every so often
    eval_steps=500,  # Evaluation steps
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model='accuracy'  # Use accuracy to identify the best model
)

# Initialize the Trainer with new training arguments
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda p: {'accuracy': accuracy_score(p.label_ids, p.predictions.argmax(-1))}
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()

# Obtain predictions
predictions = trainer.predict(test_dataset)

print(f"Accuracy: {results['eval_accuracy']:.2f}")
print("\nEvaluation done.\n")
print("\nClassification Report:\n", classification_report(test_dataset.labels, predictions.predictions.argmax(axis=-1)))
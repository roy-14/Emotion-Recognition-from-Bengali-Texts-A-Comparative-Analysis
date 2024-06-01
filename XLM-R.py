import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load and prepare the dataset
df = pd.read_excel(r"H:\DTST\train_dataset.xlsx", usecols=["Tokenized_Text", "Label"])

# Splitting the dataset
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Load XLM-RoBERTa tokenizer and model
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

class TextDataset(Dataset):
    def __init__(self, dataframe):
        self.labels = dataframe['Label']
        self.texts = dataframe['Tokenized_Text']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        encoding = tokenizer(text, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
        encoding = {key: val.squeeze() for key, val in encoding.items()}
        encoding['labels'] = torch.tensor(label, dtype=torch.long)
        return encoding

train_dataset = TextDataset(train_df)
test_dataset = TextDataset(test_df)

# Model initialization
model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=len(df['Label'].unique()))

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    learning_rate=5e-5
)

# Initialize Trainer
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
print("Evaluation done.")
print("Classification Report:\n", classification_report(test_dataset.labels, predictions.predictions.argmax(axis=-1)))

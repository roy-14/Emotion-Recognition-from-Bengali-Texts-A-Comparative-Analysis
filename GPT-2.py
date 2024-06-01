import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Config, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load and prepare the dataset
df = pd.read_excel(r'H:\DTST\train_dataset.xlsx', usecols=["Tokenized_Text", "Label"])

# Split the dataset into training and testing
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Load tokenizer and configure padding token
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token_id to the eos_token

# Configure GPT-2 with the specific padding token
config = GPT2Config.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id, num_labels=df['Label'].nunique())

# Load the GPT-2 model for sequence classification
model = GPT2ForSequenceClassification.from_pretrained('gpt2', config=config)

class TextDataset(Dataset):
    def __init__(self, dataframe):
        self.labels = dataframe['Label'].to_numpy()
        self.texts = dataframe['Tokenized_Text'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs = tokenizer(self.texts[idx], return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': torch.tensor(self.labels[idx])}

train_dataset = TextDataset(train_df)
test_dataset = TextDataset(test_df)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    learning_rate=2e-5
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

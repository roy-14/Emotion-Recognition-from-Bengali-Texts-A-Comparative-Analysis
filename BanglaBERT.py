import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd
from transformers import get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load and prepare the dataset
df = pd.read_excel(r"H:\DTST\train_dataset.xlsx", usecols=["Tokenized_Text", "Label"])

# Splitting the dataset
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Assuming the BanglaBERT tokenizer and model are correctly installed or specified by a path
tokenizer = AutoTokenizer.from_pretrained('sagorsarker/bangla-bert-base')
model = AutoModelForSequenceClassification.from_pretrained('sagorsarker/bangla-bert-base', num_labels=df['Label'].nunique())

class TextDataset(Dataset):
    def __init__(self, dataframe):
        self.labels = dataframe['Label'].to_numpy()
        self.texts = dataframe['Tokenized_Text'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], return_tensors='pt', padding='max_length', max_length=256, truncation=True)
        item = {key: torch.tensor(val).squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = TextDataset(train_df)
test_dataset = TextDataset(test_df)


# Adjusted training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,  # Increased epochs
    per_device_train_batch_size=16,  # Adjusted batch size
    per_device_eval_batch_size=32,  # Larger for faster eval
    warmup_steps=800,  # Adjusted warmup
    weight_decay=0.05,  # Increased weight decay
    logging_dir='./logs',
    learning_rate=3e-5,  # Adjusted learning rate
    evaluation_strategy='steps',  # Evaluate at regular steps
    eval_steps=100,  # Evaluate every 100 steps
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model='accuracy'  # Use accuracy to find the best model
)

# Redefine the Trainer with new args
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
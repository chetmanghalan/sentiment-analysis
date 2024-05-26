import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class SentimentModel(nn.Module):
    def __init__(self, n_classes):
        super(SentimentModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 100, (k, 768)) for k in [3, 4, 5]
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, n_classes)

    def conv_and_pool(self, x, conv):
        x = conv(x).squeeze(3)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]
        bert_output = bert_output.unsqueeze(1)
        conv_out = torch.cat([self.conv_and_pool(bert_output, conv) for conv in self.convs], 1)
        out = self.dropout(conv_out)
        out = self.fc(out)
        return out


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def main():
    # loading data
    df = pd.read_excel(r'D:\Thesis\Data\ModelData\tomato_data.xlsx')[['Phrase', 'Sentiment']].rename(
        columns={'Phrase': 'text', 'Sentiment': 'label'})

    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', force_download=True)
    MAX_LEN = 128
    BATCH_SIZE = 16

    train_dataset = SentimentDataset(
        texts=df_train.text.to_numpy(),
        labels=df_train.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    test_dataset = SentimentDataset(
        texts=df_test.text.to_numpy(),
        labels=df_test.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentimentModel(n_classes=len(df.label.unique()))
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_data_loader) * 10

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    loss_fn = FocalLoss().to(device)

    best_accuracy = 0

    for epoch in range(80):
        print(f'Epoch {epoch + 1}/{10}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            test_data_loader,
            loss_fn,
            device,
            len(df_test)
        )
        print(f'Val   loss {val_loss} accuracy {val_acc}')

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc

    print('Training complete.')


if __name__ == "__main__":
    main()

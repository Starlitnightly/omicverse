#!/usr/bin/env python3
# coding: utf-8
"""
@file: trainer.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2024/04/03  create file.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from biollm.evaluate.bm_metrices_anno import compute_metrics
import os


class Trainer:
    def __init__(self, model, lr=1e-3, batch_size=32, device=None, save_path="best_model.pth"):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.save_path = save_path
        self.best_accuracy = 0.0
        self.best_model = None

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10):
        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                torch.tensor(y_train, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

            if X_val is not None and y_val is not None:
                val_loss, accuracy, f1, recall, precision = self.eval(X_val, y_val)
                print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.4f}, acc: {accuracy:.4f}, f1: {f1:.4f}, "
                      f"recall: {recall:.4f}, precision: {precision:.4f}")

                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_model = self.model
                    torch.save(self.model.state_dict(), self.save_path)
                    print(f"Best model saved with accuracy: {accuracy:.4f}")
            else:
                torch.save(self.model.state_dict(), f"{self.save_path}.ep{epoch}")
        self.model = self.best_model
        return self.model

    def eval(self, X_val, y_val):
        dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        y_true, y_pred = [], []
        total_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)

                # 计算 Loss
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()

                # 计算预测结果
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(predictions)

        avg_loss = total_loss / len(dataloader)
        res = compute_metrics(y_true, y_pred)
        accuracy, f1, recall, precision = res['accuracy'], res['macro_f1'], res['recall'], res['precision']
        return avg_loss, accuracy, f1, recall, precision

    def infer(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        return predictions, probabilities

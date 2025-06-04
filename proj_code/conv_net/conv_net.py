import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ConvNet:

    class StegoDataset(Dataset):

        def normalize_data(self, data):
            mean = np.mean(data)
            std_dev = np.std(data)

            data = (data - mean) / std_dev
            return data

        def __init__(self, data, labels):
            self.data = self.normalize_data(data)
            self.labels = labels

        def __getitem__(self, index):
            data_tensor = torch.tensor(self.data[index]).float()
            label = torch.tensor(self.labels[index]).float()

            return data_tensor, label

        def __len__(self):
            return len(self.data)

    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, output_size, activation_fn = nn.ReLU()):
            super().__init__()
            self.input_size = input_size
            self.hidden_size_1 = hidden_size_1
            self.hidden_size_2 = hidden_size_2
            self.hidden_size_3 = hidden_size_3
            self.layer_1 = nn.Linear(input_size, hidden_size_1)
            self.layer_2 = nn.Linear(hidden_size_1, hidden_size_2)
            self.layer_3 = nn.Linear(hidden_size_2, hidden_size_3)
            self.output_layer = nn.Linear(hidden_size_3, output_size)
            self.activation_fn = activation_fn

        def forward(self, x):
            h1 = self.activation_fn(self.layer_1(x))
            h2 = self.activation_fn(self.layer_2(h1))
            h3 = self.activation_fn(self.layer_3(h2))
            out = self.output_layer(h3)

            return out
    
    def __init__(self):
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.BATCH_SIZE = 64
        self.LEARNING_RATE = 0.001
        self.NUM_EPOCHS = 100
        self.model = self.NeuralNet(2049, 512, 256, 128, 1).to(self.DEVICE)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.LEARNING_RATE)
        self.loss_crt = torch.nn.BCEWithLogitsLoss()

        pass

    def train(self):
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_dataloader)
        predictions = []
        labels = []

        for _, batch in enumerate(self.train_dataloader):
            batch_values, batch_labels = batch
            batch_values, batch_labels = batch_values.to(self.DEVICE), batch_labels.to(self.DEVICE)

            output = self.model(batch_values)

            output_probabilities = torch.sigmoid(output)
            predicted_labels = (output_probabilities > 0.5).float().squeeze().tolist()

            predictions += predicted_labels
            labels += batch_labels.float().squeeze().tolist()

            loss = self.loss_crt(output.squeeze(), batch_labels.squeeze())
            loss_scalar = loss.item()

            loss.backward()
            self.optimizer.step()
            self.model.zero_grad()

            epoch_loss += loss_scalar

        epoch_loss = epoch_loss/num_batches

        return epoch_loss, predictions, labels
    
    def eval(self, dataloader):
        self.model.eval()
        epoch_loss = 0.0
        num_batches = len(dataloader)
        predictions = []
        labels = []

        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                batch_values, batch_labels = batch
                batch_values, batch_labels = batch_values.to(self.DEVICE), batch_labels.to(self.DEVICE)

                output = self.model(batch_values)

                output_probailities = torch.sigmoid(output)
                predicted_labels = (output_probailities > 0.5).float().squeeze().tolist()

                predictions += predicted_labels
                labels += batch_labels.float().squeeze().tolist()

                loss = self.loss_crt(output.squeeze(), batch_labels.squeeze())
                loss_scalar = loss.item()

                epoch_loss += loss_scalar

        epoch_loss = epoch_loss/num_batches

        return epoch_loss, predictions, labels

    def read_dataset(self, clean_path, stego_path):
        clean_data = np.load(clean_path)
        stego_data = np.load(stego_path)

        clean_labels = np.zeros(clean_data.shape[0])
        stego_labels = np.ones(stego_data.shape[0])

        labels = np.hstack((clean_labels, stego_labels))
        data = np.vstack((clean_data, stego_data))

        random_shuffle = np.random.permutation(labels.size)

        labels = labels[random_shuffle]
        data = data[random_shuffle]

        train_data = data[:90000]
        validation_data = data[90000:120000]
        test_data = data[120000:150000]

        train_labels = labels[:90000]
        validation_labels = labels[90000:120000]
        test_labels = labels[120000:150000]

        self.train_dataset = self.StegoDataset(train_data, train_labels)
        self.validation_dataset = self.StegoDataset(validation_data, validation_labels)
        self.test_dataset = self.StegoDataset(test_data, test_labels)

        self.train_dataloader = DataLoader(dataset = self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        self.validation_dataloader = DataLoader(dataset = self.validation_dataset, batch_size=self.BATCH_SIZE, shuffle=False)
        self.test_dataloader = DataLoader(dataset = self.test_dataset, batch_size=self.BATCH_SIZE, shuffle=False)

    def compute_accuracy(self, predictions, labels):

        num_correct = len([(p,l) for (p,l) in zip(predictions,labels) if p==l])
        epoch_accuracy = num_correct / len(labels)

        return epoch_accuracy

    def export_model_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model_weights(self, path):
        weights = torch.load(path, weights_only = True)
        self.model.load_state_dict(weights)

    def train_pipeline(self):
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        for epoch_idx in range(self.NUM_EPOCHS):
            train_epoch_loss, train_predictions, train_labels = self.train()

            val_epoch_loss, val_predictions, val_labels = self.eval(self.validation_dataloader)
            
            train_acc = self.compute_accuracy(train_predictions, train_labels)
            val_acc = self.compute_accuracy(val_predictions, val_labels)

            _, _, f1_score, _ = precision_recall_fscore_support(val_labels, val_predictions, average='binary', zero_division=0.0)

            train_losses.append(train_epoch_loss)
            val_losses.append(val_epoch_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            print("epoch %d, train loss=%f, train acc=%f, val loss=%f, val acc=%f, f1=%f" % (
                epoch_idx,
                train_epoch_loss,
                train_acc,
                val_epoch_loss,
                val_acc,
                f1_score
            ))

        return train_losses, val_losses, train_accuracies, val_accuracies
    
    def plot_confusion_matrix(self, val_predictions, val_labels, save):
        # 
        cf_matrix = confusion_matrix(val_predictions, val_labels)
        
        plt.figure()
        ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='d')
        ax.set_xlabel('Clasa corectă')
        ax.set_ylabel('Clasa prezisă');

        ax.xaxis.set_ticklabels(['curat','stego'])
        ax.yaxis.set_ticklabels(['curat','stego'])

        if save == True:
            plt.savefig('/home/alexmiclea/Documents/Facultate/Licenta/plots/conf_matrix.pdf', format = 'pdf')
        plt.show()
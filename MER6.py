import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import torchvision.models as models
from torchvision.transforms.functional import resize
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.optim as optim
from torch.optim import lr_scheduler


# Define the custom dataset class
class CustomMicroExpressionDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, csv_file, transform=None, target_size=(224, 224)):
        self.data_root = data_root
        self.transform = transform
        self.frame_info = pd.read_csv(csv_file)
        self.target_size = target_size

    def __len__(self):
        return len(self.frame_info)

    def __getitem__(self, idx):
        subject = str(self.frame_info.iloc[idx]["Subject"]).zfill(2)
        filename = self.frame_info.iloc[idx]["Filename"]
        onset_frame = self.frame_info.iloc[idx]["OnsetFrame"]
        offset_frame = self.frame_info.iloc[idx]["OffsetFrame"]
        emotion_label = self.frame_info.iloc[idx]["Emotion"]
        apex_frame = self.frame_info.iloc[idx]["ApexFrame"]

        img_folder = os.path.join(self.data_root, f'sub{subject}', f'{filename}')

        frames = []
        for frame_number in range(onset_frame, offset_frame + 1):
            img_path = os.path.join(img_folder, f'reg_img{frame_number}.jpg')
            img = Image.open(img_path).convert('RGB')
            img = resize(img, self.target_size, Image.BILINEAR)
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        req_emotion = 1 if emotion_label == "happiness" else 0
        identity_label = int(subject)  # Identity label for the person

        frames_tensor = torch.stack(frames)
        seq_length = frames_tensor.shape[0]
        return frames_tensor, identity_label, req_emotion, seq_length

# Define Siamese network architecture for microexpression recognition
num_people = 26
num_channels = 3  # RGB images have 3 channels
height = 224
width = 224

class SiameseNetwork(nn.Module):
    def __init__(self, seq_length):
        super(SiameseNetwork, self).__init__()

        self.seq_length = seq_length
        # LSTM layer to process sequences
        self.lstm = nn.LSTM(input_size=num_channels * height * width, hidden_size=128, batch_first=True)
        
        self.identity_classifier = nn.Linear(128, 64)
        self.hidden_layer1 = nn.Linear(64, 64)  # Match the output dimension with the next layer
        self.output_layer1 = nn.Linear(64, num_people)
        self.happiness_classifier = nn.Linear(128, 64)
        self.hidden_layer2 = nn.Linear(64, 64)  # Match the output dimension with the next layer
        self.output_layer2 = nn.Linear(64, 7)

    def forward(self, input1, input2):
        # Unpack the PackedSequence into a tensor
        input1, lengths1 = pad_packed_sequence(input1, batch_first=True)
        input2, lengths2 = pad_packed_sequence(input2, batch_first=True)

        # Move the tensors to the desired device
        input1 = input1.to(device)
        input2 = input2.to(device)

        print("type",type(input1))
        
        batch_size, seq_length, num_features = input1.size(0), input1.size(1), input1.size(2)

        # Reshape input1 and input2 before passing to the LSTM
        input1 = input1.view(batch_size, seq_length, -1)
        input2 = input2.view(batch_size, seq_length, -1)

        # Forward pass through a single LSTM layer
        output1, _ = self.lstm(input1)
        output2, _ = self.lstm(input2)

        # Take the last hidden state as the representation of the sequence
        output1 = output1[:, -1, :]
        output2 = output2[:, -1, :]
    
        # Additional fully connected layers for identity prediction
        identity_output = self.identity_classifier(output1)
        identity_output = F.relu(identity_output)
        identity_output = self.hidden_layer1(identity_output)
        identity_output = F.relu(identity_output)
        identity_output = self.output_layer1(identity_output)

        # Additional fully connected layers for happiness prediction
        happiness_output = self.happiness_classifier(output1)
        happiness_output = F.relu(happiness_output)
        happiness_output = self.hidden_layer2(happiness_output)
        happiness_output = F.relu(happiness_output)
        happiness_output = self.output_layer2(happiness_output)

        cosine_similarity = nn.functional.cosine_similarity(output1, output2)

        return identity_output, happiness_output, cosine_similarity

# Custom loss function to enforce recognition only for happy expressions of the same person
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output1, output2, target, cosine_similarity):

        # Multiply similarity score by target (1 for happy, 0 for others)
        weighted_similarity = cosine_similarity * target

        # Calculate mean squared error loss
        loss = torch.mean((weighted_similarity - 1) ** 2)
        return loss
    
# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define paths to your dataset and CSV file
data_root = 'D:/NCSU/CS-B/project/Data/CASME2/'
train_csv_file = 'D:/NCSU/CS-B/project/Data/CASME2/CASME2_train.csv'
test_csv_file = 'D:/NCSU/CS-B/project/Data/CASME2/CASME2_test.csv'

train_dataset = CustomMicroExpressionDataset(data_root, train_csv_file, transform=transform)
test_dataset = CustomMicroExpressionDataset(data_root, test_csv_file, transform=transform)

def custom_collate_fn(batch):
    sequences, identity_labels, req_emotion_labels, lengths = zip(*batch)

    # Calculate the lengths of each sequence
    lengths = [len(seq) for seq in sequences]

    # Find the maximum sequence length in this batch
    max_length = max(lengths)

    # Initialize a tensor to store the padded sequences
    padded_sequences = torch.zeros(len(sequences), max_length, num_channels, height, width)

    for i, seq in enumerate(sequences):
        seq_length = len(seq)
        padded_sequences[i, :seq_length] = seq

    # Sort sequences by length (descending order)
    sorted_indices = sorted(range(len(sequences)), key=lambda i: lengths[i], reverse=True)
    padded_sequences = padded_sequences[sorted_indices]
    identity_labels = [identity_labels[i] for i in sorted_indices]
    req_emotion_labels = [req_emotion_labels[i] for i in sorted_indices]

    # Create a PackedSequence
    packed_sequences = pack_padded_sequence(
        padded_sequences,
        lengths,
        batch_first=True,
        enforce_sorted=False  # Set this to False if sequences are not sorted
    )

    return packed_sequences, torch.tensor(identity_labels), torch.tensor(req_emotion_labels), lengths



# Create data loaders with the custom collate function
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Siamese network
seq_length = 50  # Set your desired sequence length
siamese_net = SiameseNetwork(seq_length).to(device)

# Define custom loss function
criterion = CustomLoss().to(device)

# Define optimizer
optimizer = optim.Adam(siamese_net.parameters(), lr=0.001)

# Define a learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training loop
num_epochs = 10
# In the training loop, unpack the sequences before passing them to the model
for epoch in range(num_epochs):
    siamese_net.train()
    scheduler.step()
    for batch_data in train_loader:
        sequences, identity_label, req_emotion, lengths = batch_data
        sequences = sequences.to(device)
        identity_label = identity_label.to(device)
        req_emotion = req_emotion.to(device)

        output1, output2, cosine_similarity = siamese_net(sequences, sequences)

        target = (identity_label == identity_label.unsqueeze(1)) & (req_emotion == 1).unsqueeze(1)
        target = target.float().to(device)

        loss = criterion(output1, output2, target, cosine_similarity)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# siamese_net.load_state_dict(torch.load('siamese_model.pth'))

# Testing (evaluate the model's performance)
siamese_net.eval()
correct = 0
total_samples = 0
true_labels = []
predicted_labels = []

with torch.no_grad():
    for batch_data in test_loader:
        frames, identity_label, req_emotion, lengths = batch_data
        frames = frames.to(device)
        identity_label = identity_label.to(device)
        req_emotion = req_emotion.to(device)

        output1, output2, cosine_similarity = siamese_net(frames, frames)

        # Calculate target (1 for happy expressions of the same person, 0 otherwise)
        target = (identity_label == identity_label.unsqueeze(1)) & (req_emotion == 1).unsqueeze(1)
        target = target.float().to(device)

        # Predictions (1 if similarity > 0.5, else 0)
        predictions = (cosine_similarity > 0.5).float()

        # Calculate accuracy
        correct += (predictions == target).sum().item()
        total_samples += target.size(0)

        true_labels.extend(target.cpu().numpy())
        predicted_labels.extend(predictions.cpu().numpy())

accuracy = correct / total_samples
print(f"Accuracy: {100 * accuracy:.2f}%")

# true_labels = np.array(true_labels)
# predicted_labels = np.array(predicted_labels)

# cm = confusion_matrix(true_labels, predicted_labels)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Happy", "Happy"])

# # Plot confusion matrix
# disp.plot(cmap=plt.cm.Blues, values_format=".4g")
# plt.title("Confusion Matrix")
# plt.show()

# Save or load the trained model for later use
torch.save(siamese_net.state_dict(), 'siamese_model.pth')
# siamese_net.load_state_dict(torch.load('siamese_model.pth'))

import os
import tkinter as tk
from tkinter import filedialog
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

"""
Hier haben wir alle relevanten imports
"""

"""
Dieses Modul enthält den Code für die
Geschlechtererkennung anhand von Bildern.
"""


"""
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!kaggle datasets download -d cashutosh/gender-classification-dataset
"""


"""
try:
    with zipfile.ZipFile("/content/gender-classification-dataset.zip", "r") as zip_ref:
        zip_ref.extractall("/content")
except FileNotFoundError:
    print("Datei wurde nicht gefunden")
"""

"""
Quelle: https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset?resource=download
Objektorientierung: Eigene Klasse für die Datensätze
"""


class GenderDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = []
        self.labels = []

        for label, gender in enumerate(["male", "female"]):
            gender_dir = os.path.join(directory, gender)
            for img_file in os.listdir(gender_dir):
                if img_file.endswith(".jpg"):
                    img_path = os.path.join(gender_dir, img_file)
                    self.images.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


"""
Visualisierung der Aufteilung der Datensätze
"""


def visualize_data_distribution(dataset):
    label_to_str = lambda x: "Männlich" if x == 0 else "Weiblich"
    labels = [label_to_str(label) for _, label in dataset]
    label_count = {label: labels.count(label) for label in set(labels)}

    sizes = [label_count["Männlich"], label_count["Weiblich"]]
    colors = ["lightblue", "lightcoral"]
    explode = (0.1, 0)

    plt.pie(
        sizes,
        explode=explode,
        labels=list(label_count.keys()),
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=140,
    )
    plt.axis("equal")
    plt.title("Verteilung der Daten im Dataset")
    plt.show()


"""
Quelle:https://stefanobosisio1.medium.com/a-reverse-engineer-approach-to-explain-attention-and-memory-88ca63dfd1f1
Zeile 93 bis Zeile 98, https://www.youtube.com/watch?v=V_xro1bcAuA&t=43293s&ab_channel=freeCodeCamp.org
KI arbeiten mit Zahlen,
deshalb werden die Daten in Tensors umgewandelt.
"""

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


"""
Quelle: https://www.youtube.com/watch?v=V_xro1bcAuA&t=43293s&ab_channel=freeCodeCamp.org
Visualisierung der Aufteilung der Datensätze (Training Datensatz)
"""
dataset = GenderDataset(r"C:\Users\Rahim\Desktop\archive\Training", transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

"""
Quelle: https://stefanobosisio1.medium.com/a-reverse-engineer-approach-to-explain-attention-and-memory-88ca63dfd1f1
Ab Zeile 107 bis Zeile: 160,
Das Künstliche Neuronales Netzwerk welches wir benutzen
"""


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


"""
Quelle: https://www.youtube.com/watch?v=V_xro1bcAuA&t=43293s&ab_channel=freeCodeCamp.org
Treiningsablauf
Hier wird das Modell trainiert. Die Menge an Durchläufen kann man einstellen,
um das Ergebniss zu verbessern
"""
model_1 = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_1.parameters(), lr=0.001)

"""
for epoch in range(1):
    for images, labels in dataloader:
        # Forward-Pass
        outputs = model_1(images)
        loss = criterion(outputs, labels)

        # Backward und Optimieren
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
"""


"""
Hier laden wir unser Model was wir davon gespeichert haben, damit man einen
Lernfortschritt Verlust von über einer Stunde verhindert.
"""
model_1 = SimpleCNN()
optimizer = optim.Adam(model_1.parameters(), lr=0.001)


MODEL_FILE_PATH = r"C:\Users\Rahim\Desktop\PythonVertiefung\model_1.pth"
with open(MODEL_FILE_PATH, "rb") as file:
    model_state = torch.load(file)
model_1.load_state_dict(model_state["model"])
optimizer.load_state_dict(model_state["optimizer"])
dataset = GenderDataset(r"C:\Users\Rahim\Desktop\archive\Validation", transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


"""
Quelle: https://stefanobosisio1.medium.com/a-reverse-engineer-approach-to-explain-attention-and-memory-88ca63dfd1f1
Zeile 163 bis Zeile 175, https://www.youtube.com/watch?v=V_xro1bcAuA&t=43293s&ab_channel=freeCodeCamp.org
Das Modell wird auf den Validations Datensatz
ausgeführt und dabei auf die Korrektheit überprüft
"""
"""
model_1.eval()

total = 0
correct = 0
with torch.no_grad():
    for images, labels in dataloader:
        outputs = model_1(images)
        _, guesses = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (guesses == labels).sum().item()

accuracy = 100 * correct / total
print(f"Genauigkeit des Modells auf den Testbildern: {accuracy:.2f}%")
"""


def upload_images():
    file_paths = filedialog.askopenfilenames()
    results = []
    count_male = 0
    count_female = 0

    for file_path in file_paths:
        if file_path:
            img = Image.open(file_path)
            img = img.resize((224, 224))
            img_tensor = transforms.ToTensor()(img).unsqueeze(0)

            with torch.no_grad():
                guesses = model(img_tensor)
                _, guesses_class = torch.max(guesses, 1)
                if guesses_class.item() == 0:
                    result = "Mann"
                    count_male += 1
                else:
                    result = "Frau"
                    count_female += 1
                results.append(f"{os.path.basename(file_path)}: {result}")

    total = count_male + count_female
    if total > 0:
        male_percentage = (count_male / total) * 100
        female_percentage = (count_female / total) * 100
        results.append(f"\nAnteil Männer: {male_percentage:.2f}%")
        results.append(f"Anteil Frauen: {female_percentage:.2f}%")

    result_label.config(text="\n".join(results))


model_1 = SimpleCNN()
model_state = torch.load(r"C:\Users\Rahim\Desktop\PythonVertiefung\model_1.pth")
model_1.load_state_dict(model_state["model"])

optimizer = optim.Adam(model_1.parameters(), lr=0.001)
optimizer.load_state_dict(model_state["optimizer"])

model = SimpleCNN()
MODEL_FILE_PATH = r"C:\Users\Rahim\Desktop\PythonVertiefung\model_1.pth"
with open(MODEL_FILE_PATH, "rb") as file:
    model_state = torch.load(file)
model.load_state_dict(model_state["model"])
"""Quelle: https://stefanobosisio1.medium.com/a-reverse-engineer-approach-to-explain-attention-and-memory-88ca63dfd1f1"""
model.eval()

root = tk.Tk()
root.title("Geschlechterklassifizierung")

upload_btn = tk.Button(root, text="Bilder hochladen", command=upload_images)
upload_btn.pack()

result_label = tk.Label(root, text="", justify=tk.LEFT)
result_label.pack()

root.mainloop()

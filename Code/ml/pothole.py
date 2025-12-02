import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.optim as optim

# 1ï¸âƒ£ Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2ï¸âƒ£ Load dataset
train_data = datasets.ImageFolder(root="train", transform=transform)
test_data  = datasets.ImageFolder(root="test", transform=transform)


train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

# 3ï¸âƒ£ Model (use pretrained ResNet18)
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 2 classes: Pothole, Plain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 4ï¸âƒ£ Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5ï¸âƒ£ Training loop
for epoch in range(5):  # increase for better results
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/5] Loss: {running_loss/len(train_loader):.4f}")

# 6ï¸âƒ£ Test Accuracy
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# 7ï¸âƒ£ Save model
torch.save(model.state_dict(), "pothole_model.pth")
print("Model saved as pothole_model.pth âœ…")
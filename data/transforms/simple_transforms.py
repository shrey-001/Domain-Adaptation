from torchvision import transforms

def transform():
    transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],
                         std=[0.5])
    ])
    return transform
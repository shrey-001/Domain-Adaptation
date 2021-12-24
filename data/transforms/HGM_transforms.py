from torchvision import transforms

def transform():
    transforms = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],
                         std=[0.5])
    ])
    return transforms
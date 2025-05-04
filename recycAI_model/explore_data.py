import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Göreli yol (başka bilgisayarlarda da çalışır)
dataset_dir = os.path.join("dataset", "trashnet")

# Sınıfları ve ilk 5 görseli görselleştir
classes = os.listdir(dataset_dir)
classes.sort()

plt.figure(figsize=(10, 5))

for i, sample_class in enumerate(classes):
    sample_dir = os.path.join(dataset_dir, sample_class)
    sample_images = os.listdir(sample_dir)[:5]

    for j, image_name in enumerate(sample_images):
        img_path = os.path.join(sample_dir, image_name)
        img = mpimg.imread(img_path)
        plt.subplot(len(classes), 5, i * 5 + j + 1)
        plt.imshow(img)
        plt.title(sample_class)
        plt.axis('off')

plt.tight_layout()
plt.show()

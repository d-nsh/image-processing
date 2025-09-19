import cv2
import numpy as np
from matplotlib import pyplot as plt

# Загружаем цветное изображение
img_color = cv2.imread('winter_cat.png')
img_color_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

# Конвертируем в серое для эквализации
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
img_eq = cv2.equalizeHist(img_gray)

# Создаем сетку 2x2
plt.figure(figsize=(12, 10))

# 1. Сверху слева: исходное цветное изображение
plt.subplot(2, 2, 1)
plt.imshow(img_color_rgb)
plt.title('Исходное цветное изображение')
plt.axis('off')

# 2. Снизу слева: гистограмма серого канала
plt.subplot(2, 2, 3)
plt.hist(img_gray.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
plt.title('Гистограмма (по яркости)')
plt.xlabel('Яркость')
plt.ylabel('Количество пикселей')
plt.xlim([0, 255])
plt.grid(True, alpha=0.3)

# 3. Сверху справа: эквализированное серое изображение
plt.subplot(2, 2, 2)
plt.imshow(img_eq, cmap='gray')
plt.title('Эквализация по яркости')
plt.axis('off')

# 4. Снизу справа: гистограмма после эквализации
plt.subplot(2, 2, 4)
plt.hist(img_eq.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
plt.title('Гистограмма после эквализации')
plt.xlabel('Яркость')
plt.ylabel('Количество пикселей')
plt.xlim([0, 255])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
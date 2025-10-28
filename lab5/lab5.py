import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb

"""Загружаем изображение. Преобразуем в модель RGB"""
image = cv.imread('321.jpg')
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.title('Исходное изображение')
plt.show()

"""Отображаем разные каналы RGB на трехмерном графике"""
r, g, b = cv.split(image_rgb)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
pixel_colors = image_rgb.reshape((np.shape(image_rgb)[0]*np.shape(image_rgb)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
axis.set_title("RGB цветовое пространство")
plt.show()

"""Построим отдельные цветовые каналы"""
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Красный канал
red_channel = image_rgb.copy()
red_channel[:, :, 1] = 0  # green to 0
red_channel[:, :, 2] = 0  # blue to 0
axes[0].imshow(red_channel)
axes[0].set_title('Red Channel')
axes[0].axis('off')

# Зеленый канал  
green_channel = image_rgb.copy()
green_channel[:, :, 0] = 0  # red to 0
green_channel[:, :, 2] = 0  # blue to 0
axes[1].imshow(green_channel)
axes[1].set_title('Green Channel')
axes[1].axis('off')

# Синий канал
blue_channel = image_rgb.copy()
blue_channel[:, :, 0] = 0  # red to 0
blue_channel[:, :, 1] = 0  # green to 0
axes[2].imshow(blue_channel)
axes[2].set_title('Blue Channel')
axes[2].axis('off')

plt.tight_layout()
plt.show()

"""Преобразуем изображение в цветовую модель HSV"""
image_hsv = cv.cvtColor(image_rgb, cv.COLOR_RGB2HSV)

"""Отобразим разные каналы HSV на трехмерном графике"""
h, s, v = cv.split(image_hsv)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation") 
axis.set_zlabel("Value")
axis.set_title("HSV цветовое пространство")
plt.show()

"""Анализ гистограмм HSV каналов"""
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Гистограмма Hue
axes[0, 0].hist(h.reshape(-1), bins=180, range=[0, 180], color='red', alpha=0.7)
axes[0, 0].set_xlabel('Hue Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Гистограмма Hue канала')
axes[0, 0].grid(True, alpha=0.3)

# Гистограмма Saturation
axes[0, 1].hist(s.reshape(-1), bins=256, range=[0, 256], color='green', alpha=0.7)
axes[0, 1].set_xlabel('Saturation Value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Гистограмма Saturation канала')
axes[0, 1].grid(True, alpha=0.3)

# Гистограмма Value
axes[1, 0].hist(v.reshape(-1), bins=256, range=[0, 256], color='blue', alpha=0.7)
axes[1, 0].set_xlabel('Value (Brightness)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Гистограмма Value канала')
axes[1, 0].grid(True, alpha=0.3)

# Показать HSV изображение
axes[1, 1].imshow(image_hsv)
axes[1, 1].set_title('HSV изображение')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

"""Создадим маски для зеленой юбки и покажем диапазоны цветов"""
# Зеленая юбка: Hue ~35-85
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])

# Желтый фон: Hue ~20-35  
lower_yellow = np.array([20, 50, 50])
upper_yellow = np.array([35, 255, 255])

# Показать цветовые диапазоны
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Зеленый диапазон
lo_green_square = np.full((10, 10, 3), lower_green, dtype=np.uint8)
up_green_square = np.full((10, 10, 3), upper_green, dtype=np.uint8)

axes[0, 0].imshow(hsv_to_rgb(lo_green_square/255.0))
axes[0, 0].set_title('Нижний зеленый')
axes[0, 0].axis('off')

axes[0, 1].imshow(hsv_to_rgb(up_green_square/255.0))
axes[0, 1].set_title('Верхний зеленый')
axes[0, 1].axis('off')

# Желтый диапазон
lo_yellow_square = np.full((10, 10, 3), lower_yellow, dtype=np.uint8)
up_yellow_square = np.full((10, 10, 3), upper_yellow, dtype=np.uint8)

axes[1, 0].imshow(hsv_to_rgb(lo_yellow_square/255.0))
axes[1, 0].set_title('Нижний желтый')
axes[1, 0].axis('off')

axes[1, 1].imshow(hsv_to_rgb(up_yellow_square/255.0))
axes[1, 1].set_title('Верхний желтый')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

"""Применим маску для зеленой юбки"""
mask_green = cv.inRange(image_hsv, lower_green, upper_green)
result_green = cv.bitwise_and(image_rgb, image_rgb, mask=mask_green)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title('Исходное изображение')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask_green, cmap="gray")
plt.title('Маска зеленого цвета')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(result_green)
plt.title('Выделенная зеленая юбка')
plt.axis('off')

plt.tight_layout()
plt.show()

"""Применим маску для желтого фона"""
mask_yellow = cv.inRange(image_hsv, lower_yellow, upper_yellow)
result_yellow = cv.bitwise_and(image_rgb, image_rgb, mask=mask_yellow)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title('Исходное изображение')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask_yellow, cmap="gray")
plt.title('Маска желтого цвета')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(result_yellow)
plt.title('Выделенный желтый фон')
plt.axis('off')

plt.tight_layout()
plt.show()


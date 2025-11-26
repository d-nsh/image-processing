import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

# Загружаем и подготавливаем изображение
image = cv.imread('321.jpg')
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
pixels = image_rgb.reshape(-1, 3)

# Mean Shift кластеризация
bandwidth = estimate_bandwidth(pixels, quantile=0.1, n_samples=1000)
print(f"Bandwidth: {bandwidth}")

meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
labels = meanshift.fit_predict(pixels)
centers = meanshift.cluster_centers_

print(f"Найдено кластеров: {len(centers)}")

# Создаем сегментированное изображение
segmented = centers[labels].reshape(image_rgb.shape).astype(np.uint8)

# Находим зеленый кластер (Hue 35-85 в HSV)
centers_hsv = cv.cvtColor(centers.reshape(1, -1, 3).astype(np.uint8), cv.COLOR_RGB2HSV)
centers_hsv = centers_hsv.reshape(-1, 3)

green_cluster = None
for i, (center_rgb, center_hsv) in enumerate(zip(centers, centers_hsv)):
    hue = center_hsv[0]
    if 35 <= hue <= 85:  # Зеленый диапазон
        green_cluster = i
        print(f"Зеленый кластер {i}: RGB{tuple(center_rgb)}, HSV{tuple(center_hsv)}")
        break

# Создаем и улучшаем маску
mask = (labels == green_cluster).reshape(image_rgb.shape[:2])
mask = mask.astype(np.uint8) * 255

# Очищаем маску
kernel = np.ones((5,5), np.uint8)
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

# Применяем маску
result = cv.bitwise_and(image_rgb, image_rgb, mask=mask)

# Визуализация
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(image_rgb)
plt.title('Исходное')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(segmented)
plt.title(f'Segmentation\n{len(centers)} clusters')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(mask, cmap='gray')
plt.title('Mask')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(result)
plt.title('Result')
plt.axis('off')

plt.tight_layout()
plt.show()

# Сохраняем результат
cv.imwrite('skirt_result.jpg', cv.cvtColor(result, cv.COLOR_RGB2BGR))

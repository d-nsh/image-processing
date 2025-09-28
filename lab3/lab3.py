import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загружаем изображение
img = cv2.imread('orig.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(f'Размер изображения = {img_rgb.shape}')
print(f'Диапазон яркостей: {np.min(img_rgb)} - {np.max(img_rgb)}')

'''АНАЛИЗ ЦВЕТОВ'''


'''Исходное изображение'''
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title('1. Исходное изображение')
plt.axis('off')

'''Инверсия цветов (255 - значение)'''
img_inverted = 255 - img_rgb

plt.subplot(2, 3, 2)
plt.imshow(img_inverted)
plt.title('2. Инверсия цветов (255 - значение)')
plt.axis('off')

'''Инверсия только яркости (в HSV)'''
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(img_hsv)
v_inverted = 255 - v
img_hsv_inverted = cv2.merge([h, s, v_inverted])
img_brightness_inverted = cv2.cvtColor(img_hsv_inverted, cv2.COLOR_HSV2RGB)

plt.subplot(2, 3, 3)
plt.imshow(img_brightness_inverted)
plt.title('3. Инверсия только яркости (HSV)')
plt.axis('off')

'''Смена цветовых каналов'''
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # RGB -> BGR
img_rgb_swapped = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # BGR -> RGB (должно быть как исходное)

plt.subplot(2, 3, 4)
plt.imshow(img_bgr)
plt.title('4. BGR вместо RGB (неправильный порядок)')
plt.axis('off')

'''Разделение цветовых каналов'''
plt.subplot(2, 3, 5)
r, g, b = cv2.split(img_rgb)
'''Пробуем разные комбинации каналов'''
img_channel_swapped = cv2.merge([b, g, r])

plt.imshow(img_channel_swapped)
plt.title('5. Смена каналов R и B местами')
plt.axis('off')

'''Негативное изображение (инверсия каждого канала отдельно)'''
r_inv, g_inv, b_inv = 255 - r, 255 - g, 255 - b
img_negative = cv2.merge([r_inv, g_inv, b_inv])

plt.subplot(2, 3, 6)
plt.imshow(img_negative)
plt.title('6. Полный негатив (инверсия каналов)')
plt.axis('off')

plt.tight_layout()
plt.show()

'''АНАЛИЗ ГИСТОГРАММ ЦВЕТОВЫХ КАНАЛОВ'''
plt.figure(figsize=(15, 5))

'''Исходные гистограммы'''
plt.subplot(1, 3, 1)
plt.hist(r.flatten(), bins=50, color='red', alpha=0.7, label='Red')
plt.hist(g.flatten(), bins=50, color='green', alpha=0.7, label='Green') 
plt.hist(b.flatten(), bins=50, color='blue', alpha=0.7, label='Blue')
plt.title('Гистограммы исходного изображения')
plt.xlabel('Яркость')
plt.ylabel('Количество пикселей')
plt.legend()

'''Гистограммы после инверсии'''
plt.subplot(1, 3, 2)
r_inv, g_inv, b_inv = cv2.split(img_inverted)
plt.hist(r_inv.flatten(), bins=50, color='red', alpha=0.7, label='Red инв.')
plt.hist(g_inv.flatten(), bins=50, color='green', alpha=0.7, label='Green инв.')
plt.hist(b_inv.flatten(), bins=50, color='blue', alpha=0.7, label='Blue инв.')
plt.title('Гистограммы после инверсии')
plt.xlabel('Яркость')
plt.legend()

'''Сравнение распределений'''
plt.subplot(1, 3, 3)
plt.hist(img_rgb.flatten(), bins=50, color='gray', alpha=0.7, label='Исходное')
plt.hist(img_inverted.flatten(), bins=50, color='orange', alpha=0.7, label='Инверсия')
plt.title('Сравнение общих распределений')
plt.xlabel('Яркость')
plt.ylabel('Количество пикселей')
plt.legend()

plt.tight_layout()
plt.show()

'''ПРОВЕРКА НА ТИПИЧНЫЕ ПРОБЛЕМЫ С ЦВЕТАМИ'''


'''Проверяем распределение цветов'''
print(f"Средние значения по каналам:")
print(f"Красный: {np.mean(r):.1f}")
print(f"Зеленый: {np.mean(g):.1f}") 
print(f"Синий: {np.mean(b):.1f}")

'''Проверяем на доминирование определенного цвета'''
dominant_color = np.argmax([np.mean(r), np.mean(g), np.mean(b)])
colors = ['Красный', 'Зеленый', 'Синий']
print(f"Доминирующий цвет: {colors[dominant_color]}")

'''ТЕСТИРУЕМ ОБРАБОТКУ НА ИНВЕРТИРОВАННОМ ИЗОБРАЖЕНИИ'''

'''Берем лучший вариант инверсии и применяем обработку'''
img_to_process = img_inverted  # или img_brightness_inverted

'''Конвертируем в серое'''
img_gray_inverted = cv2.cvtColor(img_to_process, cv2.COLOR_RGB2GRAY)

'''Применяем эквализацию'''
img_eq_inverted = cv2.equalizeHist(img_gray_inverted)

'''Сглаживание + эквализация'''
img_smooth_eq_inverted = cv2.equalizeHist(cv2.GaussianBlur(img_gray_inverted, (3,3), 0))

'''Визуализация результатов обработки инвертированного изображения'''
plt.figure(figsize=(15, 5))

images_processed = [
    (img_gray_inverted, 'Инвертированное серое'),
    (img_eq_inverted, 'Инвертированное + Эквализация'),
    (img_smooth_eq_inverted, 'Инвертированное + Сглаживание + Эквализация')
]

for i, (image, title) in enumerate(images_processed):
    plt.subplot(1, 3, i+1)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()

'''Сравниваем гистограммы до и после инверсии'''
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).flatten(), bins=50, 
         alpha=0.7, label='Исходное серое', color='blue')
plt.hist(img_gray_inverted.flatten(), bins=50, 
         alpha=0.7, label='Инвертированное серое', color='red')
plt.title('Сравнение гистограмм яркости')
plt.xlabel('Яркость')
plt.ylabel('Количество пикселей')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(img_eq_inverted.flatten(), bins=50, alpha=0.7, color='green')
plt.title('Гистограмма после инверсии и эквализации')
plt.xlabel('Яркость')

plt.tight_layout()
plt.show()

## Программа, которая рисует отрезок между двумя точками, заданными пользователем

```python
import matplotlib.pyplot as plt  
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np


def draw_line(img, x0, y0, x1, y1, color):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        img.putpixel((x0, y0), color)
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def draw_grid(img, step, color):
    width, height = img.size
    # Вертикальные линии
    for x in range(0, width, step):
        draw_line(img, x, 0, x, height-1, color)
    # Горизонтальные линии
    for y in range(0, height, step):
        draw_line(img, 0, y, width-1, y, color)

x0 = int(input("Input first x0-coordinate: "))
y0 = int(input("Input first y0-coordinate: "))
x1 = int(input("Input second x1-coordinate: "))
y1 = int(input("Input second y1-coordinate: "))
img = Image.new('RGB', (1000, 900), 'white')
draw_grid(img, 50, (200, 200, 200))  
draw_line(img, x0, y0, x1, y1, (0, 0, 0))

imshow(np.asarray(img))
plt.show()
img.save('Linia.png')
```

## Программфа которая рисует окружность с заданным пользователем радиусом

```python
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def draw_circle_pixels(img, xc, yc, x, y, color):
    img.putpixel((xc + x, yc + y), color)
    img.putpixel((xc - x, yc + y), color)
    img.putpixel((xc + x, yc - y), color)
    img.putpixel((xc - x, yc - y), color)
    img.putpixel((xc + y, yc + x), color)
    img.putpixel((xc - y, yc + x), color)
    img.putpixel((xc + y, yc - x), color)
    img.putpixel((xc - y, yc - x), color)

# Алгоритм Брезенхема для окружности
def draw_circle(img, xc, yc, r, color):
    x = 0
    y = r
    d = 3 - 2 * r
    draw_circle_pixels(img, xc, yc, x, y, color)
    
    while y >= x:
        x += 1
        
        if d > 0:
            y -= 1
            d = d + 4 * (x - y) + 10
        else:
            d = d + 4 * x + 6
        
        draw_circle_pixels(img, xc, yc, x, y, color)

def draw_grid(img, step, color):
    width, height = img.size
    for x in range(0, width, step):
        draw_line(img, x, 0, x, height - 1, color)
    for y in range(0, height, step):
        draw_line(img, 0, y, width - 1, y, color)


r = int(input("Enter radius: "))

image_size = r * 2 + 20  # Добавляем по 10 пикселей отступа с каждой стороны

xc, yc = image_size // 2, image_size // 2

img = Image.new('RGB', (image_size, image_size), 'white')

draw_circle(img, xc, yc, r, (255, 0, 0))
plt.imshow(np.asarray(img))
plt.show()

img.save('Krug.png')
```
## Циферблат

```python
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math

def draw_circle_pixels(img, xc, yc, x, y, color):
    img.putpixel((xc + x, yc + y), color)
    img.putpixel((xc - x, yc + y), color)
    img.putpixel((xc + x, yc - y), color)
    img.putpixel((xc - x, yc - y), color)
    img.putpixel((xc + y, yc + x), color)
    img.putpixel((xc - y, yc + x), color)
    img.putpixel((xc + y, yc - x), color)
    img.putpixel((xc - y, yc - x), color)

def draw_circle(img, xc, yc, r, color):
    x = 0
    y = r
    d = 3 - 2 * r
    draw_circle_pixels(img, xc, yc, x, y, color)
    
    while y >= x:
        x += 1
        
        if d > 0:
            y -= 1
            d = d + 4 * (x - y) + 10
        else:
            d = d + 4 * x + 6
        
        draw_circle_pixels(img, xc, yc, x, y, color)

def draw_line(img, x0, y0, x1, y1, color):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        img.putpixel((x0, y0), color)
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

def draw_grid(img, step, color):
    width, height = img.size
    for x in range(0, width, step):
        draw_line(img, x, 0, x, height - 1, color)
    for y in range(0, height, step):
        draw_line(img, 0, y, width - 1, y, color)

def draw_ticks(img, xc, yc, r, num_ticks, color):
    for i in range(num_ticks):
        angle = 2 * math.pi * i / num_ticks
        x_end = int(xc + r * math.cos(angle))
        y_end = int(yc - r * math.sin(angle))  # Вычисляем точку по окружности
        x_start = int(xc + (r - 10) * math.cos(angle))  
        y_start = int(yc - (r - 10) * math.sin(angle))
        draw_line(img, x_start, y_start, x_end, y_end, color)

r = int(input("Enter radius: "))

image_size = r * 2 + 20  # Добавляем по 10 пикселей отступа с каждой стороны

xc, yc = image_size // 2, image_size // 2

img = Image.new('RGB', (image_size, image_size), 'white')

draw_grid(img, 50, (200, 200, 200))  

draw_circle(img, xc, yc, r, (255, 0, 0))  

# 12 засечек, как на часах
draw_ticks(img, xc, yc, r, 12, (0, 0, 0)) 

plt.imshow(np.asarray(img))
plt.show()

img.save('Clock.png')

``` 


## Реализация алгоритма Сезерленда-Коэна

```python
import matplotlib.pyplot as plt

INSIDE = 0  # 0000
LEFT = 1    # 0001
RIGHT = 2   # 0010
BOTTOM = 4  # 0100
TOP = 8     # 1000

def compute_code(x, y, x_min, y_min, x_max, y_max):
    code = INSIDE
    if x < x_min:    
        code |= LEFT
    elif x > x_max: 
        code |= RIGHT
    if y < y_min:    
        code |= BOTTOM
    elif y > y_max:
        code |= TOP
    return code


def cohen_sutherland_clip(x1, y1, x2, y2, x_min, y_min, x_max, y_max):
    code1 = compute_code(x1, y1, x_min, y_min, x_max, y_max)
    code2 = compute_code(x2, y2, x_min, y_min, x_max, y_max)
    accept = False

    while True:
        if code1 == 0 and code2 == 0:  
            accept = True
            break
        elif code1 & code2 != 0:  
            break
        else:
            x, y = 0.0, 0.0
            if code1 != 0:
                code_out = code1
            else:
                code_out = code2

            if code_out & TOP: 
                x = x1 + (x2 - x1) * (y_max - y1) / (y2 - y1)
                y = y_max
            elif code_out & BOTTOM: 
                x = x1 + (x2 - x1) * (y_min - y1) / (y2 - y1)
                y = y_min
            elif code_out & RIGHT:  
                y = y1 + (y2 - y1) * (x_max - x1) / (x2 - x1)
                x = x_max
            elif code_out & LEFT:  
                y = y1 + (y2 - y1) * (x_min - x1) / (x2 - x1)
                x = x_min

            if code_out == code1:
                x1, y1 = x, y
                code1 = compute_code(x1, y1, x_min, y_min, x_max, y_max)
            else:
                x2, y2 = x, y
                code2 = compute_code(x2, y2, x_min, y_min, x_max, y_max)

    if accept:
        return x1, y1, x2, y2
    else:
        return None

def draw_plot(lines, x_min, y_min, x_max, y_max):
    fig, ax = plt.subplots()

    ax.plot([x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min], 'k-', lw=2)

    for line in lines:
        x1, y1, x2, y2 = line
        ax.plot([x1, x2], [y1, y2], 'r--', label='Do otsecheniya')

    for line in lines:
        result = cohen_sutherland_clip(*line, x_min, y_min, x_max, y_max)
        if result:
            x1, y1, x2, y2 = result
            ax.plot([x1, x2], [y1, y2], 'g-', lw=2, label='Posle otsecheniya')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Otsechenie otrezkov algoritmom Sazerlenda-Koena')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    x_min, y_min = 10, 10
    x_max, y_max = 100, 100

    lines = [
        (5, 5, 120, 120),
        (50, 50, 60, 70),
        (70, 80, 120, 140),
        (10, 110, 110, 10),
        (0, 50, 200, 50)
    ]

    draw_plot(lines, x_min, y_min, x_max, y_max)
```

## Алгоритм Цирруса-Бека

```python
import numpy as np
import matplotlib.pyplot as plt

def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

def cyrus_beck_clip(line_start, line_end, polygon):
    d = np.array(line_end) - np.array(line_start)  
    t_enter = 0  
    t_exit = 1   

    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        edge = np.array(p2) - np.array(p1)
        normal = np.array([-edge[1], edge[0]]) 
        w = np.array(line_start) - np.array(p1)
        numerator = -dot_product(w, normal)
        denominator = dot_product(d, normal)

        if denominator != 0:
            t = numerator / denominator
            if denominator > 0: 
                t_enter = max(t_enter, t)
            else:  
                t_exit = min(t_exit, t)

            if t_enter > t_exit:
                return None  

    if t_enter <= t_exit:
        clipped_start = line_start + t_enter * d
        clipped_end = line_start + t_exit * d
        return clipped_start, clipped_end
    return None

def draw_plot(lines, polygon):
    fig, ax = plt.subplots()

        polygon.append(polygon[0])  
    polygon = np.array(polygon)
    ax.plot(polygon[:, 0], polygon[:, 1], 'k-', lw=2)


    for line in lines:
        line_start, line_end = line
        ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'r--', label='Do otsecheniya')

    for line in lines:
        result = cyrus_beck_clip(np.array(line[0]), np.array(line[1]), polygon[:-1].tolist())
        if result:
            clipped_start, clipped_end = result
            ax.plot([clipped_start[0], clipped_end[0]], [clipped_start[1], clipped_end[1]], 'g-', lw=2, label='Posle otsecheniya')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Otsechenie otrezkov algoritmom Cirrus-Beka')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    polygon = [
        [10, 10],
        [100, 30],
        [90, 100],
        [30, 90]
    ]


    lines = [
        ([0, 0], [50, 50]),
        ([20, 80], [80, 20]),
        ([60, 60], [120, 120]),
        ([0, 100], [100, 0]),
        ([70, 10], [70, 120])
    ]

    draw_plot(lines, polygon)

```
# Алгоритмы заполнения

## Алгоритм заполнения замкнутых областей посредством "затравки"

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def create_polygon_image(vertices, shape=(100, 100)):
    fig, ax = plt.subplots()
    fig.set_size_inches(shape[0] / fig.dpi, shape[1] / fig.dpi)
    ax.set_xlim(0, shape[1])
    ax.set_ylim(0, shape[0])
    ax.invert_yaxis()
    ax.axis('off')

    polygon = Polygon(vertices, closed=True, edgecolor='black', facecolor='white')
    ax.add_patch(polygon)

    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').reshape(shape[0], shape[1], 4)
    plt.close(fig)

    return image[:, :, :3].copy()

def is_background(color, threshold=68):
    return np.mean(color) > threshold

def boundary_fill(image, x, y, fill_color):
    if not is_background(image[x, y]):
        return

    stack = [(x, y)]

    while stack:
        cx, cy = stack.pop()
        if is_background(image[cx, cy]):
            image[cx, cy] = fill_color

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1] and is_background(image[nx, ny]):
                    stack.append((nx, ny))


vertices = [(30, 20), (70, 15), (90, 40), (80, 70), (50, 90), (20, 70), (10, 40)]
image = create_polygon_image(vertices)

fill_color = np.array([139, 0, 0], dtype=np.uint8)  
gray_threshold = 100
image[np.all((image[:, :, 0] < gray_threshold) & 
             (image[:, :, 1] < gray_threshold) & 
             (image[:, :, 2] < gray_threshold), axis=-1)] = [255, 255, 255]

plt.subplot(1, 2, 1)
plt.title("Исходное изображение")
plt.imshow(image)

boundary_fill(image, 50, 50, fill_color)
plt.subplot(1, 2, 2)
plt.title("После Boundary Fill")
plt.imshow(image)
plt.show()

```

## Алгоритм заполнения замкнутых областей посредством горизонтального сканирования

```python
import matplotlib.pyplot as plt
import numpy as np

def fill_polygon(vertices):
    x_min, x_max = min(vertices[:, 0]), max(vertices[:, 0])
    y_min, y_max = min(vertices[:, 1]), max(vertices[:, 1])
    
    fill_points = []

    for y in range(int(y_min), int(y_max) + 1):
        intersections = []
        for i in range(len(vertices)):
            v1, v2 = vertices[i], vertices[(i + 1) % len(vertices)]
            if (v1[1] > y) != (v2[1] > y):
                x = (v2[0] - v1[0]) * (y - v1[1]) / (v2[1] - v1[1]) + v1[0]
                intersections.append(x)
        
        intersections.sort()
     
        for i in range(0, len(intersections), 2):
            fill_points.append((intersections[i], y))
            fill_points.append((intersections[i + 1], y))
    
    return fill_points

vertices = np.array([(1, 1), (5, 0.5), (4, 4), (2, 3), (1, 4)])
fill_points = fill_polygon(vertices)

plt.fill(vertices[:, 0], vertices[:, 1], 'lightgrey')
plt.xlim(0, 6)
plt.ylim(0, 5)
plt.show()
```

## Модуль 2
### 2.1.1 Вращенип
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import imageio

# Параметры GIF
gif_filename = '11.gif'
frames = []
num_frames = 60  

def rotate(point, angle_x, angle_y, angle_z):
    # Углы поворота в радианах
    ax, ay, az = np.radians(angle_x), np.radians(angle_y), np.radians(angle_z)
    

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax), np.cos(ax)]])
    
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                   [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]])
    
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az), np.cos(az), 0],
                   [0, 0, 1]])
    
    
    rotated_point = Rz @ Ry @ Rx @ point
    return rotated_point


num_points = 50  # Количество случайных точек
points = np.random.uniform(-1, 1, (num_points, 3))


hull = ConvexHull(points)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = np.random.rand(len(hull.simplices), 3)
for i in range(num_frames):
    ax.clear()
    ax.set_title("Palanes")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    angle_x = i * 6
    angle_y = i * 3
    angle_z = i * 2

    rotated_points = np.array([rotate(p, angle_x, angle_y, angle_z) for p in points])
    for idx, simplex in enumerate(hull.simplices):
        triangle = rotated_points[simplex]
        ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                        color=colors[idx], edgecolor='k', alpha=0.8)
    plt.draw()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(image)

imageio.mimsave(gif_filename, frames, fps=15)
print(f'GIF сохранен в файл: {gif_filename}')
```

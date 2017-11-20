import random


def x(width, height):
    crop_width = random.randint(100, width)
    crop_height = random.randint(100, height)
    x = random.randint(0, width - crop_width -1)
    y = random.randint(0, height - crop_height -1)

    print(x, y, crop_width, crop_height)


for _ in range(300):
    x(3000, 2500)

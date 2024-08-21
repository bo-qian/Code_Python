import numpy as np
import matplotlib.pyplot as plt
import random


def generate_random_radius(min_radius, max_radius):
    return np.random.uniform(min_radius, max_radius)


def distance_between_points(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_tangent_point(x1, y1, r1, r2):
    distance = r1 + r2
    angle = np.random.uniform(0, 2 * np.pi)
    x2 = x1 + distance * np.cos(angle)
    y2 = y1 + distance * np.sin(angle)
    return x2, y2


def is_valid_circle(x, y, r, circles):
    for cx, cy, cr in circles:
        d = distance_between_points(x, y, cx, cy)
        if d <= r + cr:
            return False
    return True


def add_circle(circles, min_radius, max_radius, num_attempts=100):
    for _ in range(num_attempts):
        r = generate_random_radius(min_radius, max_radius)
        idx1, idx2 = random.sample(range(len(circles)), 2)
        x1, y1, r1 = circles[idx1]
        x2, y2, r2 = circles[idx2]

        r_new = generate_random_radius(min_radius, max_radius)
        dist1 = r_new + r1
        dist2 = r_new + r2

        angle1 = np.random.uniform(0, 2 * np.pi)
        angle2 = np.random.uniform(0, 2 * np.pi)

        x_new1 = x1 + dist1 * np.cos(angle1)
        y_new1 = y1 + dist1 * np.sin(angle1)

        x_new2 = x2 + dist2 * np.cos(angle2)
        y_new2 = y2 + dist2 * np.sin(angle2)

        if is_valid_circle(x_new1, y_new1, r_new, circles) and is_valid_circle(x_new2, y_new2, r_new, circles):
            return x_new1, y_new1, r_new

    return None


def plot_circles(circles):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    for (x, y, r) in circles:
        circle = plt.Circle((x, y), r, fill=False, edgecolor='r')
        ax.add_patch(circle)

    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.grid(True)
    plt.show()


def main(num_circles, min_radius, max_radius):
    circles = []

    # 画第一个圆
    r1 = generate_random_radius(min_radius, max_radius)
    circles.append((0, 0, r1))

    # 画第二个圆，与第一个圆外切
    r2 = generate_random_radius(min_radius, max_radius)
    x2, y2 = find_tangent_point(0, 0, r1, r2)
    circles.append((x2, y2, r2))

    # 画剩余的圆
    for _ in range(num_circles - 2):
        new_circle = add_circle(circles, min_radius, max_radius)
        if new_circle:
            circles.append(new_circle)
        else:
            print("无法找到有效的圆")
            break

    plot_circles(circles)


# 输入圆的数量，最小和最大半径
num_circles = 4
min_radius = 5
max_radius = 10

main(num_circles, min_radius, max_radius)

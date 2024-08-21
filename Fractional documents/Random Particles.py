import matplotlib.pyplot as plt
import numpy as np
import random
from sympy import symbols, Eq, solve, im

def plot_three_tangent_circles():
    # Step 1: Generate the first circle
    radius1 = random.uniform(1, 5)  # Random radius for the first circle
    x1, y1 = 0, 0  # The first circle's center at the origin

    # Step 2: Generate the second circle
    radius2 = random.uniform(1, 5)  # Random radius for the second circle
    distance_between_centers = radius1 + radius2  # The distance must be the sum of the radii for tangency

    # Random angle to place the second circle
    angle = random.uniform(0, 2 * np.pi)
    x2 = distance_between_centers * np.cos(angle)
    y2 = distance_between_centers * np.sin(angle)

    # Step 3: Generate the third circle that is tangent to the first two circles
    # Use the formula for the position of the third circle's center
    # The third circle must satisfy the condition for tangency with both circles
    radius3 = random.uniform(1, 5)  # Random radius for the third circle

    # Calculate the position of the third circle's center
    distance1_3 = radius1 + radius3
    distance2_3 = radius2 + radius3

    A = 2 * (x2 - x1)
    B = 2 * (y2 - y1)
    C = distance1_3 ** 2 - distance2_3 ** 2 - (x2 ** 2 - x1 ** 2) - (y2 ** 2 - y1 ** 2)

    x3 = (C / A + x1 + x2) / 2
    y3_pos = np.sqrt(distance1_3 ** 2 - (x3 - x1) ** 2) + y1
    y3_neg = -np.sqrt(distance1_3 ** 2 - (x3 - x1) ** 2) + y1

    # Check which y3 value satisfies the distance2_3 condition
    if np.isclose(np.sqrt((x3 - x2) ** 2 + (y3_pos - y2) ** 2), distance2_3):
        y3 = y3_pos
    else:
        y3 = y3_neg

    # Step 4: Plot the circles
    fig, ax = plt.subplots()
    circle1 = plt.Circle((x1, y1), radius1, color='blue', fill=False)
    circle2 = plt.Circle((x2, y2), radius2, color='red', fill=False)
    circle3 = plt.Circle((x3, y3), radius3, color='green', fill=False)
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)

    # Step 5: Draw the lines from the centers of the circles to the center of the tangent circle
    ax.plot([x1, x3], [y1, y3], 'g--')  # Green dashed line for the first circle to third circle
    ax.plot([x2, x3], [y2, y3], 'g--')  # Green dashed line for the second circle to third circle

    # Formatting the plot
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Three Tangent Circles')
    plt.grid(True)
    plt.show()

# Run the function to plot three tangent circles
# plot_three_tangent_circles()

def initial_three_particle(min_r, max_r):
    data = []
    coordinates = []
    radii = []
    sort_number = []
    for i in range(3):
        sort_number.append(i)
        radii.append(np.random.uniform(min_r, max_r))

    coordinates.append((0, 0))
    distance_between_centers = radii[0] + radii[1]
    angle = np.random.uniform(0, 2 * np.pi)
    coordinates.append((distance_between_centers * np.cos(angle), distance_between_centers * np.sin(angle)))

    distance_between_centers_0_1 = radii[0] + radii[2]
    distance_between_centers_1_2 = radii[1] + radii[2]

    x_2, y_2 = symbols('x_2 y_2')
    eq1 = Eq((x_2 - coordinates[0][0]) ** 2 + (y_2 - coordinates[0][1]) ** 2, distance_between_centers_0_1 ** 2)
    eq2 = Eq((x_2 - coordinates[1][0]) ** 2 + (y_2 - coordinates[1][1]) ** 2, distance_between_centers_1_2 ** 2)
    solution = solve((eq1, eq2), (x_2, y_2))
    real_solutions = [(sol_x, sol_y) for sol_x, sol_y in solution if im(sol_x) == 0 and im(sol_y) == 0]
    choose_option = np.random.choice([0, 1])
    coordinates.append(real_solutions[choose_option])

    return coordinates, radii, sort_number

def judgement_coordinates(initial_particle, coordinates, radii):
    while True:
        distance_between_centers = radii[initial_particle] + radii[coordinates[0]]
        x_n, y_n = symbols('x_n y_n')
        eq1 = Eq((x_n - coordinates[0][0]) ** 2 + (y_n - coordinates[0][1]) ** 2, distance_between_centers ** 2)
        solution = solve(eq1, (x_n, y_n))
        real_solutions = [(sol_x, sol_y) for sol_x, sol_y in solution if im(sol_x) == 0 and im(sol_y) == 0]
        if len(real_solutions) == 0:
            return False
        choose_option = np.random.choice([0, 1])
        coordinates.append(real_solutions[choose_option])
        return True








def keep_plot(min_r, max_, ini_coordinates, ini_radii, sort_num):
    # chosen_option = random.sample(sort_num, k=2)
    c = np.random.choice(sort_num)
    chosen_option = [c, c + 1]
    sort_now = len(ini_radii)
    sort_num.append(sort_now)
    ini_radii.append(np.random.uniform(min_r, max_))
    distance_between_centers_a = ini_radii[sort_now] + ini_radii[chosen_option[0]]
    distance_between_centers_b = ini_radii[sort_now] + ini_radii[chosen_option[1]]

    x_n, y_n = symbols('x_n y_n')
    eq1 = Eq((x_n - ini_coordinates[chosen_option[0]][0]) ** 2 + (y_n - ini_coordinates[chosen_option[0]][1]) ** 2, distance_between_centers_a ** 2)
    eq2 = Eq((x_n - ini_coordinates[chosen_option[1]][0]) ** 2 + (y_n - ini_coordinates[chosen_option[1]][1]) ** 2, distance_between_centers_b ** 2)
    solution = solve((eq1, eq2), (x_n, y_n))
    real_solutions = [(sol_x, sol_y) for sol_x, sol_y in solution if im(sol_x) == 0 and im(sol_y) == 0]
    print(f"sort_num: {sort_num}")
    print(f"ini_radii: {ini_radii}")
    for i in range(len(sort_num) - 1):
        choose_option = 0
        if (real_solutions[0][0] - ini_coordinates[i][0]) ** 2 + (real_solutions[0][0] - ini_coordinates[i][0]) ** 2 < ini_radii[i] + ini_radii[sort_now]:
            choose_option = 1
            break
    ini_coordinates.append(real_solutions[choose_option])
    print(f"ini_coordinates: {ini_coordinates}")

    return ini_coordinates, ini_radii, sort_num





def plot_random_circles(coordinates, radii, sort_num):
    fig, ax = plt.subplots()
    circle = []
    for i in range(len(coordinates)):
        circle.append(plt.Circle((coordinates[i][0], coordinates[i][1]), radii[i], fill=False))
        plt.text(coordinates[i][0], coordinates[i][1], f"{sort_num[i] + 1}", fontsize=12, ha='center', va='center')
        ax.add_artist(circle[i])

    ax.set_aspect('equal', 'box')
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Three Tangent Circles')
    plt.grid(True)
    plt.show()

particles_coordinates, particles_radii, particle_sort = initial_three_particle(10, 20)
for i in range(3):
    particles_coordinates, particles_radii, particle_sort = keep_plot(10, 20, particles_coordinates, particles_radii, particle_sort)
plot_random_circles(particles_coordinates, particles_radii, particle_sort)


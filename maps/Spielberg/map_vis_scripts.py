import os
import cv2
import yaml
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

module = os.path.join(os.path.dirname(os.path.abspath(__file__)))
print(module)
input_map_id = "0"
input_map_name = "Spielberg"


def generate_file_path(input_map_name):
    map_path = os.path.join(module, input_map_name + "_map.png")
    yaml_path = os.path.join(module, input_map_name + "_map.yaml")
    centerline_path = os.path.join(module, input_map_name + "_centerline.csv")
    return map_path, yaml_path, centerline_path


def pixel2meter(path, height, s, tx, ty):
    path = path.astype(float)
    new_path_x = path[:, 0] * s + tx
    new_path_y = (height - path[:, 1]) * s + ty
    if path.shape[1] > 2:
        new_right_dist = path[:, 2] * s
        new_left_dist = path[:, 3] * s
        return np.vstack((new_path_x, new_path_y, new_right_dist, new_left_dist)).T
    else:
        return np.vstack((new_path_x, new_path_y)).T


def meter2pixel(path, height, s, tx, ty):
    path = path.astype(float)
    new_path_x = (path[:, 0] - tx) / s
    new_path_y = height - (path[:, 1] - ty) / s
    if path.shape[1] > 2:
        new_right_dist = path[:, 2] / s
        new_left_dist = path[:, 3] / s
        return np.vstack((new_path_x, new_path_y, new_right_dist, new_left_dist)).T
    else:
        return np.vstack((new_path_x, new_path_y)).T


def read_centerline(input_map_id):
    input_map = "map" + input_map_id
    csv_file_path = os.path.join(module, input_map + ".csv")
    centerline = np.genfromtxt(csv_file_path, delimiter=",")
    return centerline

def draw_centerline(input_map_name):
    map_path, yaml_path, centerline_path = generate_file_path(input_map_name)
    with open(yaml_path, 'r') as stream:
        parsed_yaml = yaml.safe_load(stream)
    scale = parsed_yaml["resolution"]
    offset_x = parsed_yaml["origin"][0]
    offset_y = parsed_yaml["origin"][1]

    # read a color image
    input_img = cv2.imread(map_path, cv2.IMREAD_COLOR)
    h, w = input_img.shape[:2]
    print("Map Metadata: ", h, w, scale, offset_x, offset_y)
    centerline = np.genfromtxt(centerline_path, delimiter=",")
    centerline_xy = centerline[:, :2]
    centerline_pixel = meter2pixel(centerline_xy, h, scale, offset_x, offset_y)

    # imshow the centerline use dot and red color
    for i in range(len(centerline_pixel)):
        input_img[int(centerline_pixel[i][1]), int(centerline_pixel[i][0])] = (0, 0, 255)
    cv2.imshow("centerline", input_img)
    cv2.waitKey(0)

def centerline2frenet(input_map_id, centerline, wp_dist):
    input_map = "map_obs" + input_map_id
    x = centerline[:, 0]  # (n, )
    y = centerline[:, 1]
    span = centerline[0, 2]  # Assume constant width for the track
    # Ensure the last point is identical to the first point
    x[-1] = x[0]
    y[-1] = y[0]
    # Recompute distances and cumulative distances
    distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)  # (n-1, )
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)  # (n, )
    cs_x = CubicSpline(cumulative_distances, x, bc_type='periodic')
    cs_y = CubicSpline(cumulative_distances, y, bc_type='periodic')

    wp_num = round((cumulative_distances.max() - 0) / wp_dist) # number of waypoints
    s_dense = np.linspace(0, cumulative_distances.max(), wp_num)  # (wp_num, )
    x_dense = cs_x(s_dense)  # (wp_num, )
    y_dense = cs_y(s_dense)  # (wp_num, )
    psi = np.arctan2(np.gradient(y_dense), np.gradient(x_dense))
    # Unwrap psi to avoid discontinuity because psi belongs to [-pi, pi]
    psi_unwrapped = np.unwrap(psi)
    kappa = np.gradient(psi_unwrapped) / np.gradient(s_dense)
    w_tr_right = np.ones_like(s_dense) * span
    w_tr_left = np.ones_like(s_dense) * span
    # Create an array of the Frenet points
    frenet_points = np.vstack((s_dense, x_dense, y_dense, psi, kappa, w_tr_right, w_tr_left)).T
    # frenet_points = frenet_points[:-1]

    # Save the Frenet points to a file
    np.savetxt(f'./maps/{input_map}_frenet.csv', frenet_points, delimiter=',', fmt='%.8f',
               header='s_m,x_m,y_m,psi_rad,kappa_radpm,w_tr_right_m,w_tr_left_m')
    return frenet_points


if __name__ == "__main__":
    draw_centerline(input_map_name)

    # centerline = read_centerline(input_map_id)
    # centerline = centerline[::2]
    # frenet_points = centerline2frenet(input_map_id, centerline, 0.1)
    # # Draw the centerline and the frenet points
    # plt.figure(figsize=(10, 5))
    # plt.plot(centerline[:,0], centerline[:,1], label='Centerline', linewidth=2, color='blue')
    # plt.scatter(frenet_points[:,1], frenet_points[:,2], color='red', s=10, label='Frenet Points')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.title('Centerline vs. Frenet Transformation')
    # plt.legend()
    # plt.axis('equal')
    # plt.show()
    #
    # # Plot 2: Curvature (kappa) and Heading Angle (psi)
    # plt.figure(figsize=(10, 5))
    # Q = plt.quiver(frenet_points[:,1], frenet_points[:,2], np.cos(frenet_points[:,3]), np.sin(frenet_points[:,3]),
    #                frenet_points[:,4], scale=50, headwidth=3, headlength=5, width=0.0025, cmap='jet')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.title('Heading Angle (psi) and Curvature (kappa)')
    # plt.colorbar(Q, label='Curvature (kappa)')
    # plt.axis('equal')
    # plt.show()



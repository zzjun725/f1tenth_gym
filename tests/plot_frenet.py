import numpy as np
from f1tenth_gym.envs.track import Raceline, Track, find_track_dir
import matplotlib.pyplot as plt

# Test and plot
def test_and_plot():
    # Generate pseudo centerline points
    track = Track.from_track_name("Spielberg")
    centerline = np.array([track.centerline.xs, track.centerline.ys, track.centerline.yaws]).T

    # Plotting the centerline and interpolated Frenet points
    plt.figure(figsize=(10, 5))
    plt.plot(centerline[:,0], centerline[:,1], 'ro-', label='Centerline')

    # Generate some arbitrary Cartesian points to convert to Frenet
    cartesian_points = centerline[::50, :2]
    cartesian_points = cartesian_points + np.random.uniform(-0.5, 0.5, cartesian_points.shape)  # Every 10th point

    # Convert Cartesian points to Frenet and plot
    for p in cartesian_points:
        s, d, _ = track.cartesian_to_frenet(p[0], p[1], 0.0)
        plt.plot(p[0], p[1], '*', markersize=10, label=f'Original Cartesian to Frenet: s={s:.3f}, d={d:.3f}')
        x, y, _ = track.frenet_to_cartesian(s, d, 0.0)
        plt.plot(x, y, 'x', markersize=10)


    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Centerline, Frenet Points, and Cartesian to Frenet Conversion')
    plt.grid(True)
    plt.show()

test_and_plot()

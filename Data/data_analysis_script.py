# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Read CSVs
data_df = pd.read_csv('Hexapod data2025-05-22 16-34-52_laser_on_data_scan_right_labjack_0.12_0.0005_10.csv')
background_df = pd.read_csv('Hexapod data2025-05-23 11-00-39_background_data_scan_right_labjack_0.12_0.0005_10.csv')

# Merge on X and Y
merged_df = pd.merge(data_df, background_df, on=['x_index', 'y_index'], suffixes=('', '_bg'))
# Subtract background values
for v in ['v1', 'v2', 'v3', 'v4']:
    merged_df[v + '_corr'] = merged_df[v] - merged_df[v + '_bg']

# Compute V
V1 = merged_df['v1_corr']
V2 = merged_df['v2_corr']
V3 = merged_df['v3_corr']
V4 = merged_df['v4_corr']
# V1 = merged_df['v1']
# V2 = merged_df['v2']
# V3 = merged_df['v3']
# V4 = merged_df['v4']
# Square value chooses if squared or not
square_value = 2
merged_df['V'] = ((V1**square_value - V3**square_value) + (V2**square_value - V4**square_value)) / (V1 + V2 + V3 + V4)
merged_df['V_v'] = (V1**square_value - V3**square_value) / (V1 + V2 + V3 + V4)
merged_df['V_h'] =  (V2**square_value - V4**square_value) / (V1 + V2 + V3 + V4)
V1 = merged_df['v1_bg']
V2 = merged_df['v2_bg']
V3 = merged_df['v3_bg']
V4 = merged_df['v4_bg']
merged_df['v_background'] = ((V1**square_value - V3**square_value) + (V2**square_value - V4**square_value)) / (V1 + V2 + V3 + V4)
merged_df.to_csv('merged_output.csv', index=False)

# Get x_index, y_index, and V values
y_idx = merged_df['X']
x_idx = merged_df['Y']
v_vals = merged_df['V']

v_v_vals = merged_df['V_v']
v_h_vals = merged_df['V_h']

# Create custom colormap: Red at min/max, Blue at 0
custom_cmap = LinearSegmentedColormap.from_list(
    'red-blue-red',
    [(0.0, 'red'), (0.5, 'blue'), (1.0, 'red')]
)

# Symmetric color scale
vmax = np.nanmax(np.abs(v_vals))

# Plotting all points using scatter
plt.figure(figsize=(10, 8))
size = 2 # Scatter plot dot size
marker = 's'
# Scatter plot
scatter = plt.scatter(x_idx, y_idx, c=v_vals, cmap=custom_cmap, vmin=-vmax, vmax=vmax, s=size, marker = marker)
plt.colorbar(scatter, label='V Value')
plt.xlabel('hexapod Y position')
plt.ylabel('hexapod X position')
plt.title('Scatter Plot of V Values (Red at ±Max, Blue at 0)')
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()
# Scatter plot

# Symmetric color scale
vmax = np.nanmax(np.abs(v_v_vals))

# Plotting all points using scatter
plt.figure(figsize=(10, 8))
scatter = plt.scatter(x_idx, y_idx, c=v_v_vals, cmap=custom_cmap, vmin=-vmax, vmax=vmax, s=size, marker = marker)
plt.colorbar(scatter, label='V Value')
plt.xlabel('hexapod Y position')
plt.ylabel('hexapod X position')
plt.title('Scatter Plot of vertical V Values (Red at ±Max, Blue at 0)')
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()
# Scatter plot

# Symmetric color scale
vmax = np.nanmax(np.abs(v_h_vals))

# Plotting all points using scatter
plt.figure(figsize=(10, 8))

scatter = plt.scatter(x_idx, y_idx, c=v_h_vals, cmap=custom_cmap, vmin=-vmax, vmax=vmax, s=size, marker = marker)
plt.colorbar(scatter, label='V Value')
plt.xlabel('hexapod Y position')
plt.ylabel('hexapod X position')
plt.title('Scatter Plot of horizontal V Values (Red at ±Max, Blue at 0)')
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()


#%%

# Initalk ransac testing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

# --- Step 1: Filter near-zero V points ---
threshold = 0.75  # adjust if needed
v_zero = merged_df[np.abs(merged_df['V']) < threshold]

# --- Step 2: Apply bounding box ---
x_min = merged_df['X'].quantile(0.40)
x_max = merged_df['X'].quantile(0.75)
y_min = merged_df['Y'].quantile(0.24)
y_max = merged_df['Y'].quantile(0.6)

v_zero_box = v_zero[
    (v_zero['X'] >= x_min) & (v_zero['X'] <= x_max) &
    (v_zero['Y'] >= y_min) & (v_zero['Y'] <= y_max)
]

# --- Step 3: Prepare data ---
X = v_zero_box['X'].values.reshape(-1, 1)
Y = v_zero_box['Y'].values

# --- Step 4: Fit first line using RANSAC ---
ransac1 = RANSACRegressor()
ransac1.fit(X, Y)
inlier_mask1 = ransac1.inlier_mask_
outlier_mask1 = ~inlier_mask1

# Get slope and intercept
m1 = ransac1.estimator_.coef_[0]
c1 = ransac1.estimator_.intercept_

# --- Step 5: Fit second line on remaining points ---
X2 = X[outlier_mask1]
Y2 = Y[outlier_mask1]

ransac2 = RANSACRegressor()
ransac2.fit(X2, Y2)
inlier_mask2 = ransac2.inlier_mask_

m2 = ransac2.estimator_.coef_[0]
c2 = ransac2.estimator_.intercept_

# --- Step 6: Compute intersection ---
x_intersect = (c2 - c1) / (m1 - m2)
y_intersect = m1 * x_intersect + c1

print(f"Intersection at: X = {x_intersect:.4f}, Y = {y_intersect:.4f}")

# --- Step 7: Compute angle between lines ---
angle_rad = np.arctan(abs((m2 - m1) / (1 + m1 * m2)))
angle_deg = np.degrees(angle_rad)
print(f"Angle between slashes: {angle_deg:.2f} degrees")

# --- Step 8: Compute residuals ---
def point_line_residual(x, y, m, c):
    return np.abs(m * x - y + c) / np.sqrt(m**2 + 1)

residuals1 = point_line_residual(X[inlier_mask1].flatten(), Y[inlier_mask1], m1, c1)
residuals2 = point_line_residual(X2[inlier_mask2].flatten(), Y2[inlier_mask2], m2, c2)

print(f"Slash 1 Residuals → Mean: {np.mean(residuals1):.4f}, Std: {np.std(residuals1):.4f}")
print(f"Slash 2 Residuals → Mean: {np.mean(residuals2):.4f}, Std: {np.std(residuals2):.4f}")

# --- Step 9: Plot ---
plt.figure(figsize=(8, 8))
sc = plt.scatter(merged_df['X'], merged_df['Y'], c=merged_df['V'], cmap='seismic', s=1)
plt.colorbar(sc, label='V Value')

# Plot inliers
plt.scatter(X[inlier_mask1], Y[inlier_mask1], color='magenta', s=10, label='Slash 1 Points')
plt.scatter(X2[inlier_mask2], Y2[inlier_mask2], color='lime', s=10, label='Slash 2 Points')

# Plot fitted lines
x_range = np.linspace(x_min, x_max, 100)
plt.plot(x_range, m1 * x_range + c1, 'k--', label='Slash 1 Fit')
plt.plot(x_range, m2 * x_range + c2, 'k-.', label='Slash 2 Fit')

# Intersection point
plt.scatter(x_intersect, y_intersect, color='black', s=100, marker='X', label='Intersection')

# Bounding box
plt.plot([x_min, x_max, x_max, x_min, x_min],
         [y_min, y_min, y_max, y_max, y_min],
         color='lime', linestyle=':', linewidth=2, label='Bounding Box')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Intersection of Blue X Slashes with RANSAC Splitting')
plt.show()

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

# --- Step 1: Filter near-zero V points ---
threshold = 0.75
v_zero = merged_df[np.abs(merged_df['V']) < threshold]

# --- Step 2: Apply bounding box ---
x_min = merged_df['X'].quantile(0.40)
x_max = merged_df['X'].quantile(0.75)
y_min = merged_df['Y'].quantile(0.24)
y_max = merged_df['Y'].quantile(0.6)

v_zero_box = v_zero[
    (v_zero['X'] >= x_min) & (v_zero['X'] <= x_max) &
    (v_zero['Y'] >= y_min) & (v_zero['Y'] <= y_max)
]

# --- Step 3: Prepare data ---
X_all = v_zero_box['X'].values.reshape(-1, 1)
Y_all = v_zero_box['Y'].values

lines = []
remaining_mask = np.ones(len(X_all), dtype=bool)

# --- Step 4: Iteratively fit multiple lines ---
for i in range(4):  # try to extract up to 4 lines
    X = X_all[remaining_mask]
    Y = Y_all[remaining_mask]
    if len(X) < 10:
        break  # stop if too few points

    ransac = RANSACRegressor()
    ransac.fit(X, Y)
    inlier_mask = ransac.inlier_mask_

    m = ransac.estimator_.coef_[0]
    c = ransac.estimator_.intercept_

    global_inliers = np.zeros_like(remaining_mask, dtype=bool)
    global_inliers[remaining_mask] = inlier_mask
    lines.append({'m': m, 'c': c, 'inliers': global_inliers})

    # remove inliers for next round
    remaining_mask[remaining_mask] &= ~inlier_mask

print(f"Fitted {len(lines)} lines.")

# --- Step 5: Find most perpendicular pair ---
best_pair = None
best_angle_diff = 0

for i in range(len(lines)):
    for j in range(i+1, len(lines)):
        m1 = lines[i]['m']
        m2 = lines[j]['m']

        angle1 = np.arctan(m1)
        angle2 = np.arctan(m2)
        angle_diff = np.abs(angle1 - angle2) * (180 / np.pi)

        # adjust to be ≤ 90°
        if angle_diff > 90:
            angle_diff = 180 - angle_diff

        if np.abs(angle_diff - 90) < np.abs(best_angle_diff - 90):
            best_pair = (i, j)
            best_angle_diff = angle_diff

if best_pair is None:
    print("No perpendicular line pairs found.")
else:
    i, j = best_pair
    m1, c1 = lines[i]['m'], lines[i]['c']
    m2, c2 = lines[j]['m'], lines[j]['c']

    print(f"Best pair: Line {i} and Line {j} with angle {best_angle_diff:.2f}°")

    # --- Step 6: Compute intersection ---
    if np.isclose(m1, m2, atol=1e-6):
        print("Warning: Selected lines are parallel — no intersection.")
        x_intersect, y_intersect = np.nan, np.nan
    else:
        x_intersect = (c2 - c1) / (m1 - m2)
        y_intersect = m1 * x_intersect + c1
        print(f"Intersection at: X = {x_intersect:.4f}, Y = {y_intersect:.4f}")

    # --- Step 7: Plot ---
    plt.figure(figsize=(8, 8))
    sc = plt.scatter(merged_df['X'], merged_df['Y'], c=merged_df['V'], cmap='seismic', s=1)
    plt.colorbar(sc, label='V Value')

    # Plot inliers
    colors = ['magenta', 'lime', 'cyan', 'yellow']
    for idx, line in enumerate(lines):
        X_line = X_all[line['inliers']]
        Y_line = Y_all[line['inliers']]
        plt.scatter(X_line, Y_line, color=colors[idx % len(colors)], s=10, label=f'Line {idx} Points')

        # Plot fitted line
        x_range = np.linspace(x_min, x_max, 100)
        plt.plot(x_range, line['m'] * x_range + line['c'], linestyle='--', color=colors[idx % len(colors)],
                 label=f'Line {idx} Fit')

    # Plot intersection
    if not np.isnan(x_intersect):
        plt.scatter(x_intersect, y_intersect, color='black', s=100, marker='X', label='Intersection')

    # Bounding box
    plt.plot([x_min, x_max, x_max, x_min, x_min],
             [y_min, y_min, y_max, y_max, y_min],
             color='lime', linestyle=':', linewidth=2, label='Bounding Box')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Most Perpendicular Line Pair and Intersection')
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from sklearn.linear_model import RANSACRegressor

# --- CONFIGURATION ---
threshold_v = 0.75
x_qmin, x_qmax = 0.40, 0.75
y_qmin, y_qmax = 0.24, 0.6
angle_threshold = 1  # degrees from 90
max_attempts = 1000
output_figure = 'ransac_two_lines_fit.png'

# --- PREPARE DATA ---
v_zero = merged_df[np.abs(merged_df['V']) < threshold_v]
x_min = merged_df['X'].quantile(x_qmin)
x_max = merged_df['X'].quantile(x_qmax)
y_min = merged_df['Y'].quantile(y_qmin)
y_max = merged_df['Y'].quantile(y_qmax)
v_zero_box = v_zero[
    (v_zero['X'] >= x_min) & (v_zero['X'] <= x_max) &
    (v_zero['Y'] >= y_min) & (v_zero['Y'] <= y_max)
]

X_all = v_zero_box['X'].values.reshape(-1, 1)
Y_all = v_zero_box['Y'].values

# --- RUN LOOP ---
success = False
attempt = 0

while attempt < max_attempts:
    attempt += 1

    # Line 1
    ransac1 = RANSACRegressor(random_state=np.random.randint(10000))
    ransac1.fit(X_all, Y_all)
    inlier_mask1 = ransac1.inlier_mask_
    if np.sum(inlier_mask1) < 2:
        continue

    m1 = ransac1.estimator_.coef_[0]
    c1 = ransac1.estimator_.intercept_

    # Line 2 (fit on remaining points)
    X2 = X_all[~inlier_mask1]
    Y2 = Y_all[~inlier_mask1]
    if len(X2) < 2:
        continue

    ransac2 = RANSACRegressor(random_state=np.random.randint(10000))
    ransac2.fit(X2, Y2)
    inlier_mask2 = ransac2.inlier_mask_
    if np.sum(inlier_mask2) < 2:
        continue

    m2 = ransac2.estimator_.coef_[0]
    c2 = ransac2.estimator_.intercept_

    # Intersection point
    if m1 == m2:
        continue
    x_int = (c2 - c1) / (m1 - m2)
    y_int = m1 * x_int + c1

    # Compute all 4 angles
    angle_between = np.degrees(np.arctan(abs((m2 - m1) / (1 + m1 * m2))))
    angle = angle_between if angle_between <= 90 else 180 - angle_between
    angles = np.array([angle, 180 - angle, angle, 180 - angle])

    if np.all(np.abs(angles - 90) < angle_threshold):
        print(f"Attempt {attempt}: Angles = {np.round(angles, 2)} → PASS")
        break
    else:
        print(f"Attempt {attempt}: Angles = {np.round(angles, 2)} → FAIL")
else:
    print(f"\n❌ No successful fit after {max_attempts} attempts.")
    exit()

# --- PLOT ---
fig, ax = plt.subplots(figsize=(10, 10))
sc = ax.scatter(merged_df['X'], merged_df['Y'], c=merged_df['V'], cmap='seismic', s=1)
plt.colorbar(sc, label='V Value')

x_line = np.linspace(x_min, x_max, 100)
ax.plot(x_line, m1 * x_line + c1, color='magenta', linestyle='--', linewidth=2, label='Line 1')
ax.plot(x_line, m2 * x_line + c2, color='cyan', linestyle='--', linewidth=2, label='Line 2')
ax.scatter(x_int, y_int, color='black', s=150, marker='X', label=f'Intersection ({x_int:.2f}, {y_int:.2f})')

# Bounding box
ax.plot([x_min, x_max, x_max, x_min, x_min],
        [y_min, y_min, y_max, y_max, y_min],
        color='lime', linestyle=':', linewidth=2, label='Bounding Box')

# Zoom-in box
zoom_size = 0.1 * (x_max - x_min)
axins = ax.inset_axes([0.65, 0.05, 0.3, 0.3])
axins.scatter(X_all, Y_all, c='gray', s=1)
axins.plot(x_line, m1 * x_line + c1, color='magenta', linestyle='--')
axins.plot(x_line, m2 * x_line + c2, color='cyan', linestyle='--')
axins.scatter(x_int, y_int, color='black', s=50, marker='X')

# Add angle arcs and labels in zoomed inset (smaller size)
arc_radius = 0.002  # smaller radius for zoomed-in clarity

for i, ang in enumerate([angle, 180 - angle, angle, 180 - angle]):
    theta_start = i * 90
    theta_end = theta_start + ang
    arc = Arc((x_int, y_int),
              width=2 * arc_radius, height=2 * arc_radius,
              angle=0, theta1=theta_start, theta2=theta_end,
              color='red', linewidth=1.2)
    axins.add_patch(arc)

    # Label offset just beyond the arc
    theta_mid = np.radians(theta_start + ang / 2)
    x_offset = 1.5 * arc_radius * np.cos(theta_mid)
    y_offset = 1.5 * arc_radius * np.sin(theta_mid)
    axins.text(x_int + x_offset, y_int + y_offset, f"{ang:.1f}°",
               color='red', fontsize=7, ha='center', va='center')

# Inset box limits and border
axins.set_xlim(x_int - zoom_size, x_int + zoom_size)
axins.set_ylim(y_int - zoom_size, y_int + zoom_size)
axins.set_title('Zoomed Center')
axins.spines['bottom'].set_linewidth(1.5)
axins.spines['top'].set_linewidth(1.5)
axins.spines['left'].set_linewidth(1.5)
axins.spines['right'].set_linewidth(1.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.set_title(f'RANSAC Two-Line Fit with Intersection')

plt.savefig(output_figure, dpi=300)
plt.show()

print(f"\n✅ Saved figure to '{output_figure}'")
#%%
#Ransac, zoomed in plot included with zoomed out plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from sklearn.linear_model import RANSACRegressor

# --- CONFIGURATION ---
threshold_v = 0.75
x_qmin, x_qmax = 0.40, 0.75
y_qmin, y_qmax = 0.24, 0.6
angle_threshold = 1  # degrees from 90
max_attempts = 1000
output_figure = 'ransac_two_lines_fit.png'

# --- PREPARE DATA ---
v_zero = merged_df[np.abs(merged_df['V']) < threshold_v]
x_min = merged_df['X'].quantile(x_qmin)
x_max = merged_df['X'].quantile(x_qmax)
y_min = merged_df['Y'].quantile(y_qmin)
y_max = merged_df['Y'].quantile(y_qmax)
v_zero_box = v_zero[
    (v_zero['X'] >= x_min) & (v_zero['X'] <= x_max) &
    (v_zero['Y'] >= y_min) & (v_zero['Y'] <= y_max)
]

X_all = v_zero_box['X'].values.reshape(-1, 1)
Y_all = v_zero_box['Y'].values

# --- RUN LOOP ---
attempt = 0

while attempt < max_attempts:
    attempt += 1

    # Line 1
    ransac1 = RANSACRegressor(random_state=np.random.randint(10000))
    ransac1.fit(X_all, Y_all)
    inlier_mask1 = ransac1.inlier_mask_
    if np.sum(inlier_mask1) < 2:
        continue

    m1 = ransac1.estimator_.coef_[0]
    c1 = ransac1.estimator_.intercept_

    # Line 2 (fit on remaining points)
    X2 = X_all[~inlier_mask1]
    Y2 = Y_all[~inlier_mask1]
    if len(X2) < 2:
        continue

    ransac2 = RANSACRegressor(random_state=np.random.randint(10000))
    ransac2.fit(X2, Y2)
    inlier_mask2 = ransac2.inlier_mask_
    if np.sum(inlier_mask2) < 2:
        continue

    m2 = ransac2.estimator_.coef_[0]
    c2 = ransac2.estimator_.intercept_

    # Intersection point
    if m1 == m2:
        continue
    x_int = (c2 - c1) / (m1 - m2)
    y_int = m1 * x_int + c1

    # Compute slope angles (unit vectors)
    vec1 = np.array([1, m1]) / np.linalg.norm([1, m1])
    vec2 = np.array([1, m2]) / np.linalg.norm([1, m2])

    angle1 = np.degrees(np.arctan2(vec1[1], vec1[0])) % 360
    angle2 = np.degrees(np.arctan2(vec2[1], vec2[0])) % 360

    # Define the four arc spans between the lines
    arc_pairs = [
        (angle1, angle2),
        (angle2, (angle1 + 180) % 360),
        ((angle1 + 180) % 360, (angle2 + 180) % 360),
        ((angle2 + 180) % 360, angle1),
    ]

    arc_spans = [(end - start) % 360 for (start, end) in arc_pairs]
    arc_spans = [span if span <= 180 else 360 - span for span in arc_spans]

    # Check all spans against threshold
    if all(abs(span - 90) < angle_threshold for span in arc_spans):
        print(f"Attempt {attempt}: Arc spans = {np.round(arc_spans, 2)}° → PASS")
        break
    else:
        print(f"Attempt {attempt}: Arc spans = {np.round(arc_spans, 2)}° → FAIL")
else:
    print(f"\n❌ No successful fit after {max_attempts} attempts.")
    exit()

# --- PLOT ---
fig, ax = plt.subplots(figsize=(10, 10))
sc = ax.scatter(merged_df['X'], merged_df['Y'], c=merged_df['V'], cmap='seismic', s=1)
plt.colorbar(sc, label='V Value')

x_line = np.linspace(x_min, x_max, 100)
ax.plot(x_line, m1 * x_line + c1, color='magenta', linestyle='--', linewidth=2, label='Line 1')
ax.plot(x_line, m2 * x_line + c2, color='cyan', linestyle='--', linewidth=2, label='Line 2')
ax.scatter(x_int, y_int, color='black', s=150, marker='X', label=f'Intersection ({x_int:.2f}, {y_int:.2f})')

# Bounding box
ax.plot([x_min, x_max, x_max, x_min, x_min],
        [y_min, y_min, y_max, y_max, y_min],
        color='lime', linestyle=':', linewidth=2, label='Bounding Box')

# Zoom-in inset
zoom_size = 0.1 * (x_max - x_min)
axins = ax.inset_axes([0.65, 0.05, 0.3, 0.3])
axins.scatter(X_all, Y_all, c='gray', s=1)
axins.plot(x_line, m1 * x_line + c1, color='magenta', linestyle='--')
axins.plot(x_line, m2 * x_line + c2, color='cyan', linestyle='--')
axins.scatter(x_int, y_int, color='black', s=50, marker='X')

# Add arcs between slopes
arc_radius = 0.002
arc_colors = ['red', 'green', 'blue', 'orange']

for i, (start_angle, end_angle) in enumerate(arc_pairs):
    span = (end_angle - start_angle) % 360
    if span > 180:
        span = 360 - span
        start_angle, end_angle = end_angle, start_angle

    arc = Arc((x_int, y_int),
              width=2 * arc_radius, height=2 * arc_radius,
              angle=0, theta1=start_angle, theta2=end_angle,
              color=arc_colors[i], linewidth=1.5)
    axins.add_patch(arc)

    # Label midpoint
    theta_mid = np.radians((start_angle + span / 2) % 360)
    x_offset = 1.5 * arc_radius * np.cos(theta_mid)
    y_offset = 1.5 * arc_radius * np.sin(theta_mid)
    axins.text(x_int + x_offset, y_int + y_offset, f"{span:.1f}°",
               color=arc_colors[i], fontsize=10, ha='center', va='center')

# Inset box limits and border
axins.set_xlim(x_int - zoom_size, x_int + zoom_size)
axins.set_ylim(y_int - zoom_size, y_int + zoom_size)
axins.set_title('Zoomed Center')
for spine in axins.spines.values():
    spine.set_linewidth(1.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.set_title('RANSAC Two-Line Fit with Intersection')

plt.savefig(output_figure, dpi=300)
plt.show()

print(f"\n✅ Saved figure to '{output_figure}'")

for i, (start_angle, end_angle) in enumerate(spans):
    span = (end_angle - start_angle) % 360
    if span > 180:
        start_angle, end_angle = end_angle, start_angle
    arc = Arc((x_int, y_int),
              width=2 * arc_radius, height=2 * arc_radius,
              angle=0, theta1=start_angle, theta2=end_angle,
              color=arc_colors[i], linewidth=1.5)
    ax_zoom.add_patch(arc)

    # Label midpoint
    theta_mid = np.radians((start_angle + end_angle) / 2)
    x_offset = 1.5 * arc_radius * np.cos(theta_mid)
    y_offset = 1.5 * arc_radius * np.sin(theta_mid)
    ax_zoom.text(x_int + x_offset, y_int + y_offset, f"{angles_deg[i]:.1f}°",
                 color=arc_colors[i], fontsize=10, ha='center', va='center')

# Set zoom limits
zoom_size = 0.1 * (x_max - x_min)
ax_zoom.set_xlim(x_int - zoom_size, x_int + zoom_size)
ax_zoom.set_ylim(y_int - zoom_size, y_int + zoom_size)
ax_zoom.set_title('Intersection Detail')

for spine in ax_zoom.spines.values():
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.savefig(output_figure, dpi=300)
plt.show()

print(f"\n✅ Saved figure to '{output_figure}'")
#%%

#ransac  plot, subpltos for zoomed in and zoomed out
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from sklearn.linear_model import RANSACRegressor

# --- CONFIGURATION ---
threshold_v = 0.01
x_qmin, x_qmax = 0.4, 0.75
y_qmin, y_qmax = 0.28, 0.55
angle_threshold = 30 # degrees from 90
max_attempts = 10000
output_figure = 'ransac_two_lines_fit.png'

# --- PREPARE DATA ---
v_zero = merged_df[np.abs(merged_df['V']) < threshold_v]
x_min = merged_df['X'].quantile(x_qmin)
x_max = merged_df['X'].quantile(x_qmax)
y_min = merged_df['Y'].quantile(y_qmin)
y_max = merged_df['Y'].quantile(y_qmax)
v_zero_box = v_zero[
    (v_zero['X'] >= x_min) & (v_zero['X'] <= x_max) &
    (v_zero['Y'] >= y_min) & (v_zero['Y'] <= y_max)
]

X_all = v_zero_box['X'].values.reshape(-1, 1)
Y_all = v_zero_box['Y'].values


# --- RUN LOOP ---
attempt = 0
d_min = 0.001  # Minimum perpendicular distance to line 1 for second fit

while attempt < max_attempts:
    attempt += 1

    # Line 1
    ransac1 = RANSACRegressor(random_state=np.random.randint(10000))
    ransac1.fit(X_all, Y_all)
    inlier_mask1 = ransac1.inlier_mask_
    if np.sum(inlier_mask1) < 2:
        continue

    m1 = ransac1.estimator_.coef_[0]
    c1 = ransac1.estimator_.intercept_

    # Calculate perpendicular distance of remaining points to line 1
    X2_raw = X_all[~inlier_mask1]
    Y2_raw = Y_all[~inlier_mask1]
    if len(X2_raw) < 2:
        continue

    # Perpendicular distance formula
    dists = np.abs(m1 * X2_raw.flatten() - Y2_raw + c1) / np.sqrt(m1**2 + 1)
    far_mask = dists > d_min

    X2 = X2_raw[far_mask]
    Y2 = Y2_raw[far_mask]
    if len(X2) < 2:
        continue

    ransac2 = RANSACRegressor(random_state=np.random.randint(10000))
    ransac2.fit(X2, Y2)
    inlier_mask2 = ransac2.inlier_mask_
    if np.sum(inlier_mask2) < 2:
        continue

    m2 = ransac2.estimator_.coef_[0]
    c2 = ransac2.estimator_.intercept_

    # (… rest stays same: intersection, angle check, plotting …)


    # Intersection point
    if m1 == m2:
        continue
    x_int = (c2 - c1) / (m1 - m2)
    y_int = m1 * x_int + c1

    # --- Compute swept sector spans (same logic as plot) ---
    dir1 = np.array([1, m1]) / np.sqrt(1 + m1**2)
    dir2 = np.array([1, m2]) / np.sqrt(1 + m2**2)
    all_directions = [dir1, -dir1, dir2, -dir2]

    abs_angles = [np.degrees(np.arctan2(d[1], d[0])) % 360 for d in all_directions]
    abs_angles_sorted = np.sort(abs_angles)

    sector_spans = []
    for i in range(4):
        start = abs_angles_sorted[i]
        end = abs_angles_sorted[(i + 1) % 4]
        span = (end - start) % 360  # wrap around
        sector_spans.append(span)

    angles_deg = np.array(sector_spans)

    if np.all(np.abs(angles_deg - 90) < angle_threshold):
        print(f"Attempt {attempt}: Angles = {np.round(angles_deg, 2)} → PASS")
        break
    else:
        print(f"Attempt {attempt}: Angles = {np.round(angles_deg, 2)} → FAIL")
else:
    print(f"\n❌ No successful fit after {max_attempts} attempts.")
    exit()


# --- PLOT WITH THREE SUBPLOTS ---
fig, (ax_main, ax_zoom, ax_vzero) = plt.subplots(1, 3, figsize=(24, 8))

import matplotlib.colors as mcolors

# Define custom colormap
cmap = mcolors.LinearSegmentedColormap.from_list(
    'custom_u_shape',
    [(0, 'red'), (0.5, 'blue'), (1, 'red')]
)

# --- Main plot ---
sc = ax_main.scatter(merged_df['X'], merged_df['Y'], c=merged_df['V'],
                     cmap=cmap, vmin=-6, vmax=6, s=1)
plt.colorbar(sc, ax=ax_main, label='V Value')

ax_main.plot([x_min, x_max, x_max, x_min, x_min],
             [y_min, y_min, y_max, y_max, y_min],
             color='lime', linestyle=':', linewidth=2, label='Bounding Box')

ax_main.plot(x_line, m1 * x_line + c1, color='red', linestyle='-', linewidth=2, label='Line 1')
ax_main.plot(x_line, m2 * x_line + c2, color='blue', linestyle='-', linewidth=2, label='Line 2')
ax_main.scatter(X_all[inlier_mask1], Y_all[inlier_mask1], color='magenta', s=5, label='RANSAC Line 1 Inliers')
ax_main.scatter(X2[inlier_mask2], Y2[inlier_mask2], color='cyan', s=5, label='RANSAC Line 2 Inliers')
ax_main.scatter(x_int, y_int, color='black', s=150, marker='X', label=f'Intersection ({x_int:.2f}, {y_int:.2f})')

ax_main.set_xlabel('X')
ax_main.set_ylabel('Y')
ax_main.legend()
ax_main.set_title('Full Data with RANSAC Lines')

# --- Zoom plot ---
ax_zoom.scatter(X_all, Y_all, c='gray', s=1)
ax_zoom.plot(x_line, m1 * x_line + c1, color='magenta', linestyle='--')
ax_zoom.plot(x_line, m2 * x_line + c2, color='cyan', linestyle='--')
ax_zoom.scatter(x_int, y_int, color='black', s=50, marker='X')

arc_radius = 0.002
arc_colors = ['red', 'green', 'blue', 'orange']

for i in range(4):
    theta_start = abs_angles_sorted[i]
    span = angles_deg[i]
    theta_mid = (theta_start + span / 2) % 360

    arc = Arc((x_int, y_int),
              width=2 * arc_radius, height=2 * arc_radius,
              angle=0, theta1=theta_start, theta2=theta_start + span,
              color=arc_colors[i], linewidth=2)
    ax_zoom.add_patch(arc)

    theta_mid_rad = np.radians(theta_mid)
    x_offset = 1.5 * arc_radius * np.cos(theta_mid_rad)
    y_offset = 1.5 * arc_radius * np.sin(theta_mid_rad)

    ax_zoom.text(x_int + x_offset, y_int + y_offset,
                 f"{angles_deg[i]:.1f}°",
                 color=arc_colors[i], fontsize=12, fontweight='bold',
                 ha='center', va='center')

zoom_size = 0.1 * (x_max - x_min)
ax_zoom.set_xlim(x_int - zoom_size, x_int + zoom_size)
ax_zoom.set_ylim(y_int - zoom_size, y_int + zoom_size)
ax_zoom.set_title('Intersection Detail')

for spine in ax_zoom.spines.values():
    spine.set_linewidth(1.5)

# --- v_zero plot ---
ax_vzero.scatter(v_zero['X'], v_zero['Y'], c='gray', s=2, label='v_zero points')

ax_vzero.plot([x_min, x_max, x_max, x_min, x_min],
              [y_min, y_min, y_max, y_max, y_min],
              color='lime', linestyle='--', linewidth=2, label='Quantile Bounding Box')

ax_vzero.set_xlabel('X')
ax_vzero.set_ylabel('Y')
ax_vzero.set_title(f'v_zero Points (|V| < {threshold_v}) with Quantile Box')
ax_vzero.legend()
ax_vzero.grid(True)

# --- Finalize ---
plt.tight_layout()
plt.savefig(output_figure, dpi=300)
plt.show()

print(f"\n✅ Saved figure to '{output_figure}'")


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from sklearn.linear_model import RANSACRegressor

# CONFIGURATION
threshold_v = 0.8
x_qmin, x_qmax = 0.40, 0.75
y_qmin, y_qmax = 0.24, 0.6
angle_threshold = 1  # degrees from 90
max_attempts = 10000
output_figure = 'ransac_two_lines_fit.png'

# PREPARE DATA
v_zero = merged_df[np.abs(merged_df['V']) < threshold_v]
x_min = merged_df['X'].quantile(x_qmin)
x_max = merged_df['X'].quantile(x_qmax)
y_min = merged_df['Y'].quantile(y_qmin)
y_max = merged_df['Y'].quantile(y_qmax)
v_zero_box = v_zero[
    (v_zero['X'] >= x_min) & (v_zero['X'] <= x_max) &
    (v_zero['Y'] >= y_min) & (v_zero['Y'] <= y_max)
]

X_all = v_zero_box['X'].values.reshape(-1, 1)
Y_all = v_zero_box['Y'].values

# PRE-SEPARATE BY ROUGH SLOPE (APPROACH 1)
slope_est = (Y_all - np.mean(Y_all)) / (X_all.flatten() - np.mean(X_all))

pos_mask = slope_est > 0  # roughly /
neg_mask = slope_est < 0  # roughly \

X_pos, Y_pos = X_all[pos_mask], Y_all[pos_mask]
X_neg, Y_neg = X_all[neg_mask], Y_all[neg_mask]

# FIT POSITIVE SLOPE GROUP
ransac1 = RANSACRegressor(min_samples=2, random_state=0)
ransac1.fit(X_pos, Y_pos)
inlier_mask1 = ransac1.inlier_mask_
m1 = ransac1.estimator_.coef_[0]
c1 = ransac1.estimator_.intercept_

# FIT NEGATIVE SLOPE GROUP
ransac2 = RANSACRegressor(min_samples=2, random_state=1)
ransac2.fit(X_neg, Y_neg)
inlier_mask2 = ransac2.inlier_mask_
m2 = ransac2.estimator_.coef_[0]
c2 = ransac2.estimator_.intercept_

# INTERSECTION POINT
if m1 == m2:
    print("⚠ Lines are parallel!")
    x_int, y_int = np.nan, np.nan
else:
    x_int = (c2 - c1) / (m1 - m2)
    y_int = m1 * x_int + c1

# PLOT
fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(16, 8))
import matplotlib.colors as mcolors
cmap = mcolors.LinearSegmentedColormap.from_list('custom_u_shape', [(0, 'red'), (0.5, 'blue'), (1, 'red')])
sc = ax_main.scatter(merged_df['X'], merged_df['Y'], c=merged_df['V'], cmap=cmap, vmin=-6, vmax=6, s=1)
plt.colorbar(sc, ax=ax_main, label='V Value')

# Bounding box
ax_main.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], color='lime', linestyle=':', linewidth=2, label='Bounding Box')

x_line = np.linspace(x_min, x_max, 100)
ax_main.plot(x_line, m1 * x_line + c1, color='red', linestyle='-', linewidth=2, label='Line 1')
ax_main.plot(x_line, m2 * x_line + c2, color='blue', linestyle='-', linewidth=2, label='Line 2')
ax_main.scatter(X_pos[inlier_mask1], Y_pos[inlier_mask1], color='magenta', s=5, label='RANSAC Line 1 Inliers')
ax_main.scatter(X_neg[inlier_mask2], Y_neg[inlier_mask2], color='cyan', s=5, label='RANSAC Line 2 Inliers')
ax_main.scatter(x_int, y_int, color='black', s=150, marker='X', label=f'Intersection ({x_int:.2f}, {y_int:.2f})')

ax_main.set_xlabel('X')
ax_main.set_ylabel('Y')
ax_main.legend()
ax_main.set_title('Full Data with RANSAC Lines')

# ZOOM PLOT
ax_zoom.scatter(X_all, Y_all, c='gray', s=1)
ax_zoom.plot(x_line, m1 * x_line + c1, color='magenta', linestyle='--')
ax_zoom.plot(x_line, m2 * x_line + c2, color='cyan', linestyle='--')
ax_zoom.scatter(x_int, y_int, color='black', s=50, marker='X')

# ANGLE SECTORS
dir1 = np.array([1, m1]) / np.sqrt(1 + m1**2)
dir2 = np.array([1, m2]) / np.sqrt(1 + m2**2)
all_directions = [dir1, -dir1, dir2, -dir2]
abs_angles = [np.degrees(np.arctan2(d[1], d[0])) % 360 for d in all_directions]
abs_angles_sorted = np.sort(abs_angles)

sector_spans = []
for i in range(4):
    start = abs_angles_sorted[i]
    end = abs_angles_sorted[(i + 1) % 4]
    span = (end - start) % 360
    sector_spans.append(span)

arc_radius = 0.002
arc_colors = ['red', 'green', 'blue', 'orange']

for i in range(4):
    theta_start = abs_angles_sorted[i]
    span = sector_spans[i]
    theta_mid = (theta_start + span / 2) % 360
    arc = Arc((x_int, y_int), width=2 * arc_radius, height=2 * arc_radius, angle=0, theta1=theta_start, theta2=theta_start + span, color=arc_colors[i], linewidth=2)
    ax_zoom.add_patch(arc)
    theta_mid_rad = np.radians(theta_mid)
    x_offset = 1.5 * arc_radius * np.cos(theta_mid_rad)
    y_offset = 1.5 * arc_radius * np.sin(theta_mid_rad)
    ax_zoom.text(x_int + x_offset, y_int + y_offset, f"{span:.1f}°", color=arc_colors[i], fontsize=12, fontweight='bold', ha='center', va='center')

print(f"Total sector span: {np.sum(sector_spans):.6f}°")

zoom_size = 0.1 * (x_max - x_min)
ax_zoom.set_xlim(x_int - zoom_size, x_int + zoom_size)
ax_zoom.set_ylim(y_int - zoom_size, y_int + zoom_size)
ax_zoom.set_title('Intersection Detail')

for spine in ax_zoom.spines.values():
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.savefig(output_figure, dpi=300)
plt.show()

print(f"\n✅ Saved figure to '{output_figure}'")
#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely import affinity
import gdspy
import matplotlib.colors as mcolors

# === Load CSV Data ===
csv_file = 'merged_output.csv'
df = pd.read_csv(csv_file)
x_data = df['Y'].values  # swap X/Y if needed
y_data = df['X'].values
v_data = df['V'].values

# === Load GDS ===
gds_file = 'layout.gds'
gds_lib = gdspy.GdsLibrary(infile=gds_file)
top_cell = gds_lib.cells['filtered']  # update with correct top cell name

# Extract polygons
all_polys = []
for arr in top_cell.get_polygons():
    if arr.shape[0] >= 3:
        all_polys.append(Polygon(arr))

multipoly = MultiPolygon(all_polys)

# === Scale uniformly to match target square size ===
target_width_mm = 20.0e-3  # you set this
target_height_mm = 20.0e-3

inside_poly = max(all_polys, key=lambda p: p.area)  # or pick specific one
bounds = inside_poly.bounds
current_width_um = bounds[2] - bounds[0]
current_height_um = bounds[3] - bounds[1]
current_width_mm = current_width_um * 0.001
current_height_mm = current_height_um * 0.001

scale_x = target_width_mm / current_width_mm
scale_y = target_height_mm / current_height_mm
uniform_scale = min(scale_x, scale_y)

# Scale (include micron-to-mm conversion)
gds_to_mm = 0.001
multipoly_scaled = affinity.scale(
    multipoly, 
    xfact=gds_to_mm * uniform_scale, 
    yfact=gds_to_mm * uniform_scale, 
    origin='center'
)

# === Center on data ===
data_center_x = np.mean(x_data)
data_center_y = np.mean(y_data)

# Find the center polygon (largest area or specifically select)
center_poly = max(all_polys, key=lambda p: p.area)

# Scale that polygon same as the full multipoly
center_poly_scaled = affinity.scale(
    center_poly, 
    xfact=gds_to_mm * uniform_scale, 
    yfact=gds_to_mm * uniform_scale, 
    origin='center'
)

# Get its **bounding box center** before translation
bounds_scaled = center_poly_scaled.bounds
center_box_x = (bounds_scaled[0] + bounds_scaled[2]) / 2
center_box_y = (bounds_scaled[1] + bounds_scaled[3]) / 2

# Calculate offset to align center square's box center to data center
dx = data_center_x - center_box_x
dy = data_center_y - center_box_y

# Apply translation to entire multipolygon
aligned_multipoly = affinity.translate(multipoly_scaled, xoff=dx, yoff=dy)

# Apply translation to the center polygon
aligned_center_poly = affinity.translate(center_poly_scaled, xoff=dx, yoff=dy)
aligned_bounds = aligned_center_poly.bounds
aligned_center_x = (aligned_bounds[0] + aligned_bounds[2]) / 2
aligned_center_y = (aligned_bounds[1] + aligned_bounds[3]) / 2

# Print coordinates
print(f"📍 Center of middle square (aligned bounding box center): X = {aligned_center_x:.6f}, Y = {aligned_center_y:.6f} (hexapod coords)")

# === Manual offsets ===
manual_offset_x = -0.01  # mm, adjust as needed
manual_offset_y = 0.01   # mm, adjust as needed

aligned_multipoly = affinity.translate(
    aligned_multipoly,
    xoff=manual_offset_x,
    yoff=manual_offset_y
)

# Apply manual offset to center point too
final_center_x = aligned_center_x + manual_offset_x
final_center_y = aligned_center_y + manual_offset_y

# === Convert data points to Point objects ===
points = [Point(px, py) for px, py in zip(x_data, y_data)]

# === Integrate per polygon ===
total_sum = 0
print("\n🔍 Per-Polygon V Integrals:")
for idx, poly in enumerate(aligned_multipoly.geoms):
    inside_mask = np.array([poly.contains(pt) for pt in points])
    sum_v = np.sum(np.abs(v_data[inside_mask]))
    count_v = np.sum(inside_mask)
    print(f"  Polygon {idx + 1}: Sum V = {sum_v:.4f}, Points inside = {count_v}")
    total_sum += sum_v

print(f"\n✅ Total V Sum across all polygons: {total_sum:.4f}")

# === Plot ===
plt.figure(figsize=(10, 8))

abs_max = np.max(np.abs(v_data))

# Define custom colormap
cmap = mcolors.LinearSegmentedColormap.from_list(
    'custom_red_blue',
    [(0.0, 'red'),    # min (-abs_max)
     (0.5, 'blue'),  # zero
     (1.0, 'red')]   # max (+abs_max)
)

# Center normalization at zero
norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

# Plot scatter
sc = plt.scatter(x_data, y_data, c=v_data, cmap=cmap, norm=norm, s=1)
plt.colorbar(sc, label='V Value')

# Plot aligned polygons
for poly in aligned_multipoly.geoms:
    x, y = poly.exterior.xy
    plt.plot(x, y, color='lime', linewidth=2)

# Plot yellow dot at bounding box center
plt.scatter(final_center_x, final_center_y, color='yellow', s=50, label='Center Square (Box Center)')

plt.xlabel('X position (mm)')
plt.ylabel('Y position (mm)')
plt.title('Polygons Overlaid on Data with V Values')
plt.legend()
plt.show()

# %%



# %%
import phidl
import matplotlib.pyplot as plt

# Create device
D = phidl.Device('Layout')

# Center square (10x10)
center = D << phidl.geometry.rectangle(size=(18, 18))
center.center = (0, 0)

# Parameters
rect_width = 18
rect_length = 10
distance = 16
offset = 10/2 + distance + rect_length/2  # 20

# North rectangle
north = D << phidl.geometry.rectangle(size=(rect_width, rect_length))
north.center = (0, offset)

# South rectangle
south = D << phidl.geometry.rectangle(size=(rect_width, rect_length))
south.center = (0, -offset)

# East rectangle
east = D << phidl.geometry.rectangle(size=(rect_length, rect_width))
east.center = (offset, 0)

# West rectangle
west = D << phidl.geometry.rectangle(size=(rect_length, rect_width))
west.center = (-offset, 0)

# Plot
polygons = D.get_polygons()
plt.figure(figsize=(10, 10))

colors = ['blue', 'red', 'green', 'orange', 'purple']
for i, poly in enumerate(polygons):
    plt.fill(poly[:, 0], poly[:, 1], 
             color=colors[i], alpha=0.7, 
             edgecolor='black', linewidth=1)

plt.title('Center Square with 4 Rectangles')
plt.axis('equal')
plt.grid(True)
plt.show()

# Export
D.write_gds('layout.gds')
print("Layout saved as layout.gds")

# %%

import gdspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely import affinity
import matplotlib.colors as mcolors

# === First, let's check what's in the GDS file ===
gds_file = 'layout.gds'
gds_lib = gdspy.GdsLibrary(infile=gds_file)

print("Available cells in GDS file:")
for cell_name in gds_lib.cells.keys():
    print(f"  - '{cell_name}'")

# Get the first/main cell (or you can choose the correct one)
if gds_lib.cells:
    # Use the first cell or specify the correct one
    cell_names = list(gds_lib.cells.keys())
    top_cell_name = cell_names[0]  # Use first cell
    print(f"\nUsing cell: '{top_cell_name}'")
    top_cell = gds_lib.cells[top_cell_name]
else:
    print("No cells found in GDS file!")
    exit()

# === Load CSV Data ===
csv_file = 'merged_output.csv'
df = pd.read_csv(csv_file)
x_data = df['Y'].values  # swap X/Y if needed
y_data = df['X'].values
v_data = df['V'].values

# Extract polygons
all_polys = []
for arr in top_cell.get_polygons():
    if arr.shape[0] >= 3:
        all_polys.append(Polygon(arr))

if not all_polys:
    print("No polygons found in the cell!")
    exit()

print(f"\nFound {len(all_polys)} polygons in the cell")

multipoly = MultiPolygon(all_polys)

# === Scale uniformly to match target square size ===
target_width_mm = 20.0e-3  # you set this
target_height_mm = 20.0e-3

inside_poly = max(all_polys, key=lambda p: p.area)  # or pick specific one
bounds = inside_poly.bounds
current_width_um = bounds[2] - bounds[0]
current_height_um = bounds[3] - bounds[1]
current_width_mm = current_width_um * 0.001
current_height_mm = current_height_um * 0.001

scale_x = target_width_mm / current_width_mm
scale_y = target_height_mm / current_height_mm
uniform_scale = min(scale_x, scale_y)

print(f"\nScaling info:")
print(f"  Current size: {current_width_um:.1f} × {current_height_um:.1f} µm")
print(f"  Target size: {target_width_mm*1000:.1f} × {target_height_mm*1000:.1f} mm")
print(f"  Scale factor: {uniform_scale:.6f}")

# Scale (include micron-to-mm conversion)
gds_to_mm = 0.001
multipoly_scaled = affinity.scale(
    multipoly,
    xfact=gds_to_mm * uniform_scale,
    yfact=gds_to_mm * uniform_scale,
    origin='center')

# === Center on data ===
data_center_x = np.mean(x_data)
data_center_y = np.mean(y_data)

# Find the center polygon (largest area or specifically select)
center_poly = max(all_polys, key=lambda p: p.area)

# Scale that polygon same as the full multipolygon
center_poly_scaled = affinity.scale(
    center_poly,
    xfact=gds_to_mm * uniform_scale,
    yfact=gds_to_mm * uniform_scale,
    origin='center')

# Get its **bounding box center** before translation
bounds_scaled = center_poly_scaled.bounds
center_box_x = (bounds_scaled[0] + bounds_scaled[2]) / 2
center_box_y = (bounds_scaled[1] + bounds_scaled[3]) / 2

# Calculate offset to align center square's box center to data center
dx = data_center_x - center_box_x
dy = data_center_y - center_box_y

# Apply translation to entire multipolygon
aligned_multipoly = affinity.translate(multipoly_scaled, xoff=dx, yoff=dy)

# Apply translation to the center polygon
aligned_center_poly = affinity.translate(center_poly_scaled, xoff=dx, yoff=dy)
aligned_bounds = aligned_center_poly.bounds
aligned_center_x = (aligned_bounds[0] + aligned_bounds[2]) / 2
aligned_center_y = (aligned_bounds[1] + aligned_bounds[3]) / 2

# Print coordinates
print(f"\n📍 Center of middle square (aligned bounding box center): X = {aligned_center_x:.6f}, Y = {aligned_center_y:.6f} (hexapod coords)")

# === Manual offsets ===
manual_offset_x = -0.01  # mm, adjust as needed
manual_offset_y = 0.01   # mm, adjust as needed

aligned_multipoly = affinity.translate(
    aligned_multipoly,
    xoff=manual_offset_x,
    yoff=manual_offset_y)

# Apply manual offset to center point too
final_center_x = aligned_center_x + manual_offset_x
final_center_y = aligned_center_y + manual_offset_y

# === Convert data points to Point objects ===
points = [Point(px, py) for px, py in zip(x_data, y_data)]

# === Integrate per polygon ===
total_sum = 0
print("\n🔍 Per-Polygon V Integrals:")
for idx, poly in enumerate(aligned_multipoly.geoms):
    inside_mask = np.array([poly.contains(pt) for pt in points])
    sum_v = np.sum(np.abs(v_data[inside_mask]))
    count_v = np.sum(inside_mask)
    print(f"  Polygon {idx + 1}: Sum V = {sum_v:.4f}, Points inside = {count_v}")
    total_sum += sum_v

print(f"\n✅ Total V Sum across all polygons: {total_sum:.4f}")

# === Plot ===
plt.figure(figsize=(12, 10))
abs_max = np.max(np.abs(v_data))

# Define custom colormap
cmap = mcolors.LinearSegmentedColormap.from_list(
    'custom_red_blue',
    [(0.0, 'red'),    # min (-abs_max)
     (0.5, 'blue'),   # zero
     (1.0, 'red')])   # max (+abs_max)

# Center normalization at zero
norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

# Plot scatter
sc = plt.scatter(x_data, y_data, c=v_data, cmap=cmap, norm=norm, s=1)
plt.colorbar(sc, label='V Value')

# Plot aligned polygons with different colors
colors = ['lime', 'red', 'orange', 'purple', 'cyan']
for idx, poly in enumerate(aligned_multipoly.geoms):
    x, y = poly.exterior.xy
    plt.plot(x, y, color=colors[idx % len(colors)], linewidth=2, 
             label=f'Polygon {idx+1}')

# Plot yellow dot at bounding box center
plt.scatter(final_center_x, final_center_y, color='yellow', s=100, 
           label='Center Square (Box Center)', edgecolor='black', linewidth=2)

plt.xlabel('X position (mm)')
plt.ylabel('Y position (mm)')
plt.title(f'Polygons from "{top_cell_name}" Cell Overlaid on Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()

print(f"\n✅ Analysis complete using cell: '{top_cell_name}'")

























#%%


import gdspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely import affinity
import matplotlib.colors as mcolors

# === Load GDS file ===
gds_file = 'layout.gds'
gds_lib = gdspy.GdsLibrary(infile=gds_file)

print("Available cells in GDS file:")
for cell_name in gds_lib.cells.keys():
    print(f"  - '{cell_name}'")

# Use first available cell
cell_names = list(gds_lib.cells.keys())
top_cell_name = cell_names[0]
top_cell = gds_lib.cells[top_cell_name]
print(f"\nUsing cell: '{top_cell_name}'")

# === Load CSV Data ===
csv_file = 'merged_output.csv'
df = pd.read_csv(csv_file)
x_data = df['Y'].values
y_data = df['X'].values
v_data = df['V'].values

print(f"Data loaded: {len(x_data)} points")
print(f"Data range: X=[{x_data.min():.3f}, {x_data.max():.3f}], Y=[{y_data.min():.3f}, {y_data.max():.3f}]")

# === Extract and scale polygons ===
all_polys = []
for arr in top_cell.get_polygons():
    if arr.shape[0] >= 3:
        all_polys.append(Polygon(arr))

print(f"Found {len(all_polys)} polygons")
multipoly = MultiPolygon(all_polys)

# Scale to target size
target_width_mm = 20.0e-3
inside_poly = max(all_polys, key=lambda p: p.area)
bounds = inside_poly.bounds
current_width_um = bounds[2] - bounds[0]
uniform_scale = target_width_mm / (current_width_um * 0.001)

multipoly_scaled = affinity.scale(
    multipoly,
    xfact=0.001 * uniform_scale,
    yfact=0.001 * uniform_scale,
    origin='center')

print(f"Scaled from {current_width_um:.1f}μm to {target_width_mm*1000:.1f}mm")

# === Data center ===
data_center_x = np.mean(x_data)
data_center_y = np.mean(y_data)
print(f"Data center: ({data_center_x:.6f}, {data_center_y:.6f})")

# === Convert data to points (do once) ===
points = [Point(px, py) for px, py in zip(x_data, y_data)]

# === SCAN FOR OPTIMAL POSITION ===
scan_range = 1.0    # ±1mm around data center
scan_step = 0.5    # 0.05mm steps

x_offsets = np.arange(-scan_range, scan_range + scan_step, scan_step)
y_offsets = np.arange(-scan_range, scan_range + scan_step, scan_step)

print(f"\nScanning {len(x_offsets)} × {len(y_offsets)} = {len(x_offsets)*len(y_offsets)} positions")
print(f"Range: ±{scan_range}mm, Step: {scan_step}mm")

# Scan
best_sum = float('inf')
best_offset = None
scan_results = []

for x_off in x_offsets:
    for y_off in y_offsets:
        # Position polygons at data center + offset
        test_x = data_center_x + x_off
        test_y = data_center_y + y_off
        
        test_multipoly = affinity.translate(multipoly_scaled, xoff=test_x, yoff=test_y)
        
        # Calculate total integral
        total_sum = 0
        total_count = 0
        for poly in test_multipoly.geoms:
            inside_mask = np.array([poly.contains(pt) for pt in points])
            sum_v = np.sum(np.abs(v_data[inside_mask]))
            count_v = np.sum(inside_mask)
            total_sum += sum_v
            total_count += count_v
        
        scan_results.append([x_off, y_off, total_sum, total_count])
        
        if total_sum < best_sum:
            best_sum = total_sum
            best_offset = (x_off, y_off)

scan_results = np.array(scan_results)

print(f"\n✅ SCAN RESULTS:")
print(f"Best manual offset: ({best_offset[0]:.6f}, {best_offset[1]:.6f}) mm")
print(f"Minimum total integral: {best_sum:.4f}")
print(f"Maximum points inside: {int(scan_results[:, 3].max())}")
print(f"Positions with overlap: {int(np.sum(scan_results[:, 3] > 0))}")

# === Create optimal polygon position ===
optimal_multipoly = affinity.translate(multipoly_scaled, 
                                     xoff=data_center_x + best_offset[0], 
                                     yoff=data_center_y + best_offset[1])

# === Show per-polygon breakdown ===
print(f"\n🔍 Per-Polygon Integrals at Optimal Position:")
total_check = 0
for idx, poly in enumerate(optimal_multipoly.geoms):
    inside_mask = np.array([poly.contains(pt) for pt in points])
    sum_v = np.sum(np.abs(v_data[inside_mask]))
    count_v = np.sum(inside_mask)
    print(f"  Polygon {idx + 1}: Sum V = {sum_v:.4f}, Points inside = {count_v}")
    total_check += sum_v
print(f"Total check: {total_check:.4f}")

# === PLOT RESULTS ===
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Data with optimal polygons
abs_max = np.max(np.abs(v_data))
cmap = mcolors.LinearSegmentedColormap.from_list('custom', 
    [(0.0, 'red'), (0.5, 'blue'), (1.0, 'red')])
norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

sc1 = ax1.scatter(x_data, y_data, c=v_data, cmap=cmap, norm=norm, s=2)
plt.colorbar(sc1, ax=ax1, label='V Value')

colors = ['lime', 'red', 'orange', 'purple', 'cyan']
for idx, poly in enumerate(optimal_multipoly.geoms):
    x, y = poly.exterior.xy
    ax1.plot(x, y, color=colors[idx % len(colors)], linewidth=2, 
             label=f'Polygon {idx+1}')

ax1.scatter(data_center_x, data_center_y, color='yellow', s=100, 
           marker='o', label='Data Center')
ax1.scatter(data_center_x + best_offset[0], data_center_y + best_offset[1], 
           color='red', s=100, marker='x', label='Optimal Position')

ax1.set_xlabel('X position (mm)')
ax1.set_ylabel('Y position (mm)')
ax1.set_title('Data with Optimal Polygon Position')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# 2. Scan heatmap - Total Sum
if scan_results[:, 2].max() > 0:
    sc2 = ax2.scatter(scan_results[:, 0], scan_results[:, 1], 
                     c=scan_results[:, 2], cmap='viridis_r', s=15)
    plt.colorbar(sc2, ax=ax2, label='Total Sum (lower=better)')
else:
    ax2.scatter(scan_results[:, 0], scan_results[:, 1], c='blue', s=15)
    ax2.text(0.5, 0.5, 'All sums = 0\n(No overlap)', 
            transform=ax2.transAxes, ha='center', va='center', fontsize=12)

ax2.scatter(best_offset[0], best_offset[1], color='red', s=100, marker='x')
ax2.scatter(0, 0, color='yellow', s=100, marker='o', label='Data Center')
ax2.set_xlabel('Manual Offset X (mm)')
ax2.set_ylabel('Manual Offset Y (mm)')
ax2.set_title('Total Sum vs Manual Offset')
ax2.grid(True, alpha=0.3)

# 3. Points inside heatmap
sc3 = ax3.scatter(scan_results[:, 0], scan_results[:, 1], 
                 c=scan_results[:, 3], cmap='plasma', s=15)
plt.colorbar(sc3, ax=ax3, label='Points Inside')
ax3.scatter(best_offset[0], best_offset[1], color='red', s=100, marker='x')
ax3.set_xlabel('Manual Offset X (mm)')
ax3.set_ylabel('Manual Offset Y (mm)')
ax3.set_title('Points Inside vs Manual Offset')
ax3.grid(True, alpha=0.3)

# 4. Summary
ax4.axis('off')
summary_text = f"""OPTIMIZATION RESULTS:

Best Manual Offsets:
  manual_offset_x = {best_offset[0]:.6f}
  manual_offset_y = {best_offset[1]:.6f}

Performance:
  Minimum total integral: {best_sum:.4f}
  Maximum points inside: {int(scan_results[:, 3].max())}
  
Data Info:
  Total data points: {len(x_data)}
  Data center: ({data_center_x:.3f}, {data_center_y:.3f})
  
Scan Info:
  Positions scanned: {len(scan_results)}
  Positions with overlap: {int(np.sum(scan_results[:, 3] > 0))}
  Range: ±{scan_range}mm, Step: {scan_step}mm

GDS Info:
  Cell used: '{top_cell_name}'
  Polygons found: {len(all_polys)}
  Target size: {target_width_mm*1000:.1f}mm"""

ax4.text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top', 
         fontfamily='monospace', transform=ax4.transAxes)

plt.tight_layout()
plt.show()

print(f"\n🎯 USE THESE VALUES IN YOUR CODE:")
print(f"manual_offset_x = {best_offset[0]:.6f}  # mm")
print(f"manual_offset_y = {best_offset[1]:.6f}  # mm")

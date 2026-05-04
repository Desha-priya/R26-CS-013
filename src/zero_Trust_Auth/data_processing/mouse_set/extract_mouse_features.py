# extract_mouse_features.py
# BB-MAS exact columns:
#   Mouse_Button : EID, rX, rY, pX, pY, LR, state(0=press,1=release), time(datetime str)
#   Mouse_Move   : EID, rX, rY, pX, pY, time(datetime str)
#   Mouse_Wheel  : EID, rX, rY, pX, pY, delta, time(datetime str)

import pandas as pd
import numpy as np
import os
import glob

base_path  = r"BB-MAS_Dataset"
output_file = "user_mouse_profiles.csv"


def parse_time(df, col='time'):
    """Convert datetime string to float seconds from session start."""
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors='coerce')
    df = df.dropna(subset=[col]).sort_values(col).reset_index(drop=True)
    # Convert to seconds from first event
    df['t'] = (df[col] - df[col].iloc[0]).dt.total_seconds()
    return df


def extract_button_features(df):
    """
    Columns used: state (0=press, 1=release), pX, pY, LR, time
    Features:
      - click_duration   : time between press(0) and release(1) — equivalent to dwell_time
      - inter_click_time : time between consecutive press events — equivalent to flight_time
      - click_rate       : clicks per second
      - lr_ratio         : left vs right click ratio
      - double_click_rate: consecutive clicks under 300ms
    """
    if df.empty:
        return {}

    df = parse_time(df)
    features = {}

    press_df   = df[df['state'] == 0].reset_index(drop=True)
    release_df = df[df['state'] == 1].reset_index(drop=True)

    # Click duration (press → next release pairs)
    n = min(len(press_df), len(release_df))
    if n > 1:
        durations = release_df['t'].values[:n] - press_df['t'].values[:n]
        durations = durations[(durations > 0) & (durations < 5.0)]  # cap at 5s
        if len(durations) > 0:
            features['click_dur_mean'] = float(np.mean(durations))
            features['click_dur_std']  = float(np.std(durations))
            features['click_dur_min']  = float(np.min(durations))
            features['click_dur_max']  = float(np.max(durations))

    # Inter-click time (between consecutive presses)
    if len(press_df) > 1:
        inter = np.diff(press_df['t'].values)
        inter = inter[(inter > 0) & (inter < 60.0)]  # cap at 60s
        if len(inter) > 0:
            features['inter_click_mean'] = float(np.mean(inter))
            features['inter_click_std']  = float(np.std(inter))
            features['inter_click_min']  = float(np.min(inter))
            features['inter_click_max']  = float(np.max(inter))

        # Double-click rate: consecutive clicks < 300ms apart
        double = np.sum(inter < 0.3)
        features['double_click_rate'] = float(double / len(inter))

    # Click rate (clicks per second)
    total_time = df['t'].max()
    if total_time > 0:
        features['click_rate_per_sec'] = float(len(press_df) / total_time)
    features['total_clicks'] = int(len(press_df))

    # Left/right ratio (LR column: 0=left, 1=right typically)
    if 'LR' in df.columns:
        lr_counts = press_df['LR'].value_counts()
        left  = int(lr_counts.get(0, 0))
        right = int(lr_counts.get(1, 0))
        total_lr = left + right
        features['left_click_ratio'] = float(left / total_lr) if total_lr > 0 else 0.5

    return features


def extract_movement_features(df):
    """
    Columns used: pX, pY, rX, rY, time
    Features:
      - speed            : pixels/sec between consecutive points
      - acceleration     : rate of speed change
      - direction_change : angle change between consecutive vectors (measures jitter/smoothness)
      - path_efficiency  : straight-line / actual path (1.0 = perfectly straight)
      - movement_area    : bounding box of movement (work area size)
      - pause_rate       : fraction of time with near-zero movement
    """
    if df.empty or len(df) < 3:
        return {}

    df = parse_time(df)
    features = {}

    t  = df['t'].values
    px = df['pX'].values.astype(float)
    py = df['pY'].values.astype(float)

    dt   = np.diff(t)
    dx   = np.diff(px)
    dy   = np.diff(py)
    dist = np.sqrt(dx**2 + dy**2)

    valid = dt > 0
    dt_v, dist_v, dx_v, dy_v = dt[valid], dist[valid], dx[valid], dy[valid]

    # Speed
    if len(dt_v) > 0:
        speed = dist_v / dt_v
        features['move_speed_mean'] = float(np.mean(speed))
        features['move_speed_std']  = float(np.std(speed))
        features['move_speed_max']  = float(np.max(speed))

        # Acceleration
        if len(speed) > 1:
            accel = np.abs(np.diff(speed))
            features['move_accel_mean'] = float(np.mean(accel))
            features['move_accel_std']  = float(np.std(accel))

        # Pause rate: fraction of steps where speed < 5 px/s
        features['pause_rate'] = float(np.sum(speed < 5.0) / len(speed))

    # Direction change (curvature / jitter)
    if len(dx_v) > 1:
        angles     = np.arctan2(dy_v, dx_v)
        angle_diff = np.abs(np.diff(angles))
        # Wrap to [0, pi]
        angle_diff = np.where(angle_diff > np.pi, 2*np.pi - angle_diff, angle_diff)
        features['direction_change_mean'] = float(np.mean(angle_diff))
        features['direction_change_std']  = float(np.std(angle_diff))

    # Path efficiency
    total_path = float(np.sum(dist))
    if total_path > 0 and len(px) > 1:
        straight = float(np.sqrt((px[-1]-px[0])**2 + (py[-1]-py[0])**2))
        features['path_efficiency'] = float(straight / total_path)

    # Movement area (bounding box)
    features['move_range_x'] = float(np.max(px) - np.min(px))
    features['move_range_y'] = float(np.max(py) - np.min(py))

    # Also use rX, rY (relative deltas — direct from sensor, no accumulated error)
    rx = df['rX'].values.astype(float)
    ry = df['rY'].values.astype(float)
    rel_dist = np.sqrt(rx**2 + ry**2)
    features['rel_move_mean'] = float(np.mean(rel_dist))
    features['rel_move_std']  = float(np.std(rel_dist))

    return features


def extract_wheel_features(df):
    """
    Columns used: delta, pX, pY, time
    Features:
      - scroll_speed     : scroll events per second
      - scroll_magnitude : size of each scroll step
      - scroll_direction : ratio of downward scrolling
      - inter_scroll     : time between scroll events (rhythm)
      - scroll_bursts    : consecutive scrolls under 200ms (fast scrolling behaviour)
    """
    if df.empty or len(df) < 2:
        return {}

    df = parse_time(df)
    features = {}

    t      = df['t'].values
    delta  = df['delta'].values.astype(float)

    # Magnitude and direction
    mag = np.abs(delta)
    features['scroll_mag_mean']  = float(np.mean(mag))
    features['scroll_mag_std']   = float(np.std(mag))
    down  = int(np.sum(delta < 0))
    up    = int(np.sum(delta > 0))
    total = down + up
    features['scroll_down_ratio'] = float(down / total) if total > 0 else 0.5

    # Inter-scroll interval
    if len(t) > 1:
        inter = np.diff(t)
        inter = inter[inter > 0]
        if len(inter) > 0:
            features['inter_scroll_mean'] = float(np.mean(inter))
            features['inter_scroll_std']  = float(np.std(inter))

            # Burst rate: consecutive scrolls < 200ms
            bursts = np.sum(inter < 0.2)
            features['scroll_burst_rate'] = float(bursts / len(inter))

    # Scroll rate per second
    total_time = t[-1] - t[0]
    if total_time > 0:
        features['scroll_rate_per_sec'] = float(len(df) / total_time)
    features['total_scroll_events'] = int(len(df))

    return features


def process_user(user_id, user_folder):
    """Load all three mouse files for one user and combine features."""
    all_features = {'user': int(user_id)}

    file_map = {
        'Mouse_Button': extract_button_features,
        'Mouse_Move':   extract_movement_features,
        'Mouse_Wheel':  extract_wheel_features,
    }

    for file_type, extract_fn in file_map.items():
        pattern = os.path.join(user_folder, f"*_{file_type}.csv")
        files   = glob.glob(pattern)

        if not files:
            print(f"  [WARN] User {user_id}: no {file_type} file found")
            continue

        try:
            df = pd.read_csv(files[0])
            features = extract_fn(df)
            all_features.update(features)
        except Exception as e:
            print(f"  [ERROR] User {user_id} {file_type}: {e}")

    return all_features


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    user_folders = sorted(
        [f for f in os.listdir(base_path) if f.isdigit()],
        key=lambda x: int(x)
    )
    print(f"Found {len(user_folders)} users. Extracting mouse features...\n")

    results = []
    for uid in user_folders:
        folder = os.path.join(base_path, uid)
        print(f"  Processing user {uid}...")
        features = process_user(uid, folder)
        results.append(features)

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_file, index=False)

    print(f"\nDone! Saved to: {output_file}")
    print(f"Shape: {df_out.shape}")
    print(f"\nFeature columns ({len(df_out.columns)-1} features):")
    print([c for c in df_out.columns if c != 'user'])
    print(f"\nMissing values per column:")
    print(df_out.isnull().sum()[df_out.isnull().sum() > 0])
    print(f"\nSample — first 3 users:")
    print(df_out.head(3).to_string())
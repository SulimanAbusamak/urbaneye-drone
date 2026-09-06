import time
import json
import os
import math
import threading
from pymavlink import mavutil
import airsim
from object_avoidance_final import ForwardObstacleReader, handle_forward_avoidance
try:
    from llm_resolver import (
        parse_terminal_command,
        resolve_place_with_osm,
        resolve_saved_location,
        save_current_location,
        load_saved_locations,
    )
except ImportError:
    from llm_resolver import parse_terminal_command, resolve_place_with_osm

    def load_saved_locations(filepath):
        return {"locations": []}

    def resolve_saved_location(place_text, command=None, filepath=None):
        return {"ok": False, "error": "saved location support is not available in llm_resolver.py"}

    def save_current_location(name, pos, aliases=None, filepath=None):
        return {"ok": False, "error": "saved location support is not available in llm_resolver.py"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAVLINK_CONNECTION    = "udp:0.0.0.0:14551"
VEHICLE_NAME          = "PX4"
DISTANCE_SENSOR_NAME  = "Distance"

CRUISE_SPEED          = 5.0
LANDING_SPEED         = 7.0
DESIRED_AGL           = 50.0
HOVER_AGL             = 7.0
HOVER_DURATION        = 5
LAND_SETTLE_SEC       = 2.0
RELAUNCH_KICK_ALT     = 18.0
RELAUNCH_RECOVERY_SEC = 12.0
LAND_DONE_REL_ALT     = 0.35
LAND_DONE_AGL         = 2.5
ARRIVAL_RADIUS        = 5.0
ARRIVAL_CONFIRM_COUNT = 8
LOCATIONS_FILE        = "target_locations.json"
SAVED_LOCATIONS_FILE  = os.getenv("URBANEYE_SAVED_LOCATIONS_FILE", "saved_locations.json")
OFFBOARD_PRIME_SEC    = 5.0
MAX_NAV_TIME          = 3600

# fallback if AirSim distance sensor fails
FALLBACK_REL_ALT      = 50.0

# altitude control tuning
ALT_KP                = 1.00
ALT_KD                = 2.20
MAX_ALT_STEP          = 8.0
MIN_REL_ALT           = -250.0   # allow 50m AGL over terrain lower than home altitude
MAX_REL_ALT           = 300.0
TAKEOFF_MIN_REL_ALT   = 10.0     # startup/relaunch floor only; cruise can go below home altitude if AGL is valid
PARSER_TIMEOUT_SEC    = 8.0
RESOLVER_TIMEOUT_SEC  = 22.0
DISTANCE_SENSOR_MAX_TRUST = 195.0
GROUND_TRANSITION_STABLE_BAND = 8.0
GROUND_TRANSITION_CONFIRM_FRAMES = 8
LAST_GOOD_AGL_MAX_AGE = 1.0

# soft assist
LOW_AGL_BIAS_THRESHOLD = 47.0
CRITICAL_AGL_THRESHOLD = 35.0
EXTRA_CLIMB_BIAS       = 2.0
CRITICAL_CLIMB_BIAS    = 6.0


# anti-climb creep guard
ANTI_CLIMB_ENABLE         = True
ANTI_CLIMB_BAND           = 2.0
ANTI_CLIMB_CONFIRM_FRAMES = 8
ANTI_CLIMB_MAX_EXTRA_REL  = 2.0
ANTI_CLIMB_LOW_AGL_FRAMES = 6
ANTI_CLIMB_RESET_DIST_M   = 8.0

# yaw: align once when a destination is selected, then stop sending yaw updates during flight
YAW_ALIGN_ON_SELECTION      = True
YAW_ALIGN_RATE_DEG_PER_SEC  = 20.0
YAW_ALIGN_WAIT_SEC          = 10.0
YAW_ALIGN_MIN_TRAVEL_DIST_M = 8.0

# ---------------------------------------------------------------------------
# Forward distance obstacle avoidance config
# ---------------------------------------------------------------------------
FORWARD_OBSTACLE_ENABLE       = True
FORWARD_SENSOR_NAME           = "ForwardDistance"

# keep obstacle avoidance inactive until takeoff completes
FORWARD_ENABLE_AFTER_TAKEOFF  = True

# forward distance trigger
FORWARD_OBSTACLE_DISTANCE     = 50.0
FORWARD_MIN_VALID_DISTANCE    = 0.5
FORWARD_MAX_VALID_DISTANCE    = 120.0
FORWARD_PERSIST_FRAMES        = 4

# avoidance behavior
AVOIDANCE_BRAKE_SPEED       = 1.2
AVOIDANCE_HOLD_SPEED        = 0.6
AVOIDANCE_CLIMB_STEP_AGL    = 10.0
AVOIDANCE_MAX_EXTRA_AGL     = 80.0
AVOIDANCE_RECHECK_SEC       = 1.4
AVOIDANCE_SETTLE_SEC        = 1.2

# ---------------------------------------------------------------------------
# Hover-control command config
# These commands are only accepted after the drone has reached a destination
# and is holding/hovering. They are intentionally small and capped.
# ---------------------------------------------------------------------------
HOVER_CONTROL_ENABLE        = True
HOVER_CONTROL_SPEED         = 2.8
HOVER_CONTROL_YAW_RATE      = 12.0
HOVER_CONTROL_MAX_TURN_DEG  = 180.0
HOVER_CONTROL_MIN_TURN_DEG  = 1.0
HOVER_CONTROL_MAX_MOVE_M    = 200.0
HOVER_CONTROL_MIN_MOVE_M    = 1.0
HOVER_CONTROL_MAX_ALT_STEP  = 300.0
HOVER_CONTROL_MIN_AGL       = 3.0
HOVER_CONTROL_MAX_AGL       = 80.0
HOVER_CONTROL_MOVE_TIMEOUT  = 180.0
# Smooth hover-control tuning.
# A turn should hold ONE fixed GPS point, not keep updating the hold point.
# A move should crawl through short setpoints, then freeze at the actual final position
# instead of fighting for a perfect GPS point and oscillating.
HOVER_CONTROL_TURN_SETTLE_SEC = 1.2
HOVER_CONTROL_MOVE_LOOKAHEAD_M = 3.6
HOVER_CONTROL_MOVE_DONE_BAND_M = 1.2
HOVER_CONTROL_MOVE_SETTLE_SEC  = 1.8
# Movement should not keep descending while the drone is sliding forward/left/right.
# If the drone has not finished descending to HOVER_AGL yet, movement holds the
# current live AGL first. Altitude commands can still intentionally go down.
HOVER_CONTROL_MOVE_HOLD_LIVE_AGL_IF_HIGH = True
HOVER_CONTROL_MOVE_NO_DESCEND_REL        = True
HOVER_CONTROL_MOVE_PREHOLD_SEC           = 0.3
# Faster movement, but slow down near the end to reduce the final sway/overshoot.
HOVER_CONTROL_APPROACH_SPEED          = 1.1
HOVER_CONTROL_APPROACH_DIST_M         = 3.5

# v4 hover move fix: use small rolling position steps at ONE constant speed.
# This avoids the visible forward tilt + backward tilt caused by speeding up,
# then braking near the end of a short 10m command.
HOVER_CONTROL_MICRO_STEP_M            = 0.85
HOVER_CONTROL_MICRO_STEP_SEC          = 0.18
HOVER_CONTROL_MICRO_SPEED             = 1.9
HOVER_CONTROL_MICRO_DONE_BAND_M       = 0.45
HOVER_CONTROL_MICRO_SETTLE_SEC        = 1.2

# Extra stability mode:
# During hover-control, horizontal/turn/idle hover should freeze LOCAL_NED Z
# instead of continuously chasing tiny AirSim distance-sensor changes.
# This reduces visible micro up/down corrections while the drone is supposed to
# look still for the demo.
HOVER_FREEZE_REL_ALT_DURING_CONTROL   = True
HOVER_ALT_SETTLE_TIMEOUT              = 12.0
HOVER_ALT_SETTLE_BAND_M               = 0.45
HOVER_ALT_SETTLE_RATE_MPS             = 0.20
HOVER_ALT_SETTLE_CONFIRM_FRAMES       = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_locations(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["locations"]


def gps_to_distance(lat1, lon1, lat2, lon2):
    """Haversine distance in meters."""
    R = 6378137.0
    lat1r = math.radians(lat1)
    lon1r = math.radians(lon1)
    lat2r = math.radians(lat2)
    lon2r = math.radians(lon2)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r

    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def gps_to_bearing(lat1, lon1, lat2, lon2):
    """Bearing from point 1 to point 2 in degrees."""
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)

    y = math.sin(dlon) * math.cos(lat2r)
    x = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360


def gps_to_local_xy(home_lat, home_lon, lat, lon):
    """Convert GPS to local N/E meters using the original home as the local origin."""
    R = 6378137.0
    north = math.radians(lat - home_lat) * R
    east = math.radians(lon - home_lon) * R * math.cos(math.radians(home_lat))
    return north, east


def local_xy_to_gps(home_lat, home_lon, north_m, east_m):
    """Convert local N/E metres back to GPS using the same home origin.

    Hover-control movement uses LOCAL_NED metres internally because it is much
    less ambiguous than repeatedly converting tiny left/right/back GPS offsets.
    This helper lets the background streamer receive the same target as GPS.
    """
    R = 6378137.0
    lat = home_lat + math.degrees(float(north_m) / R)
    lon_scale = max(1e-9, math.cos(math.radians(home_lat)))
    lon = home_lon + math.degrees(float(east_m) / (R * lon_scale))
    return lat, lon


def clamp(value, low, high):
    return max(low, min(high, value))


def send_heartbeat(mav):
    mav.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0, 0, 0
    )


def send_position_target_local_ned(mav, north_m, east_m, rel_alt, yaw_deg=None):
    """
    Local-NED position target referenced to PX4 home.
    z in NED is positive down, so a 50 m relative altitude becomes z=-50.
    Yaw is sent in radians when provided so the drone faces the travel direction.
    """
    type_mask = 0b0000101111111000 if yaw_deg is not None else 0b0000111111111000
    yaw_rad = math.radians(yaw_deg % 360.0) if yaw_deg is not None else 0.0
    mav.mav.set_position_target_local_ned_send(
        0,
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        north_m,
        east_m,
        -rel_alt,
        0, 0, 0,
        0, 0, 0,
        yaw_rad,
        0
    )


def set_offboard_mode(mav):
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        6, 0, 0, 0, 0, 0
    )


def arm_drone(mav):
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 21196, 0, 0, 0, 0, 0
    )


def send_takeoff(mav, lat, lon, abs_alt):
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0,
        lat, lon, abs_alt
    )


def land(mav):
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0,
        0, 0, 0, 0, 0, 0, 0
    )


def disarm(mav):
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        0, 0, 0, 0, 0, 0, 0
    )


def set_speed(mav, speed):
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
        0,
        1, speed, -1, 0, 0, 0, 0
    )


def stream_yaw_relative_setpoint(mav, pos_reader, streamer, direction, degrees, hold_agl, yaw_rate_deg_s=HOVER_CONTROL_YAW_RATE):
    """Turn in hover by streaming a FIXED LOCAL_NED position + yaw setpoint.

    Important fix:
    The previous hover-yaw version kept updating the hold target to the drone's
    live position while turning. If the drone drifted 5m during yaw, the code
    accidentally accepted that drift as the new hold position. This version
    captures one fixed hover point before the turn and keeps commanding that same
    point until the yaw action finishes.

    PX4 in this setup rejected MAV_CMD_CONDITION_YAW with "command 115 unsupported",
    so we do not use command 115. We use the same supported setpoint-yaw method
    that already worked for yaw alignment before navigation.
    """
    pos = wait_for_fresh_position(pos_reader, timeout=2.0)
    if not pos:
        print("  [CONTROL] No fresh position for yaw turn.")
        return False

    current_heading = get_control_heading(pos, streamer)
    if current_heading is None:
        print("  [CONTROL] Current heading is not available yet. Try again in 1 second.")
        return False

    degrees = clamp(abs(float(degrees)), HOVER_CONTROL_MIN_TURN_DEG, HOVER_CONTROL_MAX_TURN_DEG)
    direction = str(direction).lower().strip()
    signed = degrees if direction == "right" else -degrees
    target_yaw = (current_heading + signed) % 360.0

    # Capture ONE fixed hold point. Do not update this point while yawing.
    hold_lat = pos["lat"]
    hold_lon = pos["lon"]
    hold_rel = pos["alt_rel"]
    hold_north, hold_east = gps_to_local_xy(streamer.home_lat, streamer.home_lon, hold_lat, hold_lon)

    wait_sec = clamp((degrees / max(yaw_rate_deg_s, 1.0)) + 1.5, 2.0, 18.0)
    print(f"  [CONTROL] Current yaw={current_heading:.0f}°. Target yaw={target_yaw:.0f}°.")

    # Keep the background streamer and the direct yaw setpoints asking for the
    # same fixed position. This prevents sideways drift during yaw.
    streamer.set_target(hold_lat, hold_lon, hold_rel, target_agl=None if HOVER_FREEZE_REL_ALT_DURING_CONTROL else hold_agl)

    t0 = time.time()
    tick = 0
    while time.time() - t0 < wait_sec:
        send_heartbeat(mav)
        send_position_target_local_ned(mav, hold_north, hold_east, hold_rel, yaw_deg=target_yaw)
        if tick % 10 == 0:
            set_offboard_mode(mav)
        tick += 1
        time.sleep(0.1)

    # Short fixed-position settle after yaw so PX4 does not accept any tiny drift
    # as the new target.
    settle_t0 = time.time()
    while time.time() - settle_t0 < HOVER_CONTROL_TURN_SETTLE_SEC:
        send_heartbeat(mav)
        send_position_target_local_ned(mav, hold_north, hold_east, hold_rel, yaw_deg=target_yaw)
        time.sleep(0.1)

    streamer.set_target(hold_lat, hold_lon, hold_rel, target_agl=None if HOVER_FREEZE_REL_ALT_DURING_CONTROL else hold_agl)
    # Save the commanded yaw so the next move command uses the yaw we just asked
    # PX4 to hold, instead of a stale/laggy GPS heading.
    streamer.hover_yaw_deg = target_yaw
    return True



def freeze_hover_at_current_position(pos_reader, streamer, target_agl=None, timeout=1.0):
    """Freeze the current hover point using current relative altitude.

    target_agl=None disables AGL correction. That is intentional for demo hover:
    the drone holds one LOCAL_NED Z value instead of making tiny vertical
    corrections from distance-sensor noise.
    """
    pos = wait_for_fresh_position(pos_reader, timeout=timeout)
    if not pos:
        return False

    streamer.set_target(pos["lat"], pos["lon"], pos["alt_rel"], target_agl=target_agl)
    return True


def wait_until_hover_altitude_settles(pos_reader, ground_reader, target_agl, timeout=HOVER_ALT_SETTLE_TIMEOUT):
    """Wait after an up/down command so the next left/right command does not inherit climb momentum."""
    start = time.time()
    last_agl = None
    last_time = None
    stable_frames = 0

    while time.time() - start < timeout:
        agl = get_believable_agl(pos_reader, ground_reader)
        now = time.time()

        if agl is None:
            time.sleep(0.1)
            continue

        rate = 0.0
        if last_agl is not None and last_time is not None:
            dt = max(0.001, now - last_time)
            rate = (agl - last_agl) / dt

        error = target_agl - agl
        if abs(error) <= HOVER_ALT_SETTLE_BAND_M and abs(rate) <= HOVER_ALT_SETTLE_RATE_MPS:
            stable_frames += 1
            if stable_frames >= HOVER_ALT_SETTLE_CONFIRM_FRAMES:
                return True
        else:
            stable_frames = 0

        last_agl = agl
        last_time = now
        time.sleep(0.1)

    return False

def execute_smooth_hover_move(direction, meters, pos_reader, streamer, ground_reader, mav_write, move_agl, resume_agl=None):
    """Move in hover using tiny rolling LOCAL_NED steps instead of one long target.

    v4 fix for short hover-control moves:
    The older smooth-move logic used a faster cruise speed and then slowed down
    near the end. In PX4 that looked like: tilt forward -> move -> tilt backward
    while braking. For a 10m demo movement, that sway is very visible.

    This version keeps ONE constant moderate speed and advances the commanded
    position in small steps along the intended line. It also freezes relative
    altitude during the horizontal move, so forward/right/left/back never become
    accidental descend commands.
    """
    start_pos = wait_for_fresh_position(pos_reader, timeout=2.0)
    if not start_pos:
        print("  [CONTROL] No fresh position for move.")
        return False

    heading = get_control_heading(start_pos, streamer)
    if heading is None:
        print("  [CONTROL] Heading/yaw not available yet. Move command ignored; try again in 1 second.")
        return False

    direction = str(direction).lower().strip()
    meters = clamp(abs(float(meters)), HOVER_CONTROL_MIN_MOVE_M, HOVER_CONTROL_MAX_MOVE_M)

    offset = {
        "forward": 0.0,
        "right": 90.0,
        "backward": 180.0,
        "left": -90.0,
    }.get(direction, 0.0)
    bearing = (heading + offset) % 360.0
    bearing_rad = math.radians(bearing)

    # Bearing convention: 0° = north, 90° = east.
    unit_n = math.cos(bearing_rad)
    unit_e = math.sin(bearing_rad)

    start_n, start_e = gps_to_local_xy(streamer.home_lat, streamer.home_lon, start_pos["lat"], start_pos["lon"])
    target_n = start_n + (unit_n * meters)
    target_e = start_e + (unit_e * meters)

    # Movement height is temporary. After the move, return to the original hover target.
    if resume_agl is None:
        resume_agl = move_agl

    print(
        f"  [CONTROL] Moving {direction} {meters:.1f}m using micro-steps "
        f"at constant speed {HOVER_CONTROL_MICRO_SPEED:.1f}m/s, "
        f"holding height during move, then resume {resume_agl:.1f}m AGL..."
    )

    # Constant speed: do not switch to a braking speed near the end. The small
    # step distance is what prevents overshoot/sway, not hard braking.
    set_speed(mav_write, HOVER_CONTROL_MICRO_SPEED)

    # Freeze EXACT relative altitude during horizontal motion.
    # Do not track pos_now["alt_rel"] upward, because if the previous "go up"
    # command still has climb momentum, the move command would inherit that climb
    # and keep raising the target while moving left/right.
    move_rel_floor = float(start_pos["alt_rel"])
    move_rel_cmd = move_rel_floor
    streamer.set_target(start_pos["lat"], start_pos["lon"], move_rel_cmd, target_agl=None)

    # Short pre-hold to start from a calm hover.
    pre_hold_t0 = time.time()
    while time.time() - pre_hold_t0 < HOVER_CONTROL_MOVE_PREHOLD_SEC:
        send_heartbeat(mav_write)
        send_position_target_local_ned(mav_write, start_n, start_e, move_rel_cmd, yaw_deg=heading)
        time.sleep(0.1)

    start_time = time.time()
    last_print = 0.0
    last_step_time = 0.0
    commanded_progress = 0.0
    tick = 0

    while time.time() - start_time < HOVER_CONTROL_MOVE_TIMEOUT:
        pos_now = pos_reader.get_position()
        if not pos_now:
            time.sleep(0.05)
            continue

        current_n, current_e = gps_to_local_xy(streamer.home_lat, streamer.home_lon, pos_now["lat"], pos_now["lon"])
        dn = current_n - start_n
        de = current_e - start_e

        progress = (dn * unit_n) + (de * unit_e)
        lateral_error = abs((-dn * unit_e) + (de * unit_n))
        remaining = max(0.0, meters - max(0.0, progress))
        dist_to_target = math.hypot(target_n - current_n, target_e - current_e)

        if progress >= meters - HOVER_CONTROL_MICRO_DONE_BAND_M or dist_to_target <= HOVER_CONTROL_MICRO_DONE_BAND_M:
            print("\n  [CONTROL] Move command done. Holding actual current position.")
            freeze_pos = wait_for_fresh_position(pos_reader, timeout=1.0) or pos_now
            freeze_n, freeze_e = gps_to_local_xy(streamer.home_lat, streamer.home_lon, freeze_pos["lat"], freeze_pos["lon"])
            freeze_rel = move_rel_floor

            # Set speed to the same constant speed, not a hard brake. Then hold
            # the actual position for a short settle before resuming AGL hold.
            set_speed(mav_write, HOVER_CONTROL_MICRO_SPEED)
            settle_t0 = time.time()
            while time.time() - settle_t0 < HOVER_CONTROL_MICRO_SETTLE_SEC:
                streamer.set_target(freeze_pos["lat"], freeze_pos["lon"], freeze_rel, target_agl=None)
                send_heartbeat(mav_write)
                send_position_target_local_ned(mav_write, freeze_n, freeze_e, freeze_rel, yaw_deg=heading)
                time.sleep(0.1)

            streamer.set_target(
                freeze_pos["lat"],
                freeze_pos["lon"],
                freeze_rel,
                target_agl=None if HOVER_FREEZE_REL_ALT_DURING_CONTROL else resume_agl
            )
            streamer.hover_yaw_deg = heading
            return True

        # Advance the commanded point in equal small steps. We keep the command
        # slightly ahead of actual progress, but never jump far ahead and never
        # switch to a different braking speed.
        now = time.time()
        if now - last_step_time >= HOVER_CONTROL_MICRO_STEP_SEC:
            next_step = max(progress + HOVER_CONTROL_MICRO_STEP_M, commanded_progress + HOVER_CONTROL_MICRO_STEP_M)
            commanded_progress = clamp(next_step, HOVER_CONTROL_MICRO_STEP_M, meters)
            last_step_time = now

        next_n = start_n + (unit_n * commanded_progress)
        next_e = start_e + (unit_e * commanded_progress)
        next_lat, next_lon = local_xy_to_gps(streamer.home_lat, streamer.home_lon, next_n, next_e)

        # Hold the exact starting relative altitude during the whole horizontal move.
        # This is the main fix for: go up -> go left -> drone keeps going up.
        move_rel_cmd = move_rel_floor

        streamer.set_target(next_lat, next_lon, move_rel_cmd, target_agl=None)
        send_heartbeat(mav_write)
        send_position_target_local_ned(mav_write, next_n, next_e, move_rel_cmd, yaw_deg=heading)
        if tick % 10 == 0:
            set_offboard_mode(mav_write)
        tick += 1

        if now - last_print >= 0.5:
            agl_live = get_believable_agl(pos_reader, ground_reader)
            agl_str = f"{agl_live:.1f}m" if agl_live is not None else "N/A"
            print(
                f"\r     [CONTROL] progress: {max(0.0, progress):4.1f}m / {meters:.1f}m"
                f" | cmd_step: {commanded_progress:4.1f}m"
                f" | remaining: {remaining:4.1f}m"
                f" | lateral_err: {lateral_error:3.1f}m"
                f" | agl={agl_str}   ",
                end="",
                flush=True
            )
            last_print = now

        time.sleep(0.05)

    print("\n  [CONTROL] Move timeout; holding actual current position.")
    pos_now = wait_for_fresh_position(pos_reader, timeout=1.0)
    if pos_now:
        freeze_rel = move_rel_floor
        streamer.set_target(
            pos_now["lat"],
            pos_now["lon"],
            freeze_rel,
            target_agl=None if HOVER_FREEZE_REL_ALT_DURING_CONTROL else resume_agl
        )
    streamer.hover_yaw_deg = heading
    return False

def gps_offset_by_bearing(lat, lon, bearing_deg, distance_m):
    """Move from one GPS coordinate by distance/bearing and return a new GPS point."""
    R = 6378137.0
    brng = math.radians(bearing_deg % 360.0)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    d = float(distance_m) / R

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d) +
        math.cos(lat1) * math.sin(d) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2)
    )

    return math.degrees(lat2), math.degrees(lon2)


def get_heading_from_position(pos):
    if not pos:
        return None
    # ATTITUDE yaw is usually the best source for hover-control. hdg from
    # GLOBAL_POSITION_INT can lag after an in-place yaw turn.
    yaw = pos.get("yaw")
    if yaw is not None:
        try:
            return float(yaw) % 360.0
        except Exception:
            pass
    hdg = pos.get("hdg")
    if hdg is None:
        return None
    try:
        return float(hdg) % 360.0
    except Exception:
        return None


def get_control_heading(pos, streamer=None):
    """Return the yaw to use for manual hover-control moves.

    Priority:
    1. the yaw we just commanded during hover-control rotation,
    2. live ATTITUDE yaw,
    3. GLOBAL_POSITION_INT heading fallback.
    """
    if streamer is not None:
        saved = getattr(streamer, "hover_yaw_deg", None)
        if saved is not None:
            try:
                return float(saved) % 360.0
            except Exception:
                pass
    return get_heading_from_position(pos)


def make_repeatable_hover_command(command):
    """Keep only the safe command fields needed for repeat-last-command.

    The LLM remains stateless. Repeat is controlled by the nav code memory.
    """
    if not command or command.get("type") != "control":
        return None
    if command.get("control_action") == "hover":
        return None

    allowed_keys = {
        "type", "control_action", "direction", "degrees", "meters",
        "mode", "agl", "parser"
    }
    saved = {k: v for k, v in command.items() if k in allowed_keys}
    saved["parser"] = saved.get("parser", "saved")
    return saved


def describe_hover_command(command):
    if not command:
        return "nothing"
    action = command.get("control_action")
    if action == "turn":
        return f"turn {command.get('direction', 'right')} {float(command.get('degrees', 15.0)):.0f}°"
    if action == "move":
        return f"move {command.get('direction', 'forward')} {float(command.get('meters', 5.0)):.1f}m"
    if action == "altitude":
        mode = command.get("mode", "up")
        if mode == "set":
            return f"set hover AGL to {float(command.get('agl', HOVER_AGL)):.1f}m"
        return f"altitude {mode} {float(command.get('meters', 5.0)):.1f}m"
    return str(action)

def execute_hover_control_command(command, pos_reader, streamer, ground_reader, mav_write, current_hover_agl=HOVER_AGL):
    """Execute small manual commands only while the drone is already hovering.

    This function intentionally avoids changing the main autonomous navigation logic.
    It only adjusts yaw, small local position offsets, or hover altitude after arrival.
    Returns the updated hover AGL target.
    """
    if not HOVER_CONTROL_ENABLE:
        print("  Hover-control commands are disabled.")
        return current_hover_agl

    pos = wait_for_fresh_position(pos_reader, timeout=2.0)
    if not pos:
        print("  [CONTROL] No fresh position. Command ignored.")
        return current_hover_agl

    action = command.get("control_action")

    agl_now = get_believable_agl(pos_reader, ground_reader)
    hold_agl = current_hover_agl if current_hover_agl is not None else (agl_now if agl_now is not None else HOVER_AGL)
    streamer.set_target(pos["lat"], pos["lon"], pos["alt_rel"], target_agl=hold_agl)

    if action == "turn":
        direction = command.get("direction", "right")
        degrees = clamp(abs(float(command.get("degrees", 15.0))), HOVER_CONTROL_MIN_TURN_DEG, HOVER_CONTROL_MAX_TURN_DEG)
        print(f"  [CONTROL] Turning {direction} {degrees:.0f}° while holding position...")
        ok = stream_yaw_relative_setpoint(
            mav_write, pos_reader, streamer, direction, degrees, hold_agl,
            yaw_rate_deg_s=HOVER_CONTROL_YAW_RATE
        )
        if ok:
            print("  [CONTROL] Turn command done.")
        else:
            print("  [CONTROL] Turn command failed/ignored.")
        return hold_agl

    if action == "move":
        direction = command.get("direction", "forward")
        meters = clamp(abs(float(command.get("meters", 5.0))), HOVER_CONTROL_MIN_MOVE_M, HOVER_CONTROL_MAX_MOVE_M)

        # If arrival hover is still a little high, do not let a horizontal move
        # continue that descent. Use the live height only temporarily during the
        # move, then resume the real hover target afterward.
        move_hold_agl = hold_agl
        if HOVER_CONTROL_MOVE_HOLD_LIVE_AGL_IF_HIGH and agl_now is not None:
            if agl_now > hold_agl + 2.0:
                move_hold_agl = clamp(agl_now, HOVER_CONTROL_MIN_AGL, HOVER_CONTROL_MAX_AGL)
                print(f"  [CONTROL] Temporarily holding current height during move: {move_hold_agl:.1f}m AGL")

        ok = execute_smooth_hover_move(
            direction, meters,
            pos_reader, streamer, ground_reader, mav_write,
            move_hold_agl,
            resume_agl=hold_agl
        )
        if not ok:
            print("  [CONTROL] Smooth move failed/held current position.")
        # IMPORTANT: do not save the temporary movement height as the new hover target.
        return hold_agl

    if action == "altitude":
        mode = command.get("mode", "up")
        agl_now = get_believable_agl(pos_reader, ground_reader)
        if agl_now is None:
            agl_now = hold_agl

        if mode == "set":
            target_agl = float(command.get("agl", hold_agl))
        else:
            meters = clamp(abs(float(command.get("meters", 5.0))), 1.0, HOVER_CONTROL_MAX_ALT_STEP)
            target_agl = agl_now + meters if mode == "up" else agl_now - meters

        target_agl = clamp(target_agl, HOVER_CONTROL_MIN_AGL, HOVER_CONTROL_MAX_AGL)
        rel_target = pos["alt_rel"] + (target_agl - agl_now)
        rel_target = clamp(rel_target, MIN_REL_ALT, MAX_REL_ALT)
        print(f"  [CONTROL] Changing hover target to {target_agl:.1f}m AGL...")
        streamer.set_target(pos["lat"], pos["lon"], rel_target, target_agl=target_agl)

        settled = wait_until_hover_altitude_settles(pos_reader, ground_reader, target_agl)
        if settled:
            print("  [CONTROL] Altitude settled. Freezing hover height.")
        else:
            print("  [CONTROL] Altitude settle timeout. Freezing current height anyway.")

        if HOVER_FREEZE_REL_ALT_DURING_CONTROL:
            freeze_hover_at_current_position(pos_reader, streamer, target_agl=None)
        else:
            freeze_hover_at_current_position(pos_reader, streamer, target_agl=target_agl)
        return target_agl

    if action == "hover":
        print(f"  [CONTROL] Holding current position at {hold_agl:.1f}m AGL.")
        streamer.set_target(
            pos["lat"], pos["lon"], pos["alt_rel"],
            target_agl=None if HOVER_FREEZE_REL_ALT_DURING_CONTROL else hold_agl
        )
        return hold_agl

    print(f"  [CONTROL] Unsupported control action: {action}")
    return hold_agl


def align_yaw_once_before_flight(mav_write, current_pos, target_gps, home_lat, home_lon):
    if not YAW_ALIGN_ON_SELECTION or not current_pos or not target_gps:
        return
    try:
        dist = gps_to_distance(current_pos["lat"], current_pos["lon"], target_gps["lat"], target_gps["lon"])
        if dist < YAW_ALIGN_MIN_TRAVEL_DIST_M:
            return

        bearing = gps_to_bearing(current_pos["lat"], current_pos["lon"], target_gps["lat"], target_gps["lon"])
        print(f"     Rotating in place to {bearing:.0f}° for {YAW_ALIGN_WAIT_SEC:.0f}s...")

        north, east = gps_to_local_xy(home_lat, home_lon, current_pos["lat"], current_pos["lon"])
        rel_alt = current_pos["alt_rel"]
        t0 = time.time()
        tick = 0
        while time.time() - t0 < YAW_ALIGN_WAIT_SEC:
            send_heartbeat(mav_write)
            send_position_target_local_ned(mav_write, north, east, rel_alt, yaw_deg=bearing)
            if tick % 10 == 0:
                set_offboard_mode(mav_write)
            tick += 1
            time.sleep(0.1)
    except Exception as e:
        print(f"     [YAW ALIGN WARNING] {e}")


# ---------------------------------------------------------------------------
# PX4 position reader
# ---------------------------------------------------------------------------
class PositionReader(threading.Thread):
    def __init__(self, mav):
        super().__init__(daemon=True)
        self.mav = mav
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pos = None
        self._last_time = 0.0
        self._att_yaw_deg = None

    def get_position(self):
        with self._lock:
            return self._pos.copy() if self._pos else None

    def has_fresh_position(self, max_age=1.0):
        with self._lock:
            if self._pos is None:
                return False
            return (time.time() - self._last_time) <= max_age

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                msg = self.mav.recv_match(
                    type=["GLOBAL_POSITION_INT", "ATTITUDE"],
                    blocking=True,
                    timeout=0.5
                )

                if not msg:
                    continue

                msg_type = msg.get_type()

                if msg_type == "ATTITUDE":
                    try:
                        yaw_deg = (math.degrees(float(msg.yaw)) + 360.0) % 360.0
                    except Exception:
                        yaw_deg = None
                    if yaw_deg is not None:
                        with self._lock:
                            self._att_yaw_deg = yaw_deg
                            if self._pos is not None:
                                self._pos["yaw"] = yaw_deg
                    continue

                if msg_type == "GLOBAL_POSITION_INT" and msg.lat != 0:
                    with self._lock:
                        heading = None
                        try:
                            if hasattr(msg, "hdg") and msg.hdg is not None and msg.hdg != 65535:
                                heading = (msg.hdg / 100.0) % 360.0
                        except Exception:
                            heading = None

                        self._pos = {
                            "lat": msg.lat / 1e7,
                            "lon": msg.lon / 1e7,
                            "alt_rel": msg.relative_alt / 1000.0,
                            "alt_msl": msg.alt / 1000.0,
                            "hdg": heading,
                            "yaw": self._att_yaw_deg,
                        }
                        self._last_time = time.time()

            except Exception as e:
                print(f"\n[POSITION ERROR] {e}")


# ---------------------------------------------------------------------------
# AirSim ground distance reader
# ---------------------------------------------------------------------------
class AirSimGroundReader(threading.Thread):
    def __init__(self, vehicle_name=VEHICLE_NAME, sensor_name=DISTANCE_SENSOR_NAME):
        super().__init__(daemon=True)
        self.vehicle_name = vehicle_name
        self.sensor_name = sensor_name
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._distance = None
        self._last_good_distance = None
        self._last_good_time = 0.0
        self._last_time = 0.0
        self._jump_candidate = None
        self._jump_count = 0

    def get_ground_distance(self):
        with self._lock:
            return self._distance

    def get_last_good_ground_distance(self, max_age=None):
        with self._lock:
            if self._last_good_distance is None:
                return None
            if max_age is not None and self._last_good_time > 0.0:
                if (time.time() - self._last_good_time) > max_age:
                    return None
            return self._last_good_distance

    def has_fresh_ground_distance(self, max_age=1.0):
        with self._lock:
            if self._distance is None:
                return False
            return (time.time() - self._last_time) <= max_age

    def reset_state(self):
        with self._lock:
            self._distance = None
            self._last_good_distance = None
            self._last_good_time = 0.0
            self._last_time = 0.0
            self._jump_candidate = None
            self._jump_count = 0

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                data = self.client.getDistanceSensorData(
                    distance_sensor_name=self.sensor_name,
                    vehicle_name=self.vehicle_name
                )

                dist = float(data.distance)

                if math.isfinite(dist) and 0.2 <= dist <= 500.0:
                    now = time.time()
                    with self._lock:
                        self._distance = dist
                        self._last_time = now

                        if 0.2 <= dist <= DISTANCE_SENSOR_MAX_TRUST:
                            last_good = self._last_good_distance
                            if last_good is None or abs(dist - last_good) <= 80.0:
                                self._last_good_distance = dist
                                self._last_good_time = now
                                self._jump_candidate = None
                                self._jump_count = 0
                            else:
                                if self._jump_candidate is None or abs(dist - self._jump_candidate) > GROUND_TRANSITION_STABLE_BAND:
                                    self._jump_candidate = dist
                                    self._jump_count = 1
                                else:
                                    self._jump_count += 1

                                if self._jump_count >= GROUND_TRANSITION_CONFIRM_FRAMES:
                                    self._last_good_distance = self._jump_candidate
                                    self._last_good_time = now
                                    self._jump_candidate = None
                                    self._jump_count = 0

            except Exception as e:
                print(f"\n[AIRSIM DISTANCE ERROR] {e}")

            time.sleep(0.05)


# ---------------------------------------------------------------------------
# Setpoint streamer with smooth AGL correction
# ---------------------------------------------------------------------------
class SetpointStreamer(threading.Thread):
    def __init__(self, mav, pos_reader, ground_reader, home_lat, home_lon):
        super().__init__(daemon=True)
        self.mav = mav
        self.pos_reader = pos_reader
        self.ground_reader = ground_reader
        self.home_lat = home_lat
        self.home_lon = home_lon

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._target_lat = None
        self._target_lon = None
        self._target_rel_alt = FALLBACK_REL_ALT
        self._target_agl = DESIRED_AGL
        self._tick = 0

        self._last_agl = None
        self._last_agl_time = None

        self._cruise_rel_baseline = None
        self._baseline_confirm_count = 0
        self._low_agl_count = 0
        self._last_target_signature = None
        # Separate fallback floor used only when AGL is blind.
        # Do NOT overwrite this with short AGL-control correction steps.
        self._agl_blind_rel_fallback = None
        # Last yaw commanded by hover-control. Movement uses this so that
        # "turn 90" then "move forward/right" follows the new facing direction.
        self.hover_yaw_deg = None

    def reset_altitude_state(self):
        with self._lock:
            self._last_agl = None
            self._last_agl_time = None
            self._cruise_rel_baseline = None
            self._baseline_confirm_count = 0
            self._low_agl_count = 0
            self._last_target_signature = None
            self._agl_blind_rel_fallback = None

    def set_target(self, lat, lon, rel_alt=None, target_agl=None):
        with self._lock:
            new_signature = (round(lat, 7) if lat is not None else None, round(lon, 7) if lon is not None else None, None if target_agl is None else round(float(target_agl), 1))
            old_signature = self._last_target_signature
            self._target_lat = lat
            self._target_lon = lon
            if rel_alt is not None:
                self._target_rel_alt = rel_alt
            # target_agl=None means: hold requested relative altitude without AGL correction.
            self._target_agl = target_agl

            # CRITICAL TAKEOFF FALLBACK FIX:
            # During takeoff, the AirSim downward distance sensor may briefly work,
            # command only a small +10m correction, then go N/A. The old streamer
            # overwrote _target_rel_alt with that small correction, so REL_GAIN got
            # stuck around 12m/22m. Keep a separate full relative-altitude fallback
            # floor for AGL-blind moments.
            if target_agl is not None and float(target_agl) >= DESIRED_AGL - 1.0:
                if rel_alt is not None:
                    self._agl_blind_rel_fallback = max(float(rel_alt), float(self._target_rel_alt))
                elif self._agl_blind_rel_fallback is None:
                    self._agl_blind_rel_fallback = float(self._target_rel_alt)
            else:
                self._agl_blind_rel_fallback = None
            if old_signature is None:
                self._last_target_signature = new_signature
            else:
                moved = False
                if old_signature[0] is not None and new_signature[0] is not None and old_signature[1] is not None and new_signature[1] is not None:
                    moved = gps_to_distance(old_signature[0], old_signature[1], new_signature[0], new_signature[1]) >= ANTI_CLIMB_RESET_DIST_M
                agl_changed = old_signature[2] != new_signature[2]
                if moved or agl_changed:
                    self._cruise_rel_baseline = None
                    self._baseline_confirm_count = 0
                    self._low_agl_count = 0
                self._last_target_signature = new_signature

    def get_target(self):
        with self._lock:
            return self._target_lat, self._target_lon, self._target_rel_alt, self._target_agl

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            target_lat, target_lon, rel_alt_cmd, target_agl = self.get_target()

            if target_lat is not None and target_lon is not None:
                try:
                    pos = self.pos_reader.get_position()
                    agl = get_believable_agl(self.pos_reader, self.ground_reader)
                    has_agl = (agl is not None)

                    if pos is not None:
                        current_rel = pos["alt_rel"]
                        now = time.time()
                        agl_control_enabled = (target_agl is not None)

                        if agl_control_enabled and has_agl and agl is not None:
                            error = target_agl - agl

                            agl_rate = 0.0
                            if self._last_agl is not None and self._last_agl_time is not None:
                                dt = now - self._last_agl_time
                                if dt > 0.02:
                                    agl_rate = (agl - self._last_agl) / dt

                            # Dampen the rate-based correction to prevent over-reacting to sensor noise.
                            correction = (ALT_KP * error) - (0.5 * ALT_KD * agl_rate)
                            correction = clamp(correction, -MAX_ALT_STEP, MAX_ALT_STEP)

                            rel_alt_cmd = current_rel + correction

                            # SAFETY: If we are already above the target AGL, do not command a higher relative altitude.
                            # This prevents the "keep ascending" behavior.
                            if error < -1.0: # agl > target_agl + 1.0
                                rel_alt_cmd = min(rel_alt_cmd, current_rel)

                            if target_agl >= DESIRED_AGL - 1.0:
                                if abs(error) <= ANTI_CLIMB_BAND and abs(agl_rate) <= 1.0:
                                    self._baseline_confirm_count += 1
                                else:
                                    self._baseline_confirm_count = 0

                                if self._baseline_confirm_count >= ANTI_CLIMB_CONFIRM_FRAMES:
                                    if self._cruise_rel_baseline is None:
                                        self._cruise_rel_baseline = current_rel
                                    else:
                                        # refresh very slowly so the guard can still follow genuine terrain changes
                                        self._cruise_rel_baseline = (0.85 * self._cruise_rel_baseline) + (0.15 * current_rel)

                                if agl < LOW_AGL_BIAS_THRESHOLD:
                                    self._low_agl_count += 1
                                else:
                                    self._low_agl_count = max(0, self._low_agl_count - 1)

                                allow_extra_climb = self._low_agl_count >= ANTI_CLIMB_LOW_AGL_FRAMES

                                if ANTI_CLIMB_ENABLE and self._cruise_rel_baseline is not None and not allow_extra_climb:
                                    cap_rel = self._cruise_rel_baseline + ANTI_CLIMB_MAX_EXTRA_REL
                                    if rel_alt_cmd > cap_rel and agl >= (target_agl - ANTI_CLIMB_BAND):
                                        rel_alt_cmd = min(current_rel, cap_rel)

                                if allow_extra_climb:
                                    if agl < LOW_AGL_BIAS_THRESHOLD:
                                        rel_alt_cmd += EXTRA_CLIMB_BIAS
                                    if agl < CRITICAL_AGL_THRESHOLD:
                                        rel_alt_cmd += CRITICAL_CLIMB_BIAS

                            rel_alt_cmd = clamp(rel_alt_cmd, MIN_REL_ALT, MAX_REL_ALT)

                            # Update the fallback relative altitude so that if AGL is lost,
                            # we hold the current relative altitude instead of jumping to 50m.
                            self._target_rel_alt = rel_alt_cmd

                            self._last_agl = agl
                            self._last_agl_time = now
                        else:
                            # If AGL is lost or disabled, use the currently stored relative-altitude
                            # fallback target. This is important during takeoff: if the downward
                            # sensor goes N/A at 2-5m, the vehicle must keep climbing by REL_GAIN
                            # instead of holding the small kickstart target.
                            if agl_control_enabled and target_agl is not None:
                                blind_floor = self._agl_blind_rel_fallback
                                if target_agl >= DESIRED_AGL - 1.0 and blind_floor is not None and current_rel < blind_floor - 0.5:
                                    rel_alt_cmd = max(rel_alt_cmd, blind_floor)
                                elif target_agl >= DESIRED_AGL - 1.0 and current_rel < self._target_rel_alt - 0.5:
                                    rel_alt_cmd = self._target_rel_alt

                                # However, if the user requested a lower AGL (like 30m hover) and we are stuck high,
                                # allow a slow descent based on the relative altitude difference.
                                descent_needed = max(0.0, DESIRED_AGL - target_agl)
                                if descent_needed > 5.0:
                                    # Slowly nudge the relative altitude down if we are supposed to be lower.
                                    # We limit the step to 0.5m per iteration (5m/s at 10Hz) for safety.
                                    target_rel = max(MIN_REL_ALT, self._target_rel_alt - descent_needed)
                                    if rel_alt_cmd > target_rel:
                                        rel_alt_cmd = max(target_rel, rel_alt_cmd - 0.5)

                            rel_alt_cmd = clamp(rel_alt_cmd, MIN_REL_ALT, MAX_REL_ALT)
                            self._last_agl = None
                            self._last_agl_time = None
                            self._baseline_confirm_count = 0
                            agl_rate = 0.0

                        target_north, target_east = gps_to_local_xy(self.home_lat, self.home_lon, target_lat, target_lon)
                        # Important: do NOT keep sending yaw during flight.
                        # Yaw is aligned once when the user selects a destination.
                        yaw_deg = None

                        send_heartbeat(self.mav)
                        send_position_target_local_ned(self.mav, target_north, target_east, rel_alt_cmd, yaw_deg=yaw_deg)

                        if self._tick % 20 == 0:
                            set_offboard_mode(self.mav)

                        if self._tick % 10 == 0:
                            if target_agl is None:
                                print(
                                    f"\n[STREAM] lat={target_lat:.10f} lon={target_lon:.10f}"
                                    f" | rel_cmd={rel_alt_cmd:.1f}"
                                    f" | rel_now={current_rel:.1f}"
                                    f" | agl_hold=OFF"
                                )
                            elif has_agl and agl is not None:
                                print(
                                    f"\n[STREAM] lat={target_lat:.10f} lon={target_lon:.10f}"
                                    f" | rel_cmd={rel_alt_cmd:.1f}"
                                    f" | rel_now={current_rel:.1f}"
                                    f" | agl={agl:.1f}"
                                    f" | agl_rate={agl_rate:+.2f}m/s"
                                    f" | agl_tgt={target_agl:.1f}"
                                    f" | err={target_agl - agl:+.1f}"
                                )
                            else:
                                blind_floor_str = ""
                                if agl_control_enabled and self._agl_blind_rel_fallback is not None:
                                    blind_floor_str = f" | blind_rel_floor={self._agl_blind_rel_fallback:.1f}"
                                print(
                                    f"\n[STREAM] lat={target_lat:.10f} lon={target_lon:.10f}"
                                    f" | rel_cmd={rel_alt_cmd:.1f}"
                                    f" | rel_now={current_rel:.1f}"
                                    f" | agl_tgt={target_agl:.1f}"
                                    f" | agl=N/A"
                                    f"{blind_floor_str}"
                                )

                    self._tick += 1

                except Exception as e:
                    print(f"\n[STREAMER ERROR] {e}")

            time.sleep(0.1)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def print_menu(locations, current_pos=None, ground_reader=None, forward_reader=None, forward_active=False):
    print("\n" + "=" * 72)
    print("                    DRONE NAVIGATION MENU")
    print("=" * 72)

    if current_pos:
        print(f"  Current   lat={current_pos['lat']:.10f}")
        print(f"            lon={current_pos['lon']:.10f}")
        print(f"            alt_rel={current_pos['alt_rel']:.1f}m")

        if ground_reader:
            agl = get_believable_agl(type("P", (), {"get_position": lambda _self: current_pos})(), ground_reader)
            if agl is not None:
                print(f"            agl={agl:.1f}m (AirSim distance sensor)")
            else:
                print(f"            agl=N/A")

        if forward_reader:
            status = forward_reader.get_status()
            if status["fresh"]:
                dist = status["distance"]
                dist_str = f"{dist:.1f}m" if dist is not None else "N/A"
                print(
                    f"            forward_dist={dist_str}"
                    f" | obstacle_ahead={status['obstacle_detected']}"
                    f" | forward_active={forward_active}"
                )
            else:
                print(f"            forward=N/A | forward_active={forward_active}")

    print("-" * 72)
    print("  Destinations:\n")

    for i, loc in enumerate(locations, 1):
        g = loc["gps"]
        dist_str = ""
        if current_pos:
            d = gps_to_distance(current_pos["lat"], current_pos["lon"], g["lat"], g["lon"])
            b = gps_to_bearing(current_pos["lat"], current_pos["lon"], g["lat"], g["lon"])
            dist_str = f" ({d:.0f}m away at {b:.0f}°)"

        print(f"  [{i}] {loc['name']}{dist_str}")
        print(f"      lat={g['lat']:.10f}  lon={g['lon']:.10f}")

    print("\n  [0] Land only")
    print("  [-1] Exit program")
    print("=" * 72)


def wait_for_home_position(mav, tries=40):
    for _ in range(tries):
        send_heartbeat(mav)
        msg = mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1.0)
        if msg and msg.lat != 0:
            return {
                "lat": msg.lat / 1e7,
                "lon": msg.lon / 1e7,
                "alt_rel": msg.relative_alt / 1000.0,
                "alt_msl": msg.alt / 1000.0
            }
    return None



def get_believable_agl(pos_reader, ground_reader):
    """
    Prefer a fresh downward distance reading.

    A short-lived fallback to the last good reading helps ride through brief
    sensor hiccups, but stale values must never survive a landing/relaunch cycle
    because that can trick the altitude controller into commanding a continuous
    climb on the next mission.
    """
    pos = pos_reader.get_position()
    if not pos:
        return None

    agl = ground_reader.get_ground_distance()
    if ground_reader.has_fresh_ground_distance() and agl is not None and math.isfinite(agl):
        if 0.2 <= agl <= DISTANCE_SENSOR_MAX_TRUST:
            return agl

    last_good = ground_reader.get_last_good_ground_distance(max_age=LAST_GOOD_AGL_MAX_AGE)
    if last_good is not None and math.isfinite(last_good):
        if 0.2 <= last_good <= DISTANCE_SENSOR_MAX_TRUST:
            return last_good

    return None


def wait_for_fresh_position(pos_reader, timeout=2.0):
    start = time.time()
    last_pos = None

    while time.time() - start < timeout:
        if pos_reader.has_fresh_position():
            pos = pos_reader.get_position()
            if pos:
                return pos
        pos = pos_reader.get_position()
        if pos:
            last_pos = pos
        time.sleep(0.05)

    return last_pos


def wait_until_takeoff_height(pos_reader, ground_reader, timeout=90, start_rel_alt=0.0):
    """
    Startup takeoff must also work after landing somewhere other than home.

    In that case, PX4 relative altitude may already be far from zero even though
    the drone is sitting on the ground again (for example, on a roof or elevated
    terrain). So during takeoff we should prefer true local AGL whenever the
    downward distance sensor is fresh and reasonable, and only fall back to
    rel-alt gain if AGL is unavailable.
    """
    start = time.time()
    while time.time() - start < timeout:
        pos = pos_reader.get_position()
        agl = get_believable_agl(pos_reader, ground_reader)
        has_agl = (agl is not None)

        if pos:
            rel_alt = pos["alt_rel"]

            use_agl = False
            if has_agl and agl is not None:
                # For restart-after-landing, do NOT require agl ~= rel_alt.
                # rel_alt is referenced to the original home altitude, while agl
                # is referenced to the local ground directly below the drone.
                if 0.2 <= agl <= 120.0:
                    use_agl = True

            if use_agl:
                progress_value = agl
                target_value = DESIRED_AGL
                label = "AGL"
            else:
                progress_value = max(0.0, rel_alt - start_rel_alt)
                target_value = FALLBACK_REL_ALT
                label = "REL_GAIN"

            progress = min(max(progress_value, 0.0) / max(target_value, 0.1), 1.0)
            bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
            print(f"\r  [{bar}] {progress_value:.1f} / {target_value:.1f} m {label}   ", end="", flush=True)

            if progress_value >= target_value - 3.0:
                print("\n  Reached takeoff height.")
                return True

        time.sleep(0.2)

    print("\n  Takeoff height wait timeout")
    return False


def kickstart_liftoff(pos_reader, streamer, ground_reader, mav_write, timeout=8.0):
    """
    Startup-only helper for rerun-after-landing.

    Your PX4 debug output showed that on the bad second run the vehicle can stay
    DISARMED + LANDED even though the script proceeds into takeoff. So during
    this short kickstart phase we do three things together:
    1) keep streaming a small upward rel-alt target,
    2) keep re-asserting OFFBOARD,
    3) keep re-sending ARM.

    As soon as the drone actually lifts a little, we stop and hand control back
    to the normal 50m AGL logic.
    """
    pos0 = wait_for_fresh_position(pos_reader, timeout=2.0)
    if not pos0:
        return False

    agl0 = get_believable_agl(pos_reader, ground_reader)
    if agl0 is not None and agl0 > 2.0:
        return True

    liftoff_rel_target = clamp(max(0.0, pos0["alt_rel"]) + RELAUNCH_KICK_ALT, TAKEOFF_MIN_REL_ALT, MAX_REL_ALT)
    streamer.set_target(
        pos0["lat"],
        pos0["lon"],
        liftoff_rel_target,
        target_agl=None
    )

    print("  Kickstarting liftoff...", end="", flush=True)
    start = time.time()
    last_print = 0.0
    last_arm_push = 0.0

    while time.time() - start < timeout:
        now = time.time()

        # Re-assert offboard + arm while trying to break out of landed state.
        if now - last_arm_push >= 0.5:
            try:
                set_offboard_mode(mav_write)
                time.sleep(0.05)
                arm_drone(mav_write)

                pos_now = pos_reader.get_position()
                if pos_now:
                    takeoff_msl = pos_now["alt_msl"] + max(12.0, DESIRED_AGL)
                    send_takeoff(mav_write, pos_now["lat"], pos_now["lon"], takeoff_msl)
            except Exception:
                pass
            last_arm_push = now

        pos = pos_reader.get_position()
        agl = get_believable_agl(pos_reader, ground_reader)

        if pos:
            rel_gain = pos["alt_rel"] - pos0["alt_rel"]
            if rel_gain >= 3.0:
                print(" OK")
                return True

        if agl is not None and agl >= 2.5:
            print(" OK")
            return True

        if now - last_print >= 0.5:
            print(".", end="", flush=True)
            last_print = now

        time.sleep(0.1)

    print(" timeout")
    return False


def relaunch_takeoff_recovery(pos_reader, streamer, ground_reader, mav_write, timeout=RELAUNCH_RECOVERY_SEC):
    """Second-stage recovery for the occasional later relaunch that stays glued to the ground."""
    pos0 = wait_for_fresh_position(pos_reader, timeout=2.0)
    if not pos0:
        return False

    recovery_rel_target = clamp(max(0.0, pos0["alt_rel"]) + RELAUNCH_KICK_ALT + 7.0, TAKEOFF_MIN_REL_ALT, MAX_REL_ALT)
    streamer.set_target(pos0["lat"], pos0["lon"], recovery_rel_target, target_agl=None)

    print("  Relaunch recovery...", end="", flush=True)
    start = time.time()
    last_push = 0.0
    last_dot = 0.0

    while time.time() - start < timeout:
        now = time.time()

        if now - last_push >= 0.35:
            try:
                set_offboard_mode(mav_write)
                arm_drone(mav_write)
                pos_now = pos_reader.get_position()
                if pos_now:
                    streamer.set_target(pos_now["lat"], pos_now["lon"], recovery_rel_target, target_agl=None)
                    send_takeoff(mav_write, pos_now["lat"], pos_now["lon"], pos_now["alt_msl"] + DESIRED_AGL)
            except Exception:
                pass
            last_push = now

        pos = pos_reader.get_position()
        agl = get_believable_agl(pos_reader, ground_reader)
        if pos and (pos["alt_rel"] - pos0["alt_rel"]) >= 2.0:
            print(" OK")
            return True
        if agl is not None and agl >= 2.0:
            print(" OK")
            return True

        if now - last_dot >= 0.5:
            print(".", end="", flush=True)
            last_dot = now

        time.sleep(0.1)

    print(" timeout")
    return False


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
def fly_to_target(name, gps, pos_reader, streamer, ground_reader, mav_write, cruise_rel_alt, forward_reader=None, forward_active=True):
    pos0 = wait_for_fresh_position(pos_reader, timeout=2.0)
    if not pos0:
        print("  ERROR: no current PX4 position.")
        return False

    initial_dist = gps_to_distance(pos0["lat"], pos0["lon"], gps["lat"], gps["lon"])
    initial_bearing = gps_to_bearing(pos0["lat"], pos0["lon"], gps["lat"], gps["lon"])

    print(f"\n  >> Flying to: {name}")
    print(f"     TARGET   lat={gps['lat']:.10f}  lon={gps['lon']:.10f}")
    print(f"     START    lat={pos0['lat']:.10f}  lon={pos0['lon']:.10f}")
    print(f"     Distance: {initial_dist:.1f}m at bearing {initial_bearing:.0f}°")

    if initial_dist <= 2.0:
        print("     Already at target.")
        return True

    align_yaw_once_before_flight(mav_write, pos0, gps, streamer.home_lat, streamer.home_lon)

    # Lock cruise to the rel-alt actually reached at takeoff. This is more stable after landing and re-launching.
    streamer.set_target(gps["lat"], gps["lon"], cruise_rel_alt, target_agl=DESIRED_AGL)
    set_speed(mav_write, CRUISE_SPEED)

    best_dist = initial_dist
    close_count = 0
    start_time = time.time()
    last_print = 0.0

    while time.time() - start_time < MAX_NAV_TIME:
        pos = pos_reader.get_position()
        if pos:
            target_dist = gps_to_distance(pos["lat"], pos["lon"], gps["lat"], gps["lon"])
            bearing = gps_to_bearing(pos["lat"], pos["lon"], gps["lat"], gps["lon"])
            best_dist = min(best_dist, target_dist)

            agl = get_believable_agl(pos_reader, ground_reader)
            has_agl = (agl is not None)

            now = time.time()

            lidar_suffix = ""
            if forward_reader is not None and forward_reader.has_fresh_data():
                fwd_dist = forward_reader.get_distance()
                dist_str = f"{fwd_dist:.1f}m" if fwd_dist is not None else "N/A"
                if forward_reader.obstacle_ahead():
                    lidar_suffix = f" | fwd={dist_str} BLOCKED | active={forward_active}"
                else:
                    lidar_suffix = f" | fwd={dist_str} clear | active={forward_active}"
            elif forward_reader is not None:
                lidar_suffix = f" | fwd=N/A | active={forward_active}"

            if now - last_print >= 0.5:
                if has_agl and agl is not None:
                    print(
                        f"\r     Dist: {target_dist:6.1f}m @ {bearing:3.0f}°"
                        f" | rel_alt: {pos['alt_rel']:5.1f}m"
                        f" | agl: {agl:5.1f}m"
                        f" | Best: {best_dist:6.1f}m"
                        f"{lidar_suffix}   ",
                        end="",
                        flush=True
                    )
                else:
                    print(
                        f"\r     Dist: {target_dist:6.1f}m @ {bearing:3.0f}°"
                        f" | rel_alt: {pos['alt_rel']:5.1f}m"
                        f" | agl:  N/A "
                        f" | Best: {best_dist:6.1f}m"
                        f"{lidar_suffix}   ",
                        end="",
                        flush=True
                    )
                last_print = now

            # forward obstacle avoidance hook
            if forward_reader is not None and forward_reader.has_fresh_data():
                if forward_reader.obstacle_ahead():
                    handled = handle_forward_avoidance(
                        name,
                        gps,
                        pos_reader,
                        streamer,
                        ground_reader,
                        mav_write,
                        forward_reader,
                        get_believable_agl,
                        DESIRED_AGL,
                        CRUISE_SPEED,
                        AVOIDANCE_BRAKE_SPEED,
                        AVOIDANCE_HOLD_SPEED,
                        AVOIDANCE_CLIMB_STEP_AGL,
                        AVOIDANCE_MAX_EXTRA_AGL,
                        AVOIDANCE_RECHECK_SEC,
                        AVOIDANCE_SETTLE_SEC,
                        active=forward_active,
                        clear_frames_required=6,
                        clear_distance_margin=15.0,
                        extra_clearance_agl=15.0,
                        roof_cross_time_sec=10.0,
                        post_cross_suppress_sec=8.0,
                    )
                    if handled:
                        close_count = 0
                        last_print = 0.0
                        continue

            if target_dist <= ARRIVAL_RADIUS:
                close_count += 1
            else:
                close_count = 0

            if close_count >= ARRIVAL_CONFIRM_COUNT:
                print(f"\n  >> Arrived at '{name}'!")
                return True

        time.sleep(0.1)

    print(f"\n  >> Timeout before reaching '{name}'")
    return False


# ---------------------------------------------------------------------------
# Target hold at lower hover height
# ---------------------------------------------------------------------------
def hold_at_target(name, gps, pos_reader, streamer, ground_reader, cruise_rel_alt, hold_agl=HOVER_AGL, hold_seconds=HOVER_DURATION):
    pos = pos_reader.get_position()
    hover_drop = max(0.0, DESIRED_AGL - hold_agl)
    hover_rel = max(MIN_REL_ALT, cruise_rel_alt - hover_drop)

    streamer.set_target(gps["lat"], gps["lon"], hover_rel, target_agl=hold_agl)

    print(f"     Descending and holding at target: {hold_agl:.1f}m hover target for {hold_seconds}s...")
    start = time.time()
    last_print = 0.0

    while time.time() - start < hold_seconds:
        pos = pos_reader.get_position()
        agl = get_believable_agl(pos_reader, ground_reader)
        has_agl = (agl is not None)

        now = time.time()
        if now - last_print >= 0.5:
            if pos and has_agl and agl is not None:
                dist = gps_to_distance(pos["lat"], pos["lon"], gps["lat"], gps["lon"])
                print(
                    f"\r     Hold dist: {dist:5.2f}m"
                    f" | rel_alt: {pos['alt_rel']:5.1f}m"
                    f" | agl: {agl:5.1f}m"
                    f" | rel_tgt: {hover_rel:5.1f}m   ",
                    end="",
                    flush=True
                )
            elif pos:
                dist = gps_to_distance(pos["lat"], pos["lon"], gps["lat"], gps["lon"])
                print(
                    f"\r     Hold dist: {dist:5.2f}m"
                    f" | rel_alt: {pos['alt_rel']:5.1f}m"
                    f" | rel_tgt: {hover_rel:5.1f}m"
                    f" | agl: N/A   ",
                    end="",
                    flush=True
                )
            last_print = now

        time.sleep(0.1)

    print()
    # Ensure we keep the last commanded relative altitude as the fallback.
    streamer.set_target(gps["lat"], gps["lon"], hover_rel, target_agl=hold_agl)


def is_believable_agl(agl, rel_alt):
    if agl is None:
        return False
    if not math.isfinite(agl):
        return False
    if agl < 0.0 or agl > 120.0:
        return False
    if rel_alt is not None and abs(agl - rel_alt) > 20.0:
        return False
    return True


def wait_for_touchdown(pos_reader, ground_reader, timeout=45):
    start = time.time()
    stable_count = 0
    last_print = 0.0

    while time.time() - start < timeout:
        pos = pos_reader.get_position()
        rel_alt = pos["alt_rel"] if pos else None

        agl = ground_reader.get_ground_distance()
        believable_agl = is_believable_agl(agl, rel_alt)

        now = time.time()
        if now - last_print >= 0.5:
            agl_str = f"{agl:5.1f}m" if believable_agl else "  N/A "
            rel_str = f"{rel_alt:5.1f}m" if rel_alt is not None else "  N/A "
            print(f"\r     Landing... rel_alt: {rel_str} | agl: {agl_str}   ", end="", flush=True)
            last_print = now

        touched = False
        if rel_alt is not None and rel_alt <= LAND_DONE_REL_ALT:
            touched = True
        elif believable_agl and agl <= LAND_DONE_AGL:
            touched = True

        if touched:
            stable_count += 1
            if stable_count >= 4:
                print()
                return True
        else:
            stable_count = 0

        time.sleep(0.2)

    print()
    return False




def ensure_forward_reader_started(forward_reader, vehicle_name=VEHICLE_NAME):
    if forward_reader is not None:
        return forward_reader, True
    print("Connecting to forward distance sensor...", end="", flush=True)
    try:
        forward_reader = ForwardObstacleReader(
            vehicle_name=vehicle_name,
            sensor_name=FORWARD_SENSOR_NAME,
            obstacle_distance=FORWARD_OBSTACLE_DISTANCE,
            min_valid_distance=FORWARD_MIN_VALID_DISTANCE,
            max_valid_distance=FORWARD_MAX_VALID_DISTANCE,
            persist_frames=FORWARD_PERSIST_FRAMES,
            poll_period=0.05
        )
        forward_reader.start()
        time.sleep(1.0)
        print(" OK")
        print("Forward obstacle avoidance is now ACTIVE.")
        return forward_reader, True
    except Exception as e:
        print(f" FAILED ({e})")
        return None, False


def perform_takeoff_sequence(mav, pos_reader, streamer, ground_reader):
    # Clear any latched ground-distance value from a previous flight or landing
    # before the next altitude-control cycle starts.
    ground_reader.reset_state()
    streamer.reset_altitude_state()
    time.sleep(0.2)

    current_pos = wait_for_fresh_position(pos_reader, timeout=2.0)
    if not current_pos:
        print("  ERROR: no current PX4 position for takeoff.")
        return False, None

    # Important fallback target for takeoff:
    # If the AirSim downward AGL sensor goes blind during climb, the streamer must
    # still have a full 50m REL_GAIN target. The old version used current_rel here,
    # so after the kickstart it could get stuck around ~20-22m when agl=N/A.
    takeoff_start_rel_alt = current_pos["alt_rel"]
    fallback_takeoff_rel_alt = clamp(takeoff_start_rel_alt + FALLBACK_REL_ALT, TAKEOFF_MIN_REL_ALT, MAX_REL_ALT)
    streamer.set_target(current_pos["lat"], current_pos["lon"], fallback_takeoff_rel_alt, target_agl=DESIRED_AGL)

    print(f"\nPriming offboard mode ({OFFBOARD_PRIME_SEC:.0f}s)...", end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < OFFBOARD_PRIME_SEC:
        print(".", end="", flush=True)
        time.sleep(0.2)
    print(" OK")

    print("Arming...")
    for attempt in range(2):
        set_offboard_mode(mav)
        time.sleep(0.1)
        arm_drone(mav)
        print(f"  Arm command sent (attempt {attempt + 1})")
        time.sleep(0.25)

    print("\nTaking off...")
    set_speed(mav, CRUISE_SPEED)

    lifted = kickstart_liftoff(pos_reader, streamer, ground_reader, mav, timeout=8.0)
    if not lifted:
        lifted = relaunch_takeoff_recovery(pos_reader, streamer, ground_reader, mav, timeout=RELAUNCH_RECOVERY_SEC)

    pos_before_agl_takeoff = wait_for_fresh_position(pos_reader, timeout=2.0)
    if pos_before_agl_takeoff:
        # After kickstart_liftoff(), do NOT keep the small kickstart relative target.
        # If AGL disappears, continue climbing until REL_GAIN reaches 50m from the
        # takeoff starting relative altitude. If AGL comes back, normal AGL control
        # will override this fallback automatically.
        fallback_takeoff_rel_alt = clamp(takeoff_start_rel_alt + FALLBACK_REL_ALT, TAKEOFF_MIN_REL_ALT, MAX_REL_ALT)
        streamer.set_target(
            pos_before_agl_takeoff["lat"],
            pos_before_agl_takeoff["lon"],
            fallback_takeoff_rel_alt,
            target_agl=DESIRED_AGL
        )

    takeoff_ok = wait_until_takeoff_height(
        pos_reader,
        ground_reader,
        timeout=90,
        start_rel_alt=(current_pos["alt_rel"] if current_pos else 0.0)
    )

    cruise_rel_alt = None
    if takeoff_ok:
        pos_after_takeoff = wait_for_fresh_position(pos_reader, timeout=2.0)
        if pos_after_takeoff:
            cruise_rel_alt = pos_after_takeoff["alt_rel"]
            streamer.set_target(
                pos_after_takeoff["lat"],
                pos_after_takeoff["lon"],
                cruise_rel_alt,
                target_agl=DESIRED_AGL
            )
    return takeoff_ok, cruise_rel_alt


def perform_landing_sequence(mav, pos_reader, streamer, ground_reader, forward_reader=None):
    print("\n  >> Landing...")
    if streamer is not None:
        streamer.stop()
        time.sleep(0.5)

    set_speed(mav, LANDING_SPEED)
    land(mav)
    wait_for_touchdown(pos_reader, ground_reader, timeout=45)
    time.sleep(1.0)
    disarm(mav)
    time.sleep(LAND_SETTLE_SEC)
    ground_reader.reset_state()
    if streamer is not None:
        streamer.reset_altitude_state()
    if forward_reader is not None:
        try:
            forward_reader.stop()
            time.sleep(0.5)
        except Exception:
            pass
    print("  >> Landed. Returning to menu.\n")
    return None, None, False


# ---------------------------------------------------------------------------
# Safe command parsing / resolving
# ---------------------------------------------------------------------------
def _fallback_navigate_command(raw):
    """Emergency fallback when the LLM/parser is slow on the first command.

    This keeps the terminal responsive. It does not replace the resolver; it only
    gives the nav loop a simple destination command when the normal parser times
    out before returning anything.
    """
    text = (raw or "").strip()
    lowered = text.lower()

    if not text:
        return {"type": "empty", "parser": "timeout_fallback"}
    if lowered in {"land", "landing", "هبوط"}:
        return {"type": "land", "parser": "timeout_fallback"}
    if lowered in {"exit", "quit", "close", "stop program", "خروج", "سكر"}:
        return {"type": "exit", "parser": "timeout_fallback"}

    place = text
    for prefix in [
        "go to ", "fly to ", "navigate to ", "take me to ",
        "روح على ", "اذهب الى ", "اذهب إلى ", "وديني على ", "روح ل"
    ]:
        if place.lower().startswith(prefix.lower()):
            place = place[len(prefix):].strip()
            break

    candidates = [place]
    if "عمان" not in place and "jordan" not in place.lower() and "amman" not in place.lower():
        candidates.append(f"{place}, عمان, الأردن")
        candidates.append(f"{place}, Amman, Jordan")

    deduped = []
    seen = set()
    for item in candidates:
        key = " ".join(str(item).lower().split())
        if item and key not in seen:
            seen.add(key)
            deduped.append(item)

    return {
        "type": "navigate",
        "place_text": deduped[0],
        "extracted_location": place,
        "original_text": text,
        "osm_candidates": deduped,
        "parser": "timeout_fallback",
    }


def parse_terminal_command_safe(raw, timeout=PARSER_TIMEOUT_SEC):
    """Run parse_terminal_command without letting a slow first LLM call freeze input."""
    result = {}
    done = threading.Event()

    def worker():
        try:
            result["command"] = parse_terminal_command(raw)
        except Exception as e:
            result["command"] = {"type": "unknown", "error": str(e), "parser": "safe_wrapper"}
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    if done.wait(timeout):
        return result.get("command") or {"type": "unknown", "error": "Parser returned no command.", "parser": "safe_wrapper"}

    print(f"  [PARSER WARNING] Parser took more than {timeout:.0f}s. Using direct destination fallback.")
    return _fallback_navigate_command(raw)


def resolve_place_with_osm_safe(place_text, command=None, timeout=RESOLVER_TIMEOUT_SEC):
    """Run OSM resolving with a timeout so one slow network call cannot freeze the terminal."""
    result = {}
    done = threading.Event()

    def worker():
        try:
            result["resolved"] = resolve_place_with_osm(place_text, command)
        except Exception as e:
            result["resolved"] = {"ok": False, "error": str(e)}
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    if done.wait(timeout):
        return result.get("resolved") or {"ok": False, "error": "Resolver returned no result."}

    return {
        "ok": False,
        "error": f"Resolver timed out after {timeout:.0f}s. Try a simpler place name or a saved/local coordinate.",
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"\nConnecting to PX4 on {MAVLINK_CONNECTION} ...")
    mav = mavutil.mavlink_connection(MAVLINK_CONNECTION)

    print("Waiting for heartbeat...", end="", flush=True)
    mav.wait_heartbeat()
    mav.target_system = 1
    mav.target_component = 1
    print(" OK")

    print("Reading home position...", end="", flush=True)
    home_pos = wait_for_home_position(mav)
    if not home_pos:
        print(" FAILED")
        return
    print(" OK")
    print(f"  lat={home_pos['lat']:.10f}  lon={home_pos['lon']:.10f}")
    print(f"  rel_alt={home_pos['alt_rel']:.1f}m  msl={home_pos['alt_msl']:.1f}m")

    print("Connecting to AirSim distance sensor...", end="", flush=True)
    ground_reader = AirSimGroundReader(vehicle_name=VEHICLE_NAME, sensor_name=DISTANCE_SENSOR_NAME)
    ground_reader.start()
    time.sleep(1.0)
    if ground_reader.has_fresh_ground_distance():
        print(" OK")
    else:
        print(" WARNING: no fresh AirSim ground distance yet")

    # forward obstacle reader is intentionally NOT started before takeoff.
    # This prevents near-ground / startup false obstacle triggers from blocking climb.
    forward_reader = None
    forward_active = False

    # Hover-control commands are only accepted after a successful arrival/hover.
    hover_mode = False
    hover_agl_current = HOVER_AGL
    last_hover_control_command = None

    pos_reader = PositionReader(mav)
    streamer = SetpointStreamer(mav, pos_reader, ground_reader, home_pos["lat"], home_pos["lon"])

    pos_reader.start()
    streamer.start()

    takeoff_ok, cruise_rel_alt = perform_takeoff_sequence(mav, pos_reader, streamer, ground_reader)
    if cruise_rel_alt is None and home_pos is not None:
        cruise_rel_alt = DESIRED_AGL

    if FORWARD_ENABLE_AFTER_TAKEOFF and FORWARD_OBSTACLE_ENABLE and takeoff_ok:
        forward_reader, forward_active = ensure_forward_reader_started(forward_reader)

    print("\nReady for navigation.")
    print("GPS controls direction. AirSim distance sensor controls real height above ground.")
    print("Type a destination after takeoff, hover-control after arrival, save current location, or type 'land' / 'exit'.")
    if forward_reader is not None:
        print(f"Forward obstacle sensor is ENABLED. Forward avoidance active={forward_active}\n")
    else:
        print("Forward obstacle avoidance is DISABLED.\n")

    try:
        saved_data = load_saved_locations(SAVED_LOCATIONS_FILE)
        saved_count = len(saved_data.get("locations") or [])
        print(f"Saved local locations file: {SAVED_LOCATIONS_FILE} ({saved_count} saved)\n")
    except Exception as e:
        print(f"Saved local locations file: {SAVED_LOCATIONS_FILE} (could not read: {e})\n")

    while True:
        current_pos = pos_reader.get_position()
        if current_pos:
            print("\n" + "=" * 72)
            print("                    DRONE NAVIGATION TERMINAL")
            print("=" * 72)
            print(f"  Current   lat={current_pos['lat']:.10f}")
            print(f"            lon={current_pos['lon']:.10f}")
            print(f"            alt_rel={current_pos['alt_rel']:.1f}m")
            agl = get_believable_agl(type("P", (), {"get_position": lambda _self: current_pos})(), ground_reader)
            if agl is not None:
                print(f"            agl={agl:.1f}m (AirSim distance sensor)")
            else:
                print("            agl=N/A")
            if hover_mode:
                print(f"            hover_control=ACTIVE | hover_agl_target={hover_agl_current:.1f}m")
                if last_hover_control_command is not None:
                    print(f"            last_control={describe_hover_command(last_hover_control_command)}")
            else:
                print("            hover_control=OFF until arrival")
            try:
                saved_locations = load_saved_locations(SAVED_LOCATIONS_FILE).get("locations") or []
            except Exception:
                saved_locations = []
            if saved_locations:
                preview = ", ".join(str(loc.get("name", "unnamed")) for loc in saved_locations[:6])
                more = "..." if len(saved_locations) > 6 else ""
                print(f"            saved_locations={len(saved_locations)} | {preview}{more}")
            else:
                print("            saved_locations=0")
            print("-" * 72)
            print("  Enter destination text, hover-control command, save command, or type 'land' / 'exit'.")
            print("  Save example: save this location as checkpoint one")
            print("=" * 72)

        try:
            raw = input("\n  Enter command: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting...")
            break

        print("  Parsing command...", flush=True)
        command = parse_terminal_command_safe(raw)

        if command["type"] == "empty":
            print("  Please type a destination, 'land', or 'exit'.")
            continue

        if command["type"] == "exit":
            break

        if command["type"] == "land":
            hover_mode = False
            last_hover_control_command = None
            streamer, forward_reader, forward_active = perform_landing_sequence(
                mav, pos_reader, streamer, ground_reader, forward_reader
            )
            print("  Type a destination to relaunch automatically, or 'exit' to quit.")
            continue

        if command["type"] == "save_location":
            pos_to_save = wait_for_fresh_position(pos_reader, timeout=2.0)
            if not pos_to_save:
                print("  [SAVE] No fresh drone GPS position. Location was not saved.")
                continue

            save_result = save_current_location(
                command.get("name", ""),
                pos_to_save,
                aliases=command.get("aliases") or [],
                filepath=SAVED_LOCATIONS_FILE,
            )
            if not save_result.get("ok"):
                print(f"  [SAVE] Failed: {save_result.get('error', 'unknown error')}")
                continue

            saved_loc = save_result["location"]
            gps_saved = saved_loc.get("gps", {})
            status_word = "Updated" if save_result.get("updated") else "Saved"
            print(f"  [SAVE] {status_word} local location: {saved_loc.get('name')}")
            print(f"  [SAVE] lat={float(gps_saved['lat']):.10f}  lon={float(gps_saved['lon']):.10f}")
            print(f"  [SAVE] File: {save_result.get('filepath', SAVED_LOCATIONS_FILE)}")
            print("  [SAVE] Next time you can say: go to " + str(saved_loc.get("name")))
            continue

        if command["type"] == "repeat_control":
            if not hover_mode:
                print("  Repeat only works after the drone reaches a destination and hover-control is active.")
                continue
            if last_hover_control_command is None:
                print("  No previous hover-control command to repeat yet.")
                continue
            if streamer is None or not streamer.is_alive():
                print("  Streamer is not active. Repeat command ignored.")
                continue
            print(f"  [CONTROL] Repeating last command: {describe_hover_command(last_hover_control_command)}")
            hover_agl_current = execute_hover_control_command(
                last_hover_control_command, pos_reader, streamer, ground_reader, mav, hover_agl_current
            )
            continue

        if command["type"] == "unknown":
            print(f"  Unsupported command: {command.get('error', 'unknown command')}")
            continue

        if command["type"] == "control":
            if not hover_mode:
                print("  Hover-control commands are only accepted after the drone reaches a destination and is hovering.")
                print("  For now, type a destination first.")
                continue
            if streamer is None or not streamer.is_alive():
                print("  Streamer is not active. Control command ignored.")
                continue
            hover_agl_current = execute_hover_control_command(
                command, pos_reader, streamer, ground_reader, mav, hover_agl_current
            )
            repeatable = make_repeatable_hover_command(command)
            if repeatable is not None:
                last_hover_control_command = repeatable
                print(f"  [CONTROL] Saved for repeat: {describe_hover_command(last_hover_control_command)}")
            continue

        if command["type"] != "navigate":
            print(f"  Unsupported command type: {command['type']}")
            continue

        hover_mode = False
        last_hover_control_command = None
        if streamer is not None:
            streamer.hover_yaw_deg = None
        place_text = command["place_text"]
        print(f"  Checking saved locations first: {place_text}", flush=True)
        try:
            resolved = resolve_saved_location(place_text, command, filepath=SAVED_LOCATIONS_FILE)
        except Exception as e:
            resolved = {"ok": False, "error": str(e)}

        if resolved.get("ok"):
            print(
                f"  Saved location match: {resolved['display_name']} "
                f"(score={float(resolved.get('match_score', 0.0)):.2f})"
            )
        else:
            print("  No saved location match. Falling back to OpenStreetMap...")
            print(f"  Resolving destination: {place_text}", flush=True)
            resolved = resolve_place_with_osm_safe(place_text, command)
            if not resolved["ok"]:
                print(f"  Resolver error: {resolved['error']}")
                continue

        gps = {
            "lat": resolved["lat"],
            "lon": resolved["lon"],
            "alt": 0.0
        }
        name = resolved["display_name"]

        print(f"  Resolved to: {name}")
        if resolved.get("source") == "saved_location":
            print("  Source: saved_locations.json")
        if resolved.get("osm_query"):
            print(f"  OSM query used: {resolved['osm_query']}")
        print(f"  lat={gps['lat']:.10f}  lon={gps['lon']:.10f}")

        if streamer is None or not streamer.is_alive():
            streamer = SetpointStreamer(mav, pos_reader, ground_reader, home_pos["lat"], home_pos["lon"])
            streamer.start()
            takeoff_ok, cruise_rel_alt = perform_takeoff_sequence(mav, pos_reader, streamer, ground_reader)
            if cruise_rel_alt is None:
                pos_now = pos_reader.get_position()
                cruise_rel_alt = pos_now["alt_rel"] if pos_now else DESIRED_AGL
            if FORWARD_ENABLE_AFTER_TAKEOFF and FORWARD_OBSTACLE_ENABLE and takeoff_ok:
                forward_reader, forward_active = ensure_forward_reader_started(forward_reader)
            else:
                forward_active = False

        success = fly_to_target(
            name,
            gps,
            pos_reader,
            streamer,
            ground_reader,
            mav,
            cruise_rel_alt,
            forward_reader,
            forward_active=forward_active
        )

        if success:
            hold_at_target(
                name, gps, pos_reader, streamer, ground_reader, cruise_rel_alt,
                hold_agl=HOVER_AGL, hold_seconds=HOVER_DURATION
            )

            pos_after_hold = wait_for_fresh_position(pos_reader, timeout=2.0)
            if pos_after_hold:
                # For the cleanest demo hover, freeze the actual current relative altitude.
                # This avoids tiny AGL sensor corrections while waiting for hover-control commands.
                streamer.set_target(
                    pos_after_hold["lat"],
                    pos_after_hold["lon"],
                    pos_after_hold["alt_rel"],
                    target_agl=None if HOVER_FREEZE_REL_ALT_DURING_CONTROL else HOVER_AGL
                )
                # Initialize hover yaw memory from the current vehicle yaw.
                # After this, turn commands update the memory exactly.
                streamer.hover_yaw_deg = get_heading_from_position(pos_after_hold)
            hover_mode = True
            hover_agl_current = HOVER_AGL
            print("     Hover-control is now ACTIVE. Examples: 'لف يمين 15 درجة', 'turn right 15 degrees', 'move forward 5 meters', 'do it again'.")
        else:
            hover_mode = False
            pos = pos_reader.get_position()
            if pos:
                streamer.set_target(pos["lat"], pos["lon"], max(MIN_REL_ALT, cruise_rel_alt - (DESIRED_AGL - HOVER_AGL)), target_agl=HOVER_AGL)
            print("     Target not reached. Hover-control remains OFF.\n")

    if streamer is not None:
        print("\n  >> Landing before exit...")
        streamer.stop()
        time.sleep(0.5)
        set_speed(mav, LANDING_SPEED)
        land(mav)
        wait_for_touchdown(pos_reader, ground_reader, timeout=45)

    pos_reader.stop()
    ground_reader.stop()
    if forward_reader is not None:
        forward_reader.stop()

    time.sleep(1.0)
    disarm(mav)
    print("  >> Done.\n")


if __name__ == "__main__":
    main()

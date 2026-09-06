import time
import math
import threading
import airsim
from pymavlink import mavutil


class ForwardObstacleReader(threading.Thread):
    def __init__(
        self,
        vehicle_name="PX4",
        sensor_name="ForwardDistance",
        obstacle_distance=35.0,
        min_valid_distance=0.5,
        max_valid_distance=120.0,
        persist_frames=2,
        poll_period=0.05,
        max_trust_distance=120.0,
    ):
        super().__init__(daemon=True)
        self.vehicle_name = vehicle_name
        self.sensor_name = sensor_name
        self.obstacle_distance = obstacle_distance
        self.min_valid_distance = min_valid_distance
        self.max_valid_distance = max_valid_distance
        self.persist_frames = max(1, int(persist_frames))
        self.poll_period = poll_period
        self.max_trust_distance = max_trust_distance

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._distance = None
        self._last_good_distance = None
        self._last_time = 0.0
        self._has_data = False
        self._obstacle_counter = 0
        self._obstacle_detected = False
        self._suppress_until = 0.0

    def stop(self):
        self._stop_event.set()

    def suppress_for(self, seconds):
        with self._lock:
            self._suppress_until = max(self._suppress_until, time.time() + max(0.0, float(seconds)))
            self._obstacle_counter = 0
            self._obstacle_detected = False

    def is_suppressed(self):
        with self._lock:
            return time.time() < self._suppress_until

    def has_fresh_data(self, max_age=1.0):
        with self._lock:
            return self._has_data and ((time.time() - self._last_time) <= max_age)

    def get_distance(self):
        with self._lock:
            return self._distance

    def obstacle_ahead(self):
        with self._lock:
            if time.time() < self._suppress_until:
                return False
            return self._obstacle_detected

    def get_last_good_distance(self):
        with self._lock:
            return self._last_good_distance

    def is_path_clear(self, clear_distance):
        with self._lock:
            if time.time() < self._suppress_until:
                return True
            return self._has_data and self._distance is not None and self._distance >= clear_distance

    def get_status(self):
        with self._lock:
            suppressed = time.time() < self._suppress_until
            return {
                "has_data": self._has_data,
                "fresh": self._has_data and ((time.time() - self._last_time) <= 1.0),
                "distance": self._distance,
                "obstacle_detected": (False if suppressed else self._obstacle_detected),
                "suppressed": suppressed,
            }

    def _is_valid_distance(self, dist):
        if dist is None or not math.isfinite(dist):
            return False
        if dist < self.min_valid_distance:
            return False
        if dist > self.max_valid_distance:
            return False
        if dist > self.max_trust_distance:
            return False
        return True

    def run(self):
        debug_tick = 0
        while not self._stop_event.is_set():
            try:
                data = self.client.getDistanceSensorData(
                    distance_sensor_name=self.sensor_name,
                    vehicle_name=self.vehicle_name
                )

                distance = None
                has_data = False
                if data is not None:
                    raw = float(data.distance)
                    if self._is_valid_distance(raw):
                        distance = raw
                        has_data = True

                obstacle_this_frame = has_data and (distance <= self.obstacle_distance)

                with self._lock:
                    self._distance = distance
                    self._has_data = has_data
                    self._last_time = time.time()
                    if has_data:
                        self._last_good_distance = distance

                    if time.time() < self._suppress_until:
                        self._obstacle_counter = 0
                        self._obstacle_detected = False
                    else:
                        if obstacle_this_frame:
                            self._obstacle_counter = min(self.persist_frames, self._obstacle_counter + 1)
                        else:
                            self._obstacle_counter = max(0, self._obstacle_counter - 1)
                        self._obstacle_detected = (self._obstacle_counter >= self.persist_frames)

                debug_tick += 1
                if debug_tick % 20 == 0:
                    dist_str = f"{distance:.1f}m" if distance is not None else "N/A"
                    suppressed = self.is_suppressed()
                    print(f"\n[FWD] dist={dist_str} | obstacle={self.obstacle_ahead()} | suppressed={suppressed}")

            except Exception as e:
                print(f"\n[FWD SENSOR ERROR] {e}")

            time.sleep(self.poll_period)


def _set_speed(mav, speed):
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
        0, 1, speed, -1, 0, 0, 0, 0
    )


def _make_crawl_point(current_pos, target_gps, crawl_m=4.0):
    """Return a short intermediate GPS point a few meters toward the target.
    This avoids a hard stop at the current point while still preventing the drone
    from pushing all the way into the obstacle.
    """
    if not current_pos or not target_gps:
        return None

    lat1 = float(current_pos["lat"])
    lon1 = float(current_pos["lon"])
    lat2 = float(target_gps["lat"])
    lon2 = float(target_gps["lon"])

    # Local flat-earth approximation is enough for a few meters.
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = max(1.0, 111320.0 * math.cos(math.radians(lat1)))

    dn = (lat2 - lat1) * meters_per_deg_lat
    de = (lon2 - lon1) * meters_per_deg_lon
    dist = math.hypot(dn, de)

    if dist < 0.5:
        return {"lat": lat1, "lon": lon1}

    step = min(float(crawl_m), max(1.5, dist * 0.25))
    ratio = min(1.0, step / dist)
    return {
        "lat": lat1 + ((lat2 - lat1) * ratio),
        "lon": lon1 + ((lon2 - lon1) * ratio),
    }


def handle_forward_avoidance(
    target_name,
    target_gps,
    pos_reader,
    streamer,
    ground_reader,
    mav_write,
    obstacle_reader,
    get_believable_agl_fn,
    desired_agl,
    cruise_speed,
    brake_speed,
    hold_speed,
    climb_step_agl,
    max_extra_agl,
    recheck_sec,
    settle_sec,
    active=True,
    clear_frames_required=6,
    clear_distance_margin=5.0,
    extra_clearance_agl=5.0,
    roof_cross_time_sec=10.0,
    post_cross_suppress_sec=6.0,
):
    if not active or obstacle_reader is None:
        return False
    if not obstacle_reader.has_fresh_data() or not obstacle_reader.obstacle_ahead():
        return False

    pos = pos_reader.get_position()
    if not pos:
        return False

    agl = get_believable_agl_fn(pos_reader, ground_reader)
    if agl is not None:
        current_agl = agl
    else:
        last_good_agl = ground_reader.get_last_good_ground_distance()
        current_agl = last_good_agl if last_good_agl is not None else desired_agl

    base_agl = max(current_agl, desired_agl)
    hold_lat = pos["lat"]
    hold_lon = pos["lon"]
    hold_rel = pos["alt_rel"]
    distance = obstacle_reader.get_distance()
    distance_str = f"{distance:.1f}m" if distance is not None else "unknown"

    print(f"\n[AVOID] Obstacle ahead while flying to '{target_name}'")
    print(f"[AVOID] Forward obstacle distance: {distance_str}")
    print("[AVOID] Smooth braking...")

    # Smoother braking: more gradual speed reduction to prevent aggressive pitch changes
    speed_step1 = max(cruise_speed * 0.88, 3.5)
    speed_step2 = max(cruise_speed * 0.72, 2.5)
    speed_step3 = max(cruise_speed * 0.50, 1.5)
    final_hold_speed = max(hold_speed, 1.8)

    _set_speed(mav_write, speed_step1)
    time.sleep(0.5)
    _set_speed(mav_write, speed_step2)
    time.sleep(0.5)
    _set_speed(mav_write, speed_step3)
    time.sleep(0.4)

    pos = pos_reader.get_position()
    crawl_target = None
    if pos:
        hold_lat = pos["lat"]
        hold_lon = pos["lon"]
        hold_rel = pos["alt_rel"]
        crawl_target = _make_crawl_point(pos, target_gps, crawl_m=3.0)

    if crawl_target is not None:
        streamer.set_target(crawl_target["lat"], crawl_target["lon"], hold_rel, target_agl=base_agl)
    else:
        streamer.set_target(target_gps["lat"], target_gps["lon"], hold_rel, target_agl=base_agl)
    time.sleep(0.3)

    _set_speed(mav_write, final_hold_speed)
    time.sleep(0.3)

    streamer.set_target(hold_lat, hold_lon, hold_rel, target_agl=base_agl)
    time.sleep(max(0.8, settle_sec))

    extra_agl = 0.0
    clear_distance = obstacle_reader.obstacle_distance + clear_distance_margin

    while extra_agl <= max_extra_agl:
        if not obstacle_reader.has_fresh_data():
            print("[AVOID] Forward sensor not fresh. Holding current position.")
            streamer.set_target(hold_lat, hold_lon, hold_rel, target_agl=base_agl + extra_agl)
            return True

        clear_frames = 0
        while clear_frames < clear_frames_required:
            if not obstacle_reader.has_fresh_data():
                time.sleep(0.15)
                continue

            pos_now = pos_reader.get_position()
            if pos_now:
                # Keep the good smooth brake entry, but after that stop creeping forward.
                # Hold the captured position while checking/climbing so we do not walk into the wall.
                streamer.set_target(hold_lat, hold_lon, pos_now["alt_rel"], target_agl=base_agl + extra_agl)

            dist_now = obstacle_reader.get_distance()
            blocked_now = obstacle_reader.obstacle_ahead()
            far_enough = (dist_now is not None and dist_now >= clear_distance)
            if (not blocked_now) or far_enough:
                clear_frames += 1
            else:
                clear_frames = 0
                break
            time.sleep(0.15)

        if clear_frames >= clear_frames_required:
            print("[AVOID] Building is clear above the top.")
            break

        extra_agl = min(extra_agl + climb_step_agl, max_extra_agl)
        climb_agl_target = base_agl + extra_agl
        pos_now = pos_reader.get_position()
        if not pos_now:
            return False

        print(f"[AVOID] Building still there. Climbing in place... target AGL={climb_agl_target:.1f}m")
        streamer.set_target(
            hold_lat,
            hold_lon,
            pos_now["alt_rel"] + min(climb_step_agl, 4.0),
            target_agl=climb_agl_target
        )
        time.sleep(recheck_sec)

    clearance_agl = base_agl + extra_agl
    # Add more safety margin - go higher above the detected clearance point
    high_cruise_agl = clearance_agl + extra_clearance_agl + 10.0  # Extra 10m safety buffer
    print(f"[AVOID] Going {extra_clearance_agl + 10.0:.0f}m above clearance. Cross for {roof_cross_time_sec:.0f}s at AGL={high_cruise_agl:.1f}m")

    # IMPORTANT FIX: once we are above the building, ignore forward sensor during the roof-cross
    # and for a few seconds after, so the ground does not retrigger avoidance.
    obstacle_reader.suppress_for(roof_cross_time_sec + post_cross_suppress_sec)

    # Move to target at high altitude with VERY slow speed to prevent aggressive forward tilt
    streamer.set_target(target_gps["lat"], target_gps["lon"], None, target_agl=high_cruise_agl)
    _set_speed(mav_write, 2.0)  # Very slow speed during crossing to prevent tilt-forward collision
    time.sleep(roof_cross_time_sec)

    print(f"[AVOID] {roof_cross_time_sec:.0f}s crossing done. Return to target and maintain {desired_agl:.0f} AGL.")
    pos_resume = pos_reader.get_position()
    resume_rel = pos_resume["alt_rel"] if pos_resume else desired_agl
    streamer.set_target(target_gps["lat"], target_gps["lon"], resume_rel, target_agl=desired_agl)
    _set_speed(mav_write, cruise_speed)
    return True

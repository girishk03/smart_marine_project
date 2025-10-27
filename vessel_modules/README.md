# 🚤 Smart Marine Vessel - Autonomous Navigation Module

## Overview

The Vessel Modules provide autonomous navigation, GPS tracking, and plastic collection capabilities for the Smart Marine Project. This system enables the vessel to automatically detect, navigate to, and collect plastic waste in marine environments.

## Features

### 🗺️ GPS Navigation
- Real-time GPS position tracking
- Automatic pathfinding to detected plastics
- Waypoint-based navigation
- Heading and bearing calculations
- Distance measurement using Haversine formula

### 📹 Camera & Detection
- YOLOv5-based plastic detection
- Relative position calculation (left/center/right)
- Distance estimation from camera
- Target selection and tracking
- Real-time bounding box visualization

### 🗑️ Collection System
- Automatic collection counter
- GPS coordinate logging for each collection
- Timestamp tracking
- Data export (CSV/JSON)
- Session statistics

### 🖥️ Simulation Mode
- Digital twin for testing without hardware
- Virtual boat navigation on map
- Simulated plastic objects
- Realistic camera feed generation
- Safe testing environment

## Installation

### Required Dependencies

```bash
pip install streamlit-folium folium geopy pyyaml numpy opencv-python pandas
```

### File Structure

```
vessel_modules/
├── __init__.py                 # Module initialization
├── camera_module.py            # Camera & detection
├── gps_navigation.py           # GPS & navigation
├── object_counter.py           # Collection counter
├── simulator.py                # Digital twin simulator
├── vessel_config.yaml          # Configuration file
└── README.md                   # This file
```

## Usage

### 1. Simulation Mode (No Hardware Required)

Access the **🚤 Autonomous Mode** tab in the Streamlit app:

1. Click on the 5th tab "🚤 Autonomous Mode"
2. Select "🖥️ Simulation Mode"
3. View the GPS map with boat and plastic markers
4. Click "▶️ Start Autopilot"
5. Watch the boat autonomously navigate and collect plastics

**Features:**
- Interactive GPS map with real-time boat position
- Simulated camera feed with plastic detections
- Automatic navigation to closest plastic
- Collection counter with GPS logging
- Export collection data (CSV/JSON)

### 2. Hardware Mode (Raspberry Pi Required)

**Hardware Requirements:**
- Raspberry Pi 4 or Jetson Nano
- GPS Module (Neo-6M or similar)
- Compass (HMC5883L)
- Motor Driver (L298N)
- Servo for collection mechanism
- Camera (Pi Camera or USB)
- Ultrasonic sensors for obstacle detection

**Setup:**
1. Connect hardware according to pin configuration in `vessel_config.yaml`
2. Install additional dependencies:
   ```bash
   pip install RPi.GPIO pyserial smbus2
   ```
3. Configure GPIO pins in `vessel_config.yaml`
4. Test in simulation mode first
5. Switch to Hardware Mode in the app

## Configuration

Edit `vessel_config.yaml` to customize:

### Simulation Settings
```yaml
simulation:
  map_center_lat: 13.0827      # Starting latitude
  map_center_lon: 80.2707      # Starting longitude
  boat_speed_mps: 1.5          # Boat speed (m/s)
  detection_range_m: 50        # Detection range
  collection_range_m: 2        # Collection range
  spawn_plastic_count: 10      # Number of plastics
```

### Navigation Parameters
```yaml
navigation:
  waypoint_threshold_m: 2.0    # Distance to consider waypoint reached
  heading_tolerance_deg: 5.0   # Acceptable heading error
  max_speed_mps: 2.0           # Maximum speed
  obstacle_distance_m: 3.0     # Obstacle avoidance distance
```

### Detection Settings
```yaml
detection:
  confidence_threshold: 0.15   # Detection confidence
  iou_threshold: 0.40          # NMS IoU threshold
  camera_fov_deg: 60           # Camera field of view
  camera_range_m: 30           # Max detection distance
```

## Module Documentation

### VesselCamera

Handles camera feed processing and plastic detection.

```python
from vessel_modules import VesselCamera

camera = VesselCamera(model, camera_fov_deg=60, camera_range_m=30)
result = camera.detect_and_track(frame, conf_threshold=0.15)

# Returns:
# {
#     'detections': [...],
#     'target': {...},
#     'navigation_command': 'forward',
#     'frame': annotated_frame
# }
```

### GPSNavigator

Manages GPS tracking and navigation calculations.

```python
from vessel_modules import GPSNavigator

navigator = GPSNavigator(config)
navigator.update_position(latitude, longitude)
navigator.set_target(target_lat, target_lon)

nav_command = navigator.get_navigation_command()
# Returns: {'command': 'forward', 'heading': 45.0, 'speed': 1.5, ...}
```

### CollectionCounter

Tracks and logs collected plastics.

```python
from vessel_modules import CollectionCounter

counter = CollectionCounter(log_directory="vessel_logs")
counter.add_collection(gps_lat=13.08, gps_lon=80.27, confidence=0.9)

stats = counter.get_statistics()
counter.save_to_csv()
counter.save_to_json()
```

### VesselSimulator

Digital twin for testing without hardware.

```python
from vessel_modules import VesselSimulator

sim = VesselSimulator(config)
sim.update(dt=0.1, command='forward', target_heading=45.0)

visible_plastics = sim.get_visible_plastics()
camera_frame = sim.generate_camera_frame(visible_plastics)
collected = sim.try_collect_plastic()
```

## How It Works

### Autonomous Navigation Flow

1. **Detection**: Camera detects plastics in view
2. **Target Selection**: System selects closest/best target
3. **Navigation**: Calculate bearing and distance to target
4. **Movement**: Send motor commands to approach target
5. **Collection**: Deploy net when within collection range
6. **Logging**: Record GPS coordinates and timestamp
7. **Repeat**: Search for next target

### Navigation Commands

- `forward` - Move straight ahead
- `turn_left` - Turn left to align with target
- `turn_right` - Turn right to align with target
- `collect` - Deploy collection mechanism
- `stop` - Stop all movement
- `search` - Search pattern when no target

## Data Export

### Collection Log Format (CSV)
```csv
id,timestamp,gps_latitude,gps_longitude,confidence,session_time_seconds
1,2025-01-20T10:30:00,13.0827,80.2707,0.92,45.3
2,2025-01-20T10:31:15,13.0829,80.2709,0.88,120.7
```

### Statistics Export (JSON)
```json
{
  "session_start": "2025-01-20T10:00:00",
  "total_collections": 15,
  "statistics": {
    "total_count": 15,
    "session_duration_minutes": 45.2,
    "collections_per_hour": 19.9,
    "avg_confidence": 0.87
  }
}
```

## Troubleshooting

### "Vessel modules not fully available"
- Ensure all files are in `vessel_modules/` folder
- Check that `__init__.py` exists
- Verify Python can import the modules

### "Install map dependencies"
```bash
pip install streamlit-folium folium geopy
```

### Simulation not starting
- Check `vessel_config.yaml` exists
- Verify GPS coordinates are valid
- Ensure PyYAML is installed

### GPS map not displaying
- Install folium: `pip install folium streamlit-folium`
- Check internet connection (map tiles require internet)
- Verify GPS coordinates are reasonable

## Performance Tips

### Simulation Mode
- Adjust `boat_speed_mps` for faster/slower simulation
- Reduce `spawn_plastic_count` for simpler scenarios
- Increase `detection_range_m` for wider search area

### Hardware Mode
- Use GPS with external antenna for better accuracy
- Calibrate compass before deployment
- Test motor drivers with low voltage first
- Add battery voltage monitoring

## Safety Considerations

⚠️ **Important Safety Notes:**

1. **Always test in simulation first** before hardware deployment
2. **Monitor battery levels** during operation
3. **Include emergency stop mechanism** (hardware kill switch)
4. **Test in controlled environment** before open water
5. **Have manual override** capability at all times
6. **Check local regulations** for autonomous vessels
7. **Ensure proper waterproofing** of electronics
8. **Use appropriate safety equipment** (flotation, lights)

## Future Enhancements

Planned features:
- [ ] Multi-camera support
- [ ] Advanced obstacle avoidance (LIDAR)
- [ ] Swarm coordination (multiple vessels)
- [ ] Machine learning for optimal path planning
- [ ] Solar panel integration
- [ ] Cellular/satellite communication
- [ ] Weather condition monitoring
- [ ] Automatic return-to-base

## Contributing

To add new features:
1. Create new module in `vessel_modules/`
2. Update `__init__.py` to export new classes
3. Add configuration to `vessel_config.yaml`
4. Update this README
5. Test in simulation mode first

## License

Part of the Smart Marine Project - Marine Conservation Initiative

## Support

For issues or questions:
- Check this README
- Review `vessel_config.yaml` settings
- Test in simulation mode
- Check console output for errors

---

**Built for Marine Conservation** 🌊🚤♻️

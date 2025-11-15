"""
GPS Navigator - Location-Aware Navigation Module for Beyond Sight

Provides:
- Real-time GPS location tracking
- Direction and distance calculations
- Location-based context awareness
- Nearby landmark detection
- Route planning and navigation
"""

import logging
import threading
import json
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import math

# Try to import geopy for advanced location features
try:
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

# Try to import GPS library
try:
    import pynmea2
    PYNMEA_AVAILABLE = True
except ImportError:
    PYNMEA_AVAILABLE = False

logger = logging.getLogger(__name__)


class Direction(Enum):
    """Cardinal and intercardinal directions."""
    NORTH = "North"
    NORTHEAST = "Northeast"
    EAST = "East"
    SOUTHEAST = "Southeast"
    SOUTH = "South"
    SOUTHWEST = "Southwest"
    WEST = "West"
    NORTHWEST = "Northwest"


@dataclass
class GPSLocation:
    """Data class for GPS location."""
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    accuracy: Optional[float] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def distance_to(self, other: 'GPSLocation') -> float:
        """
        Calculate distance to another location in kilometers.
        Uses Haversine formula for great-circle distance.
        """
        R = 6371  # Earth's radius in km
        
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def bearing_to(self, other: 'GPSLocation') -> float:
        """
        Calculate bearing (compass direction) to another location.
        Returns degrees (0-360, where 0=North, 90=East, etc.)
        """
        lat1 = math.radians(self.latitude)
        lat2 = math.radians(other.latitude)
        dlon = math.radians(other.longitude - self.longitude)
        
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        
        bearing = math.atan2(x, y)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing


class GPSNavigator:
    """
    GPS Navigator for location-aware navigation in Beyond Sight.
    
    Provides real-time location tracking, direction calculation,
    and navigation guidance.
    """
    
    def __init__(self):
        """Initialize GPS Navigator."""
        logger.info("Initializing GPS Navigator...")
        
        # Location state
        self.current_location: Optional[GPSLocation] = None
        self.previous_location: Optional[GPSLocation] = None
        self.home_location: Optional[GPSLocation] = None
        self.location_history: List[GPSLocation] = []
        
        # Waypoints and destinations
        self.waypoints: List[Dict] = []
        self.current_waypoint_index = 0
        self.destination: Optional[GPSLocation] = None
        
        # Threading
        self.location_lock = threading.Lock()
        self.is_tracking = False
        self.tracking_thread: Optional[threading.Thread] = None
        
        # Geolocation service
        if GEOPY_AVAILABLE:
            try:
                self.geocoder = Nominatim(user_agent="beyond_sight_navigator")
                self.geocoding_available = True
            except Exception as e:
                logger.warning(f"Geocoding service unavailable: {e}")
                self.geocoding_available = False
        else:
            self.geocoding_available = False
        
        # Nearby landmarks cache
        self.nearby_landmarks: List[Dict] = []
        self.landmark_update_time = 0
        
        # Statistics
        self.total_distance_traveled = 0.0
        self.location_updates = 0
        self.start_time: Optional[datetime] = None
        
        logger.info("✅ GPS Navigator initialized")
    
    def set_home_location(self, latitude: float, longitude: float) -> bool:
        """
        Set home location for navigation reference.
        
        Args:
            latitude: Home latitude
            longitude: Home longitude
            
        Returns:
            bool: Success status
        """
        try:
            with self.location_lock:
                self.home_location = GPSLocation(
                    latitude=latitude,
                    longitude=longitude,
                    timestamp=datetime.now().isoformat()
                )
            
            logger.info(f"Home location set: {latitude}, {longitude}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting home location: {e}")
            return False
    
    def update_location(self, latitude: float, longitude: float, 
                       altitude: Optional[float] = None,
                       accuracy: Optional[float] = None) -> bool:
        """
        Update current GPS location.
        
        Args:
            latitude: Current latitude
            longitude: Current longitude
            altitude: Optional altitude in meters
            accuracy: Optional accuracy in meters
            
        Returns:
            bool: Success status
        """
        try:
            with self.location_lock:
                # Save previous location
                if self.current_location:
                    self.previous_location = self.current_location
                    
                    # Calculate distance traveled
                    new_location = GPSLocation(
                        latitude=latitude,
                        longitude=longitude,
                        altitude=altitude,
                        accuracy=accuracy,
                        timestamp=datetime.now().isoformat()
                    )
                    distance = self.current_location.distance_to(new_location)
                    self.total_distance_traveled += distance
                
                # Update current location
                self.current_location = GPSLocation(
                    latitude=latitude,
                    longitude=longitude,
                    altitude=altitude,
                    accuracy=accuracy,
                    timestamp=datetime.now().isoformat()
                )
                
                # Track location history (keep last 100)
                self.location_history.append(self.current_location)
                if len(self.location_history) > 100:
                    self.location_history.pop(0)
                
                self.location_updates += 1
                
                if self.start_time is None:
                    self.start_time = datetime.now()
            
            logger.debug(f"Location updated: {latitude}, {longitude}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating location: {e}")
            return False
    
    def get_current_location(self) -> Optional[Dict]:
        """
        Get current GPS location.
        
        Returns:
            dict: Current location data or None
        """
        try:
            with self.location_lock:
                if self.current_location:
                    return {
                        'latitude': self.current_location.latitude,
                        'longitude': self.current_location.longitude,
                        'altitude': self.current_location.altitude,
                        'accuracy': self.current_location.accuracy,
                        'timestamp': self.current_location.timestamp
                    }
            return None
            
        except Exception as e:
            logger.error(f"Error getting current location: {e}")
            return None
    
    def get_distance_to_destination(self) -> Optional[float]:
        """
        Get distance to current destination in kilometers.
        
        Returns:
            float: Distance in km or None
        """
        try:
            with self.location_lock:
                if self.current_location and self.destination:
                    return self.current_location.distance_to(self.destination)
            return None
            
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return None
    
    def get_distance_to_home(self) -> Optional[float]:
        """
        Get distance to home location in kilometers.
        
        Returns:
            float: Distance in km or None
        """
        try:
            with self.location_lock:
                if self.current_location and self.home_location:
                    return self.current_location.distance_to(self.home_location)
            return None
            
        except Exception as e:
            logger.error(f"Error calculating distance to home: {e}")
            return None
    
    def get_bearing_to_destination(self) -> Optional[Tuple[float, str]]:
        """
        Get bearing (compass direction) to destination.
        
        Returns:
            tuple: (bearing_degrees, direction_name) or None
        """
        try:
            with self.location_lock:
                if self.current_location and self.destination:
                    bearing = self.current_location.bearing_to(self.destination)
                    direction = self._bearing_to_direction(bearing)
                    return bearing, direction
            return None
            
        except Exception as e:
            logger.error(f"Error calculating bearing: {e}")
            return None
    
    def get_bearing_to_home(self) -> Optional[Tuple[float, str]]:
        """
        Get bearing (compass direction) to home.
        
        Returns:
            tuple: (bearing_degrees, direction_name) or None
        """
        try:
            with self.location_lock:
                if self.current_location and self.home_location:
                    bearing = self.current_location.bearing_to(self.home_location)
                    direction = self._bearing_to_direction(bearing)
                    return bearing, direction
            return None
            
        except Exception as e:
            logger.error(f"Error calculating bearing to home: {e}")
            return None
    
    def _bearing_to_direction(self, bearing: float) -> str:
        """Convert bearing degrees to cardinal direction."""
        directions = [
            (348.75, Direction.NORTH.value),
            (11.25, Direction.NORTHEAST.value),
            (33.75, Direction.NORTHEAST.value),
            (56.25, Direction.EAST.value),
            (78.75, Direction.SOUTHEAST.value),
            (101.25, Direction.SOUTHEAST.value),
            (123.75, Direction.SOUTH.value),
            (146.25, Direction.SOUTHWEST.value),
            (168.75, Direction.SOUTHWEST.value),
            (191.25, Direction.WEST.value),
            (213.75, Direction.NORTHWEST.value),
            (236.25, Direction.NORTHWEST.value),
            (258.75, Direction.NORTH.value),
            (281.25, Direction.NORTHEAST.value),
            (303.75, Direction.EAST.value),
            (326.25, Direction.SOUTHEAST.value),
        ]
        
        for threshold, direction in directions:
            if bearing < threshold:
                return direction
        
        return Direction.NORTH.value
    
    def set_destination(self, latitude: float, longitude: float) -> bool:
        """
        Set navigation destination.
        
        Args:
            latitude: Destination latitude
            longitude: Destination longitude
            
        Returns:
            bool: Success status
        """
        try:
            with self.location_lock:
                self.destination = GPSLocation(
                    latitude=latitude,
                    longitude=longitude
                )
                self.current_waypoint_index = 0
            
            logger.info(f"Destination set: {latitude}, {longitude}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting destination: {e}")
            return False
    
    def clear_destination(self) -> bool:
        """Clear current destination."""
        try:
            with self.location_lock:
                self.destination = None
                self.waypoints = []
                self.current_waypoint_index = 0
            
            logger.info("Destination cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing destination: {e}")
            return False
    
    def add_waypoint(self, latitude: float, longitude: float, 
                    name: str = "") -> bool:
        """
        Add waypoint to route.
        
        Args:
            latitude: Waypoint latitude
            longitude: Waypoint longitude
            name: Optional waypoint name
            
        Returns:
            bool: Success status
        """
        try:
            with self.location_lock:
                self.waypoints.append({
                    'latitude': latitude,
                    'longitude': longitude,
                    'name': name,
                    'reached': False
                })
            
            logger.info(f"Waypoint added: {name or f'({latitude}, {longitude})'}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding waypoint: {e}")
            return False
    
    def get_navigation_guidance(self) -> Optional[str]:
        """
        Get natural language navigation guidance.
        
        Returns:
            str: Navigation instructions or None
        """
        try:
            if not self.current_location or not self.destination:
                return "Destination not set. Please set a destination first."
            
            distance = self.current_location.distance_to(self.destination)
            bearing_result = self.get_bearing_to_destination()
            
            if not bearing_result:
                return "Unable to calculate bearing to destination."
            
            bearing, direction = bearing_result
            
            # Format distance in appropriate units
            if distance < 1:
                distance_text = f"{distance * 1000:.0f} meters"
            else:
                distance_text = f"{distance:.1f} kilometers"
            
            # Generate guidance
            guidance = f"Your destination is {distance_text} to the {direction.lower()}."
            
            # Add waypoint info if available
            if self.waypoints:
                unreached = [w for w in self.waypoints if not w['reached']]
                if unreached:
                    next_wp = unreached[0]
                    guidance += f" Next waypoint: {next_wp.get('name', 'Unnamed')}."
            
            return guidance
            
        except Exception as e:
            logger.error(f"Error generating navigation guidance: {e}")
            return None
    
    def get_location_context(self) -> Optional[str]:
        """
        Get human-readable location context for scene analysis.
        
        Returns:
            str: Location context or None
        """
        try:
            if not self.current_location:
                return "Location: Unknown"
            
            context_parts = []
            
            # Current location
            context_parts.append(f"Location: {self.current_location.latitude:.4f}, "
                                f"{self.current_location.longitude:.4f}")
            
            # Distance to home
            if self.home_location:
                distance_home = self.get_distance_to_home()
                if distance_home:
                    bearing_home = self.get_bearing_to_home()
                    direction = bearing_home[1] if bearing_home else "unknown"
                    context_parts.append(f"Home is {distance_home:.1f} km {direction.lower()}")
            
            # Distance to destination
            if self.destination:
                distance_dest = self.get_distance_to_destination()
                if distance_dest:
                    bearing_dest = self.get_bearing_to_destination()
                    direction = bearing_dest[1] if bearing_dest else "unknown"
                    context_parts.append(f"Destination is {distance_dest:.1f} km {direction.lower()}")
            
            # Altitude if available
            if self.current_location.altitude:
                context_parts.append(f"Elevation: {self.current_location.altitude:.0f}m")
            
            return ". ".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error generating location context: {e}")
            return None
    
    def get_statistics(self) -> Dict:
        """
        Get navigation statistics.
        
        Returns:
            dict: Navigation stats
        """
        try:
            with self.location_lock:
                elapsed_time = None
                if self.start_time:
                    elapsed_time = (datetime.now() - self.start_time).total_seconds()
                
                return {
                    'location_updates': self.location_updates,
                    'total_distance_traveled': round(self.total_distance_traveled, 2),
                    'location_history_count': len(self.location_history),
                    'elapsed_time_seconds': elapsed_time,
                    'has_destination': self.destination is not None,
                    'has_home_location': self.home_location is not None,
                    'waypoints_count': len(self.waypoints)
                }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def reset(self) -> bool:
        """Reset navigator to initial state."""
        try:
            with self.location_lock:
                self.current_location = None
                self.previous_location = None
                self.waypoints = []
                self.destination = None
                self.location_history = []
                self.total_distance_traveled = 0.0
                self.location_updates = 0
                self.start_time = None
                self.current_waypoint_index = 0
            
            logger.info("GPS Navigator reset")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting navigator: {e}")
            return False


# Global navigator instance
_navigator: Optional[GPSNavigator] = None


def get_navigator() -> GPSNavigator:
    """Get or create GPS Navigator instance."""
    global _navigator
    if _navigator is None:
        _navigator = GPSNavigator()
    return _navigator

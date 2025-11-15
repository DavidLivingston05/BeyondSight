"""
Navigation Module - GPS and Place Memory
Location-aware navigation and saved places management
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import geopy
try:
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

# ============= Enums =============

class Direction(Enum):
    """Cardinal directions."""
    NORTH = "North"
    NORTHEAST = "Northeast"
    EAST = "East"
    SOUTHEAST = "Southeast"
    SOUTH = "South"
    SOUTHWEST = "Southwest"
    WEST = "West"
    NORTHWEST = "Northwest"


# ============= Data Classes =============

@dataclass
class GPSLocation:
    """GPS location data."""
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    accuracy: Optional[float] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def distance_to(self, other: 'GPSLocation') -> float:
        """Distance to another location in km."""
        R = 6371  # Earth radius
        
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c


@dataclass
class SavedPlace:
    """Saved place/location."""
    name: str
    latitude: float
    longitude: float
    description: str = ""
    tags: List[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SavedPlace':
        return cls(**data)


# ============= GPS Navigator =============

class GPSNavigator:
    """GPS location and navigation management."""
    
    def __init__(self):
        """Initialize GPS navigator."""
        self.current_location: Optional[GPSLocation] = None
        self.last_location: Optional[GPSLocation] = None
        self.trip_distance = 0.0
        logger.info("GPS Navigator initialized")
    
    def update_location(self, latitude: float, longitude: float, 
                       altitude: Optional[float] = None,
                       accuracy: Optional[float] = None) -> None:
        """Update current location."""
        try:
            new_location = GPSLocation(
                latitude=latitude,
                longitude=longitude,
                altitude=altitude,
                accuracy=accuracy,
                timestamp=datetime.now().isoformat()
            )
            
            if self.current_location:
                self.last_location = self.current_location
                self.trip_distance += self.current_location.distance_to(new_location)
            
            self.current_location = new_location
            logger.debug(f"Location updated: {latitude}, {longitude}")
        except Exception as e:
            logger.error(f"Location update error: {e}")
    
    def get_distance_to_location(self, latitude: float, longitude: float) -> Optional[float]:
        """Get distance to a location."""
        if not self.current_location:
            return None
        
        try:
            target = GPSLocation(latitude=latitude, longitude=longitude)
            return self.current_location.distance_to(target)
        except Exception as e:
            logger.error(f"Distance calculation error: {e}")
            return None
    
    def get_bearing_to_location(self, latitude: float, longitude: float) -> Optional[float]:
        """Get bearing to a location."""
        if not self.current_location:
            return None
        
        try:
            lat1 = math.radians(self.current_location.latitude)
            lat2 = math.radians(latitude)
            lon1 = math.radians(self.current_location.longitude)
            lon2 = math.radians(longitude)
            
            dlon = lon2 - lon1
            y = math.sin(dlon) * math.cos(lat2)
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
            bearing = math.degrees(math.atan2(y, x))
            
            return (bearing + 360) % 360
        except Exception as e:
            logger.error(f"Bearing calculation error: {e}")
            return None
    
    def bearing_to_direction(self, bearing: float) -> str:
        """Convert bearing to cardinal direction."""
        if bearing < 22.5 or bearing >= 337.5:
            return "North"
        elif bearing < 67.5:
            return "Northeast"
        elif bearing < 112.5:
            return "East"
        elif bearing < 157.5:
            return "Southeast"
        elif bearing < 202.5:
            return "South"
        elif bearing < 247.5:
            return "Southwest"
        elif bearing < 292.5:
            return "West"
        else:
            return "Northwest"


# ============= Place Memory =============

class PlaceMemory:
    """Saved places management."""
    
    PLACES_FILE = Path("saved_places.json")
    
    def __init__(self, storage_file: Path = PLACES_FILE):
        """Initialize place memory."""
        self.storage_file = storage_file
        self.places: Dict[str, SavedPlace] = {}
        self._load_places()
        logger.info(f"Place Memory loaded with {len(self.places)} places")
    
    def _load_places(self) -> None:
        """Load places from file."""
        try:
            if self.storage_file.exists():
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                    for name, place_data in data.items():
                        self.places[name] = SavedPlace.from_dict(place_data)
        except Exception as e:
            logger.error(f"Load places error: {e}")
    
    def _save_places(self) -> None:
        """Save places to file."""
        try:
            data = {name: place.to_dict() for name, place in self.places.items()}
            with open(self.storage_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Save places error: {e}")
    
    def save_place(self, name: str, latitude: float, longitude: float,
                   description: str = "", tags: List[str] = None) -> bool:
        """Save a new place."""
        try:
            place = SavedPlace(
                name=name,
                latitude=latitude,
                longitude=longitude,
                description=description,
                tags=tags or []
            )
            self.places[name] = place
            self._save_places()
            logger.info(f"Saved place: {name}")
            return True
        except Exception as e:
            logger.error(f"Save place error: {e}")
            return False
    
    def get_place(self, name: str) -> Optional[SavedPlace]:
        """Get saved place."""
        return self.places.get(name)
    
    def list_places(self) -> List[str]:
        """List all saved places."""
        return list(self.places.keys())
    
    def delete_place(self, name: str) -> bool:
        """Delete a saved place."""
        try:
            if name in self.places:
                del self.places[name]
                self._save_places()
                logger.info(f"Deleted place: {name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Delete place error: {e}")
            return False
    
    def search_places(self, query: str) -> List[SavedPlace]:
        """Search places by name or tag."""
        query = query.lower()
        results = []
        
        for place in self.places.values():
            if query in place.name.lower():
                results.append(place)
            elif any(query in tag.lower() for tag in place.tags):
                results.append(place)
        
        return results


# ============= Factory Functions =============

_navigator_instance: Optional[GPSNavigator] = None
_place_memory_instance: Optional[PlaceMemory] = None


def get_navigator() -> GPSNavigator:
    """Get GPS navigator instance."""
    global _navigator_instance
    if _navigator_instance is None:
        _navigator_instance = GPSNavigator()
    return _navigator_instance


def get_place_memory() -> PlaceMemory:
    """Get place memory instance."""
    global _place_memory_instance
    if _place_memory_instance is None:
        _place_memory_instance = PlaceMemory()
    return _place_memory_instance

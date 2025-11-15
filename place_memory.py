"""
Place Memory - Named Location Storage & Management for Beyond Sight

Provides:
- Save favorite/named locations
- Retrieve saved places
- Distance to saved places
- Search places by name
- Remove places
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

# Storage path
PLACES_FILE = Path("saved_places.json")


@dataclass
class SavedPlace:
    """Data class for a saved location."""
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
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SavedPlace':
        """Create from dictionary."""
        return cls(**data)


class PlaceMemory:
    """
    Place Memory for saving and retrieving named locations.
    
    Allows users to save favorite places and retrieve them with
    distance calculations and search functionality.
    """
    
    def __init__(self, storage_file: Path = PLACES_FILE):
        """Initialize Place Memory."""
        logger.info("Initializing Place Memory...")
        
        self.storage_file = storage_file
        self.places: Dict[str, SavedPlace] = {}
        self.lock = threading.Lock()
        
        # Load existing places
        self._load_places()
        
        logger.info(f"✅ Place Memory initialized with {len(self.places)} saved places")
    
    def _load_places(self) -> bool:
        """Load places from storage file."""
        try:
            if self.storage_file.exists():
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                    self.places = {
                        name: SavedPlace.from_dict(place_data)
                        for name, place_data in data.items()
                    }
                logger.info(f"Loaded {len(self.places)} places from storage")
                return True
            else:
                logger.info("No saved places file found, starting fresh")
                return True
                
        except Exception as e:
            logger.error(f"Error loading places: {e}")
            return False
    
    def _save_places(self) -> bool:
        """Save places to storage file."""
        try:
            with open(self.storage_file, 'w') as f:
                data = {
                    name: place.to_dict()
                    for name, place in self.places.items()
                }
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self.places)} places to storage")
            return True
            
        except Exception as e:
            logger.error(f"Error saving places: {e}")
            return False
    
    def save_place(self, name: str, latitude: float, longitude: float,
                   description: str = "", tags: List[str] = None) -> bool:
        """
        Save a new named place.
        
        Args:
            name: Place name (must be unique)
            latitude: Place latitude
            longitude: Place longitude
            description: Optional place description
            tags: Optional tags for categorization
            
        Returns:
            bool: Success status
        """
        try:
            with self.lock:
                if name in self.places:
                    logger.warning(f"Place '{name}' already exists, updating...")
                
                place = SavedPlace(
                    name=name,
                    latitude=latitude,
                    longitude=longitude,
                    description=description,
                    tags=tags or []
                )
                
                self.places[name] = place
                self._save_places()
            
            logger.info(f"✅ Saved place: '{name}' at ({latitude}, {longitude})")
            return True
            
        except Exception as e:
            logger.error(f"Error saving place: {e}")
            return False
    
    def get_place(self, name: str) -> Optional[Dict]:
        """
        Retrieve a saved place by name.
        
        Args:
            name: Place name
            
        Returns:
            dict: Place data or None if not found
        """
        try:
            with self.lock:
                if name in self.places:
                    return self.places[name].to_dict()
            return None
            
        except Exception as e:
            logger.error(f"Error getting place: {e}")
            return None
    
    def get_all_places(self) -> List[Dict]:
        """
        Get all saved places.
        
        Returns:
            list: All saved places
        """
        try:
            with self.lock:
                return [place.to_dict() for place in self.places.values()]
                
        except Exception as e:
            logger.error(f"Error getting places: {e}")
            return []
    
    def delete_place(self, name: str) -> bool:
        """
        Delete a saved place.
        
        Args:
            name: Place name
            
        Returns:
            bool: Success status
        """
        try:
            with self.lock:
                if name in self.places:
                    del self.places[name]
                    self._save_places()
                    logger.info(f"Deleted place: '{name}'")
                    return True
                else:
                    logger.warning(f"Place '{name}' not found")
                    return False
                    
        except Exception as e:
            logger.error(f"Error deleting place: {e}")
            return False
    
    def search_places(self, query: str) -> List[Dict]:
        """
        Search places by name or description.
        
        Args:
            query: Search query (case-insensitive)
            
        Returns:
            list: Matching places
        """
        try:
            query_lower = query.lower()
            results = []
            
            with self.lock:
                for place in self.places.values():
                    if (query_lower in place.name.lower() or
                        query_lower in place.description.lower() or
                        any(query_lower in tag.lower() for tag in place.tags)):
                        results.append(place.to_dict())
            
            logger.info(f"Found {len(results)} places matching '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error searching places: {e}")
            return []
    
    def get_places_by_tag(self, tag: str) -> List[Dict]:
        """
        Get all places with a specific tag.
        
        Args:
            tag: Tag name
            
        Returns:
            list: Places with the tag
        """
        try:
            results = []
            tag_lower = tag.lower()
            
            with self.lock:
                for place in self.places.values():
                    if any(t.lower() == tag_lower for t in place.tags):
                        results.append(place.to_dict())
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting places by tag: {e}")
            return []
    
    def get_distance_to_place(self, place_name: str, current_lat: float,
                             current_lon: float) -> Optional[float]:
        """
        Calculate distance to a saved place.
        
        Args:
            place_name: Name of saved place
            current_lat: Current latitude
            current_lon: Current longitude
            
        Returns:
            float: Distance in km or None
        """
        try:
            place = self.places.get(place_name)
            if not place:
                return None
            
            # Haversine formula
            import math
            R = 6371  # Earth's radius in km
            
            lat1, lon1 = math.radians(current_lat), math.radians(current_lon)
            lat2, lon2 = math.radians(place.latitude), math.radians(place.longitude)
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            
            distance = R * c
            logger.debug(f"Distance to '{place_name}': {distance:.2f} km")
            
            return distance
            
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return None
    
    def get_nearest_place(self, current_lat: float,
                         current_lon: float) -> Optional[Dict]:
        """
        Find the nearest saved place.
        
        Args:
            current_lat: Current latitude
            current_lon: Current longitude
            
        Returns:
            dict: Nearest place with distance or None
        """
        try:
            if not self.places:
                return None
            
            nearest = None
            min_distance = float('inf')
            
            with self.lock:
                for place_name, place in self.places.items():
                    distance = self.get_distance_to_place(
                        place_name, current_lat, current_lon
                    )
                    
                    if distance and distance < min_distance:
                        min_distance = distance
                        nearest = {
                            **place.to_dict(),
                            'distance_km': round(distance, 2)
                        }
            
            return nearest
            
        except Exception as e:
            logger.error(f"Error finding nearest place: {e}")
            return None
    
    def get_places_within_radius(self, current_lat: float, current_lon: float,
                                radius_km: float) -> List[Dict]:
        """
        Get all places within a given radius.
        
        Args:
            current_lat: Current latitude
            current_lon: Current longitude
            radius_km: Search radius in kilometers
            
        Returns:
            list: Places within radius, sorted by distance
        """
        try:
            results = []
            
            with self.lock:
                for place_name, place in self.places.items():
                    distance = self.get_distance_to_place(
                        place_name, current_lat, current_lon
                    )
                    
                    if distance and distance <= radius_km:
                        results.append({
                            **place.to_dict(),
                            'distance_km': round(distance, 2)
                        })
            
            # Sort by distance
            results.sort(key=lambda x: x['distance_km'])
            
            logger.info(f"Found {len(results)} places within {radius_km}km")
            return results
            
        except Exception as e:
            logger.error(f"Error getting places within radius: {e}")
            return []
    
    def clear_all_places(self) -> bool:
        """Clear all saved places."""
        try:
            with self.lock:
                self.places.clear()
                self._save_places()
            
            logger.warning("All places cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing places: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get place memory statistics."""
        try:
            with self.lock:
                return {
                    'total_places': len(self.places),
                    'places': list(self.places.keys()),
                    'file_path': str(self.storage_file)
                }
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}


# Global place memory instance
_place_memory: Optional[PlaceMemory] = None


def get_place_memory() -> PlaceMemory:
    """Get or create Place Memory instance."""
    global _place_memory
    if _place_memory is None:
        _place_memory = PlaceMemory()
    return _place_memory

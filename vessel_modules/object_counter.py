"""
Collection Counter Module - Track and Log Collected Plastics
============================================================

Manages collection counting, logging, and data export for the
Smart Marine Vessel.
"""

import json
import csv
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd


class CollectionCounter:
    """
    Counter and logger for collected plastic items
    """
    
    def __init__(self, log_directory: str = "vessel_logs"):
        """
        Initialize collection counter
        
        Args:
            log_directory: Directory to save logs
        """
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        
        self.collection_count = 0
        self.collections = []
        self.session_start = datetime.now()
        
    def add_collection(self, gps_lat: Optional[float] = None, 
                      gps_lon: Optional[float] = None,
                      confidence: Optional[float] = None,
                      detection_data: Optional[Dict] = None):
        """
        Record a new plastic collection
        
        Args:
            gps_lat: GPS latitude of collection
            gps_lon: GPS longitude of collection
            confidence: Detection confidence score
            detection_data: Additional detection information
        """
        self.collection_count += 1
        
        collection_record = {
            'id': self.collection_count,
            'timestamp': datetime.now().isoformat(),
            'gps_latitude': gps_lat,
            'gps_longitude': gps_lon,
            'confidence': confidence,
            'session_time_seconds': (datetime.now() - self.session_start).total_seconds()
        }
        
        if detection_data:
            collection_record.update(detection_data)
        
        self.collections.append(collection_record)
        
        # Auto-save every 10 collections
        if self.collection_count % 10 == 0:
            self.save_to_file()
    
    def get_count(self) -> int:
        """Get total collection count"""
        return self.collection_count
    
    def get_recent_collections(self, n: int = 10) -> List[Dict]:
        """
        Get most recent collections
        
        Args:
            n: Number of recent collections to return
            
        Returns:
            List of collection records
        """
        return self.collections[-n:]
    
    def get_all_collections(self) -> List[Dict]:
        """Get all collection records"""
        return self.collections
    
    def get_statistics(self) -> Dict:
        """
        Calculate collection statistics
        
        Returns:
            Dictionary with statistics
        """
        if not self.collections:
            return {
                'total_count': 0,
                'session_duration_minutes': 0,
                'collections_per_hour': 0,
                'avg_confidence': 0,
                'locations_count': 0
            }
        
        session_duration = (datetime.now() - self.session_start).total_seconds() / 60
        collections_per_hour = (self.collection_count / session_duration * 60) if session_duration > 0 else 0
        
        # Calculate average confidence
        confidences = [c['confidence'] for c in self.collections if c.get('confidence')]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Count unique locations
        locations = set()
        for c in self.collections:
            if c.get('gps_latitude') and c.get('gps_longitude'):
                locations.add((c['gps_latitude'], c['gps_longitude']))
        
        return {
            'total_count': self.collection_count,
            'session_duration_minutes': round(session_duration, 2),
            'collections_per_hour': round(collections_per_hour, 2),
            'avg_confidence': round(avg_confidence, 3),
            'locations_count': len(locations),
            'first_collection': self.collections[0]['timestamp'] if self.collections else None,
            'last_collection': self.collections[-1]['timestamp'] if self.collections else None
        }
    
    def save_to_csv(self, filename: Optional[str] = None) -> str:
        """
        Save collections to CSV file
        
        Args:
            filename: Custom filename (optional)
            
        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"collections_{timestamp}.csv"
        
        filepath = self.log_directory / filename
        
        if not self.collections:
            return str(filepath)
        
        # Convert to DataFrame for easy CSV export
        df = pd.DataFrame(self.collections)
        df.to_csv(filepath, index=False)
        
        return str(filepath)
    
    def save_to_json(self, filename: Optional[str] = None) -> str:
        """
        Save collections to JSON file
        
        Args:
            filename: Custom filename (optional)
            
        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"collections_{timestamp}.json"
        
        filepath = self.log_directory / filename
        
        data = {
            'session_start': self.session_start.isoformat(),
            'session_end': datetime.now().isoformat(),
            'total_collections': self.collection_count,
            'statistics': self.get_statistics(),
            'collections': self.collections
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(filepath)
    
    def save_to_file(self, format: str = 'json') -> str:
        """
        Save collections to file (auto-named)
        
        Args:
            format: File format ('json' or 'csv')
            
        Returns:
            Path to saved file
        """
        if format == 'csv':
            return self.save_to_csv()
        else:
            return self.save_to_json()
    
    def reset(self):
        """Reset counter and clear collections"""
        # Save before reset
        if self.collections:
            self.save_to_file()
        
        self.collection_count = 0
        self.collections = []
        self.session_start = datetime.now()
    
    def export_summary(self) -> str:
        """
        Export a summary report
        
        Returns:
            Path to summary file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.log_directory / f"summary_{timestamp}.txt"
        
        stats = self.get_statistics()
        
        summary = f"""
Smart Marine Vessel - Collection Summary Report
================================================

Session Information:
-------------------
Start Time: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}
End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {stats['session_duration_minutes']:.2f} minutes

Collection Statistics:
---------------------
Total Plastics Collected: {stats['total_count']}
Collections per Hour: {stats['collections_per_hour']:.2f}
Average Detection Confidence: {stats['avg_confidence']:.3f}
Unique Collection Locations: {stats['locations_count']}

Recent Collections (Last 5):
---------------------------
"""
        
        for collection in self.get_recent_collections(5):
            summary += f"\n{collection['id']}. {collection['timestamp']}"
            if collection.get('gps_latitude'):
                summary += f" | GPS: ({collection['gps_latitude']:.6f}, {collection['gps_longitude']:.6f})"
            if collection.get('confidence'):
                summary += f" | Confidence: {collection['confidence']:.3f}"
        
        summary += f"\n\nFull data saved to: {self.save_to_json()}\n"
        
        with open(filepath, 'w') as f:
            f.write(summary)
        
        return str(filepath)

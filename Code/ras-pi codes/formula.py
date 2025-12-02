#!/usr/bin/env python3
"""
Safety Index Calculator for Ride Data
Usage: python3 formula.py data/ride01
       python3 formula.py data/ride01 --detailed

Calculates a safety index (0-10) based on:
- Acceleration patterns (harsh braking/acceleration)
- Gyroscope patterns (aggressive turns)
- Event-based speed analysis (speed changes before/after events)
- Pothole encounters
- Overall speed consistency
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
from supabase import create_client, Client

SUPABASE_URL = "https://ghtqafnlnijxvsmzdnmh.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdodHFhZm5sbmlqeHZzbXpkbm1oIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTk3NDQyNjcsImV4cCI6MjA3NTMyMDI2N30.Q1LGQP8JQdWn6rJJ1XRYT8rfo9b2Q5YfWUytrzQEsa0"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


class SafetyIndexCalculator:
    """Calculate safety index from ride data"""
    
    def __init__(self, ride_folder):
        self.ride_folder = ride_folder
        self.sensor_data = None
        self.events_data = None
        self.pothole_data = None
        self.metrics = {}
        
    def load_data(self):
        """Load all CSV files from ride folder"""
        print(f"📂 Loading data from: {self.ride_folder}")
        
        # Load sensor data
        sensor_path = os.path.join(self.ride_folder, "sensor_data.csv")
        if os.path.exists(sensor_path):
            self.sensor_data = pd.read_csv(sensor_path)
            print(f"✅ Loaded {len(self.sensor_data)} sensor samples")
        else:
            print(f"❌ sensor_data.csv not found")
            return False
        
        # Load events data
        events_path = os.path.join(self.ride_folder, "events.csv")
        if os.path.exists(events_path):
            self.events_data = pd.read_csv(events_path)
            print(f"✅ Loaded {len(self.events_data)} events")
        else:
            print(f"⚠️ events.csv not found")
            self.events_data = pd.DataFrame()
        
        # Load pothole data
        pothole_path = os.path.join(self.ride_folder, "pothole_log.csv")
        if os.path.exists(pothole_path):
            self.pothole_data = pd.read_csv(pothole_path)
            print(f"✅ Loaded {len(self.pothole_data)} potholes")
        else:
            print(f"⚠️ pothole_log.csv not found")
            self.pothole_data = pd.DataFrame()
        
        return True
    
    def calculate_acceleration_score(self):
        """
        Score based on acceleration patterns (0-10)
        Lower scores for harsh braking/acceleration
        """
        if self.sensor_data is None or len(self.sensor_data) == 0:
            return 10.0
        
        # Calculate magnitude of acceleration changes
        ax = self.sensor_data['ax_g'].values
        ay = self.sensor_data['ay_g'].values
        az = self.sensor_data['az_g'].values
        
        # Calculate total acceleration magnitude
        accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
        
        # Calculate changes (jerk)
        accel_changes = np.abs(np.diff(accel_mag))
        
        # Statistics
        mean_change = np.mean(accel_changes)
        max_change = np.max(accel_changes)
        harsh_events = np.sum(accel_changes > 0.6)  # Threshold for harsh event (relaxed from 0.5g)
        
        self.metrics['accel_mean_change'] = float(mean_change)
        self.metrics['accel_max_change'] = float(max_change)
        self.metrics['harsh_accel_events'] = int(harsh_events)
        
        # Scoring: penalize harsh acceleration changes (adjusted for sensor noise)
        score = 10.0
        score -= min(2.0, harsh_events * 0.02)  # Up to -2 points (lenient for noisy data)
        score -= min(1.5, max_change * 1.2)      # Up to -1.5 points
        score -= min(1.5, mean_change * 6.0)    # Up to -1.5 points
        
        return max(0.0, score)
    
    def calculate_gyroscope_score(self):
        """
        Score based on gyroscope patterns (0-10)
        Lower scores for aggressive turning
        """
        if self.sensor_data is None or len(self.sensor_data) == 0:
            return 10.0
        
        gx = self.sensor_data['gx_dps'].values
        gy = self.sensor_data['gy_dps'].values
        gz = self.sensor_data['gz_dps'].values
        
        # Calculate rotation magnitude
        rotation_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        
        # Statistics
        mean_rotation = np.mean(rotation_mag)
        max_rotation = np.max(rotation_mag)
        aggressive_turns = np.sum(rotation_mag > 60)  # Threshold for aggressive turn (relaxed from 50°/s)
        
        self.metrics['gyro_mean_rotation'] = float(mean_rotation)
        self.metrics['gyro_max_rotation'] = float(max_rotation)
        self.metrics['aggressive_turns'] = int(aggressive_turns)
        
        # Scoring: penalize aggressive turns (adjusted for sensor noise)
        score = 10.0
        score -= min(2.5, aggressive_turns * 0.015)  # Up to -2.5 points (lenient for noisy data)
        score -= min(1.5, max_rotation / 130.0)     # Up to -1.5 points
        score -= min(1.5, mean_rotation / 13.0)     # Up to -1.5 points
        
        return max(0.0, score)
    
    def calculate_event_smoothness_score(self):
        """
        Score based on event distribution and speed changes around events (0-10)
        Analyzes speed before/after events to detect unsafe behavior
        Lower scores for:
        - Frequent event changes (jerky driving)
        - Poor speed management around events (not slowing for turns, etc.)
        """
        if self.events_data is None or len(self.events_data) == 0:
            return 10.0
        
        # Count event transitions
        num_events = len(self.events_data)
        
        # Calculate average event duration
        if 'duration_seconds' in self.events_data.columns:
            avg_duration = self.events_data['duration_seconds'].mean()
            min_duration = self.events_data['duration_seconds'].min()
        else:
            avg_duration = 5.0
            min_duration = 1.0
        
        # Count event types
        event_counts = {}
        if 'event_type' in self.events_data.columns:
            event_counts = self.events_data['event_type'].value_counts().to_dict()
        
        self.metrics['num_events'] = int(num_events)
        self.metrics['avg_event_duration'] = float(avg_duration)
        self.metrics['min_event_duration'] = float(min_duration)
        self.metrics['event_distribution'] = event_counts
        
        # Analyze speed changes around events
        speed_violations = 0
        if self.sensor_data is not None and 'gps_speed_kn' in self.sensor_data.columns and 'start_time' in self.events_data.columns:
            speed_violations = self._analyze_event_speeds()
            self.metrics['event_speed_violations'] = int(speed_violations)
        
        # Scoring: penalize frequent event changes (jerky driving)
        score = 10.0
        
        # Penalize very short events (jerky driving)
        if min_duration < 0.5:
            score -= 2.0
        elif min_duration < 1.0:
            score -= 1.0
        
        # Penalize low average duration
        if avg_duration < 2.0:
            score -= 2.0
        elif avg_duration < 3.0:
            score -= 1.0
        
        # Penalize excessive event changes per minute
        total_duration = self.events_data['duration_seconds'].sum() if 'duration_seconds' in self.events_data.columns else 60
        events_per_minute = (num_events / total_duration) * 60
        
        if events_per_minute > 20:
            score -= 3.0
        elif events_per_minute > 10:
            score -= 1.5
        
        # Penalize poor speed management around events
        if speed_violations > 10:
            score -= 2.0
        elif speed_violations > 5:
            score -= 1.0
        
        self.metrics['events_per_minute'] = float(events_per_minute)
        
        return max(0.0, score)
    
    def _analyze_event_speeds(self):
        """
        Analyze speed changes before and after events
        Returns count of speed violations (unsafe speed management)
        """
        violations = 0
        
        # Convert timestamp columns to datetime if needed
        self.sensor_data['timestamp'] = pd.to_datetime(self.sensor_data['timestamp'], format='%Y-%m-%d_%H-%M-%S-%f', errors='coerce')
        
        for idx, event in self.events_data.iterrows():
            try:
                event_type = event.get('event_type', 'UNKNOWN')
                start_time = pd.to_datetime(event['start_time'])
                end_time = pd.to_datetime(event['end_time'])
                
                # Get speed data 2 seconds before and after event
                before_mask = (self.sensor_data['timestamp'] >= start_time - pd.Timedelta(seconds=2)) & \
                              (self.sensor_data['timestamp'] < start_time)
                after_mask = (self.sensor_data['timestamp'] > end_time) & \
                             (self.sensor_data['timestamp'] <= end_time + pd.Timedelta(seconds=2))
                
                speeds_before = pd.to_numeric(self.sensor_data.loc[before_mask, 'gps_speed_kn'], errors='coerce').dropna()
                speeds_after = pd.to_numeric(self.sensor_data.loc[after_mask, 'gps_speed_kn'], errors='coerce').dropna()
                
                if len(speeds_before) > 0 and len(speeds_after) > 0:
                    avg_speed_before = speeds_before.mean()
                    avg_speed_after = speeds_after.mean()
                    speed_change = abs(avg_speed_after - avg_speed_before)
                    
                    # Check for unsafe speed patterns
                    if event_type in ['LEFT', 'RIGHT']:
                        # Should slow down for turns
                        if avg_speed_before > 15 and speed_change < 2:
                            violations += 1  # Didn't slow for turn
                    elif event_type == 'STOP':
                        # Should decelerate smoothly
                        if speed_change > 10:
                            violations += 1  # Too sudden stop
                    elif event_type == 'STRAIGHT':
                        # Speed should be relatively stable
                        if speed_change > 8:
                            violations += 1  # Erratic speed on straight
            except Exception:
                # Skip events with parsing errors
                continue
        
        return violations
    
    def calculate_pothole_score(self):
        """
        Score based on pothole encounters (0-10)
        NOTE: This measures ROUTE hazard level, not rider skill
        Lower scores indicate more hazardous route (not rider's fault)
        Given low weight (10%) since rider can't control road conditions
        """
        if self.pothole_data is None or len(self.pothole_data) == 0:
            return 10.0
        
        num_potholes = len(self.pothole_data)
        
        # Calculate average confidence
        if 'confidence_percent' in self.pothole_data.columns:
            avg_confidence = self.pothole_data['confidence_percent'].mean()
        else:
            avg_confidence = 80.0
        
        self.metrics['num_potholes'] = int(num_potholes)
        self.metrics['avg_pothole_confidence'] = float(avg_confidence)
        
        # Scoring: indicates route hazard level (REDUCED penalty - not rider's fault)
        score = 10.0
        score -= min(3.0, num_potholes * 0.3)  # REDUCED: Up to -3 points (was -5)
        
        return max(0.0, score)
    
    def calculate_speed_consistency_score(self):
        """
        Score based on speed consistency (0-10)
        Lower scores for erratic speed changes
        """
        if self.sensor_data is None or len(self.sensor_data) == 0:
            return 10.0
        
        # Try to get speed from GPS
        if 'gps_speed_kn' in self.sensor_data.columns:
            speeds = pd.to_numeric(self.sensor_data['gps_speed_kn'], errors='coerce')
            speeds = speeds.dropna()
            
            if len(speeds) > 0:
                speed_changes = np.abs(np.diff(speeds))
                mean_speed = speeds.mean()
                std_speed = speeds.std()
                max_speed_change = speed_changes.max() if len(speed_changes) > 0 else 0
                
                self.metrics['mean_speed_kn'] = float(mean_speed)
                self.metrics['std_speed_kn'] = float(std_speed)
                self.metrics['max_speed_change_kn'] = float(max_speed_change)
                
                # Scoring: penalize erratic speed changes
                score = 10.0
                
                # Penalize high standard deviation
                if std_speed > 5.0:
                    score -= 2.0
                elif std_speed > 3.0:
                    score -= 1.0
                
                # Penalize sudden speed changes
                if max_speed_change > 10.0:
                    score -= 2.0
                elif max_speed_change > 5.0:
                    score -= 1.0
                
                return max(0.0, score)
        
        return 10.0
    
    def calculate_overall_safety_index(self):
        """
        Calculate weighted overall safety index (0-10)
        """
        # Calculate individual scores
        accel_score = self.calculate_acceleration_score()
        gyro_score = self.calculate_gyroscope_score()
        event_score = self.calculate_event_smoothness_score()
        pothole_score = self.calculate_pothole_score()
        speed_score = self.calculate_speed_consistency_score()
        
        # Store subscores
        self.metrics['subscores'] = {
            'acceleration': round(accel_score, 2),
            'gyroscope': round(gyro_score, 2),
            'event_smoothness': round(event_score, 2),
            'pothole_avoidance': round(pothole_score, 2),
            'speed_consistency': round(speed_score, 2)
        }
        
        # Weighted average (weights sum to 1.0) - IMPROVED for real-world riding
        weights = {
            'acceleration': 0.25,      # 25% - harsh braking/acceleration (MOST IMPORTANT)
            'gyroscope': 0.25,         # 25% - aggressive turns (MOST IMPORTANT)
            'event_smoothness': 0.20,  # 20% - smooth driving style
            'pothole_avoidance': 0.10, # 10% - route hazard indicator (LOW - not rider's fault)
            'speed_consistency': 0.20  # 20% - steady speed (IMPORTANT)
        }
        
        overall = (
            accel_score * weights['acceleration'] +
            gyro_score * weights['gyroscope'] +
            event_score * weights['event_smoothness'] +
            pothole_score * weights['pothole_avoidance'] +
            speed_score * weights['speed_consistency']
        )
        
        self.metrics['weights'] = weights
        self.metrics['overall_safety_index'] = round(overall, 2)
        
        return overall
    
    def get_safety_rating(self, index):
        """Convert numerical index to text rating"""
        if index >= 9.0:
            return "EXCELLENT"
        elif index >= 7.5:
            return "GOOD"
        elif index >= 6.0:
            return "FAIR"
        elif index >= 4.0:
            return "POOR"
        else:
            return "DANGEROUS"
    
    def print_report(self, detailed=False):
        """Print safety index report"""
        overall = self.metrics['overall_safety_index']
        rating = self.get_safety_rating(overall)
        
        print("\n" + "="*60)
        print("🛡️  RIDE SAFETY INDEX REPORT")
        print("="*60)
        print(f"📁 Ride: {os.path.basename(self.ride_folder)}")
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        print(f"\n🎯 OVERALL SAFETY INDEX: {overall:.2f} / 10.0")
        print(f"📊 Rating: {rating}")
        
        print("\n📋 Component Scores:")
        for component, score in self.metrics['subscores'].items():
            bar_length = int(score)
            bar = "█" * bar_length + "░" * (10 - bar_length)
            print(f"  • {component.replace('_', ' ').title():.<30} {score:>5.2f} {bar}")
        
        if detailed:
            print("\n📈 Detailed Metrics:")
            print(f"  Acceleration:")
            print(f"    - Mean change: {self.metrics.get('accel_mean_change', 0):.4f} g")
            print(f"    - Max change: {self.metrics.get('accel_max_change', 0):.4f} g")
            print(f"    - Harsh events: {self.metrics.get('harsh_accel_events', 0)}")
            
            print(f"\n  Gyroscope:")
            print(f"    - Mean rotation: {self.metrics.get('gyro_mean_rotation', 0):.2f} °/s")
            print(f"    - Max rotation: {self.metrics.get('gyro_max_rotation', 0):.2f} °/s")
            print(f"    - Aggressive turns: {self.metrics.get('aggressive_turns', 0)}")
            
            print(f"\n  Events & Speed Management:")
            print(f"    - Total events: {self.metrics.get('num_events', 0)}")
            print(f"    - Avg duration: {self.metrics.get('avg_event_duration', 0):.2f} s")
            print(f"    - Events/minute: {self.metrics.get('events_per_minute', 0):.2f}")
            print(f"    - Speed violations (before/after events): {self.metrics.get('event_speed_violations', 0)}")
            if 'event_distribution' in self.metrics:
                print(f"    - Event distribution:")
                for event_type, count in self.metrics['event_distribution'].items():
                    print(f"      • {event_type}: {count}")
            
            print(f"\n  Potholes:")
            print(f"    - Total potholes: {self.metrics.get('num_potholes', 0)}")
            print(f"    - Avg confidence: {self.metrics.get('avg_pothole_confidence', 0):.1f}%")
            
            if 'mean_speed_kn' in self.metrics:
                print(f"\n  Speed:")
                print(f"    - Mean speed: {self.metrics.get('mean_speed_kn', 0):.2f} knots")
                print(f"    - Std deviation: {self.metrics.get('std_speed_kn', 0):.2f} knots")
                print(f"    - Max change: {self.metrics.get('max_speed_change_kn', 0):.2f} knots")
        
        print("\n💡 Recommendations:")
        if overall >= 9.0:
            print("  ✅ Excellent ride! Keep up the safe driving.")
        elif overall >= 7.5:
            print("  👍 Good ride with minor areas for improvement.")
        elif overall >= 6.0:
            print("  ⚠️ Fair ride. Consider smoother acceleration and turning.")
        elif overall >= 4.0:
            print("  ⚠️ Poor ride. Significant safety concerns detected.")
            print("  💡 Focus on: smoother braking, gentler turns, avoiding hazards.")
        else:
            print("  🚨 Dangerous ride! Immediate improvement needed.")
            print("  💡 Recommendations: defensive driving course, route planning.")
        
        print("\n" + "="*60)
    
    def save_report(self):
        """Save report as JSON file"""
        report_path = os.path.join(self.ride_folder, "safety_report.json")
        
        with open(report_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"💾 Safety report saved: {report_path}")
    
    def save_index_csv(self):
        """Save simple index.csv with single safety score (0-10)"""
        index_path = os.path.join(self.ride_folder, "index.csv")
        
        # Get overall safety index (already calculated)
        overall = self.metrics.get('overall_safety_index', 0.0)
        
        # Write single value to CSV
        with open(index_path, 'w') as f:
            f.write(f"{overall:.2f}\n")
        
        print(f"📊 Safety index saved: {index_path} (Score: {overall:.2f}/10)")
        return overall
def upload_safety_index(rider_id, file_id, metrics):
    """Upload safety index + subscores to Supabase ride_safety_index table"""

    data = {
        "rider_id": rider_id,
        "file_id": file_id,
        "overall_score": metrics["overall_safety_index"],
        "acceleration_score": metrics["subscores"]["acceleration"],
        "gyroscope_score": metrics["subscores"]["gyroscope"],
        "event_smoothness_score": metrics["subscores"]["event_smoothness"],
        "pothole_score": metrics["subscores"]["pothole_avoidance"],
        "speed_consistency_score": metrics["subscores"]["speed_consistency"],
    }

    # Remove any old entry (avoid duplicates)
    supabase.table("ride_safety_index") \
        .delete() \
        .eq("rider_id", rider_id) \
        .eq("file_id", file_id) \
        .execute()

    # Insert the new one
    supabase.table("ride_safety_index") \
        .insert(data) \
        .execute()

    print("? Safety Index uploaded to Supabase successfully!")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python3 safety_index.py <ride_folder> [--detailed]")
        print("Example: python3 safety_index.py data/ride01")
        print("         python3 safety_index.py data/ride01 --detailed")
        sys.exit(1)
    
    ride_folder = sys.argv[1]
    detailed = "--detailed" in sys.argv or "-d" in sys.argv
    
    if not os.path.exists(ride_folder):
        print(f"❌ Error: Folder not found: {ride_folder}")
        sys.exit(1)
    
    # Calculate safety index
    calculator = SafetyIndexCalculator(ride_folder)
    
    if not calculator.load_data():
        print("❌ Error: Failed to load ride data")
        sys.exit(1)
    
    # Calculate and display
    overall = calculator.calculate_overall_safety_index()
    calculator.print_report(detailed=detailed)
    calculator.save_report()
    calculator.save_index_csv()  # Save simple index.csv
    # Rider & file IDs from environment variables
    rider_id = os.environ.get("RIDER_ID")
    file_id = os.environ.get("FILE_ID")

    if not rider_id or not file_id:
            print("? rider_id or file_id missing � cannot upload safety index to Supabase")        
    else:
            upload_safety_index(rider_id, file_id, calculator.metrics)

    
    return 0


if __name__ == "__main__":
    sys.exit(main())

import os
import sys
import time
import pandas as pd
import numpy as np

class DemoSerialReader:
    """A mock SerialStreamReader that yields packets from dummycsv.csv."""
    def __init__(self, parent=None):
        self.running = False
        self.data_count = 0
        self.df = None
        self.row_index = 0
        self.last_read_time = time.time()
        # The demo data is sampled at ~80Hz (1/80 = 0.0125s per packet)
        self.base_delay = 1.0 / 80.0

        # Resolve dummycsv.csv
        ecg_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(ecg_dir, '..', '..'))
        if getattr(sys, 'frozen', False):
            bundle_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
            candidates = [
                os.path.join(bundle_dir, 'dummycsv.csv'),
                os.path.join(bundle_dir, '_internal', 'dummycsv.csv'),
                os.path.join(os.path.dirname(sys.executable), 'dummycsv.csv'),
                os.path.join(ecg_dir, 'dummycsv.csv'),
                os.path.join(project_root, 'dummycsv.csv'),
            ]
        else:
            candidates = [
                os.path.join(ecg_dir, 'dummycsv.csv'),
                os.path.join(os.path.dirname(ecg_dir), 'dummycsv.csv'),
                os.path.join(project_root, 'dummycsv.csv'),
                os.path.abspath('dummycsv.csv')
            ]
            
        csv_path = None
        for p in candidates:
            if os.path.exists(p):
                csv_path = p
                break
                
        if csv_path:
            try:
                self.df = pd.read_csv(csv_path)
                if len(self.df.columns) == 1:
                    self.df = pd.read_csv(csv_path, sep='\t')
                print(f" DemoSerialReader initialized with {len(self.df)} rows.")
            except Exception as e:
                print(f" DemoSerialReader error reading CSV: {e}")
                self.df = None
        else:
            print(" DemoSerialReader: dummycsv.csv not found!")

    def start(self, *args, **kwargs):
        print(" DemoSerialReader: Started.")
        self.running = True
        self.last_read_time = time.time()
        self.row_index = 0

    def stop(self):
        print(" DemoSerialReader: Stopped.")
        self.running = False

    def close(self):
        self.running = False

    def read_packets(self, max_packets=100):
        if not self.running or self.df is None or len(self.df) == 0:
            return []
            
        current_time = time.time()
        elapsed = current_time - self.last_read_time
        packets_to_read = int(elapsed / self.base_delay)
        
        if packets_to_read == 0:
            return []
            
        # Update the time we last read data so we continually emit at base_delay speed
        self.last_read_time = current_time
        packets_to_read = min(packets_to_read, max_packets)
        
        out = []
        for _ in range(packets_to_read):
            if self.row_index >= len(self.df):
                self.row_index = 0
                
            row = self.df.iloc[self.row_index]
            packet = {}
            for col in self.df.columns:
                if col != 'Sample':
                    try:
                        packet[col] = float(row[col])
                    except:
                        packet[col] = 0.0
            
            # Add synthetic derived leads (III, aVR, aVL, aVF) if missing
            if 'I' in packet and 'II' in packet:
                I = packet['I']
                II = packet['II']
                if 'III' not in packet: packet['III'] = II - I
                if 'aVR' not in packet: packet['aVR'] = -(I + II) / 2
                if 'aVL' not in packet: packet['aVL'] = I - II / 2
                if 'aVF' not in packet: packet['aVF'] = II - I / 2
                
            out.append(packet)
            self.row_index += 1
            self.data_count += 1
            
        return out

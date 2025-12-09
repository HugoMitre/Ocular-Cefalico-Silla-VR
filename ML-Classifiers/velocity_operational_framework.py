#!/usr/bin/env python3
"""
Head Velocity Classification System
Operational Framework for Real-Time and Batch Processing

This system provides comprehensive velocity pattern classification for wheelchair
navigation control. Built on Proximity Forest 2.0 with MSM distance metrics, it
supports real-time TCP communication with Unity, batch CSV processing, and
synthetic data validation.

The framework enables both controlled testing environments and production deployment,
facilitating seamless integration with Unity-based wheelchair interfaces and
offline data analysis pipelines.

Key capabilities:
- Real-time TCP classification servers (ports 5556/5558)
- Parallel batch processing with optimized caching
- Synthetic data generation and validation
- Multi-mode classification (fast/precise)
- Comprehensive performance metrics

Author: Manuel
Institution: CIMAT - Centro de Investigación en Matemáticas
Date: December 2025
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys
import time
import json
import socket
import struct
import threading
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import warnings

warnings.filterwarnings('ignore')

try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent))

try:
    from velocity_classifier import InferencePipeline, SyntheticVelocityGenerator
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False
    print("Warning: Velocity classifier module not available")


# ============================================================================
# CONSTANTS
# ============================================================================

VELOCITY_CATEGORIES = [
    'MUY_BAJA',
    'BAJA',
    'MEDIA_BAJA',
    'MEDIA',
    'MEDIA_ALTA',
    'ALTA',
    'MUY_ALTA'
]

VELOCITY_RANGES = {
    'MUY_BAJA': {'min': 2.0, 'max': 12.0},
    'BAJA': {'min': 12.0, 'max': 24.0},
    'MEDIA_BAJA': {'min': 24.0, 'max': 36.0},
    'MEDIA': {'min': 36.0, 'max': 48.0},
    'MEDIA_ALTA': {'min': 48.0, 'max': 60.0},
    'ALTA': {'min': 60.0, 'max': 72.0},
    'MUY_ALTA': {'min': 72.0, 'max': 90.0}
}

CONFIG_FILE = Path("velocity_config.json")


# ============================================================================
# CONFIGURATION
# ============================================================================

class SystemConfiguration:
    """System configuration parameters"""
    
    def __init__(self):
        self.tcp_host = '0.0.0.0'
        self.tcp_port_synthetic = 5556
        self.tcp_port_real = 5558
        self.tcp_timeout = 30.0
        self.n_workers = max(1, cpu_count() - 1)


# ============================================================================
# MODEL MANAGER
# ============================================================================

class ModelManager:
    """
    Centralized model management
    
    Handles loading, validation, and access to trained Proximity Forest models
    and associated metadata.
    """
    
    def __init__(self, model_directory: Optional[str] = None):
        if model_directory:
            self.directory = Path(model_directory)
        else:
            self.directory = self._get_configured_directory()
        
        if self.directory:
            self.directory.mkdir(exist_ok=True)
        
        self.models: Dict[str, Any] = {}
    
    def _get_configured_directory(self) -> Optional[Path]:
        """Retrieve previously configured model directory"""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    directory = Path(config.get('model_directory', ''))
                    if directory.exists():
                        return directory
            except:
                pass
        
        return self._select_directory()
    
    def _select_directory(self) -> Optional[Path]:
        """Interactive directory selection"""
        print("\n" + "="*80)
        print("MODEL DIRECTORY CONFIGURATION")
        print("="*80)
        print("\nSelect directory containing trained model files\n")
        
        directory_path = None
        
        if TKINTER_AVAILABLE:
            try:
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                
                directory = filedialog.askdirectory(
                    title="Select Model Directory",
                    mustexist=True
                )
                
                root.destroy()
                
                if directory:
                    directory_path = Path(directory)
            except:
                pass
        
        if not directory_path:
            directory = input("Model directory path: ").strip()
            if directory:
                directory_path = Path(directory)
        
        if not directory_path or not directory_path.exists():
            return None
        
        config = {'model_directory': str(directory_path.absolute())}
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nDirectory configured: {directory_path.absolute()}\n")
        return directory_path
    
    def change_directory(self) -> bool:
        """Change model directory"""
        new_directory = self._select_directory()
        
        if new_directory:
            self.directory = new_directory
            self.models.clear()
            return True
        
        return False
    
    def load_all(self) -> Dict[str, Any]:
        """Load all model files from directory"""
        if not self.directory:
            return {}
        
        print(f"\n{'='*80}")
        print(f"LOADING MODELS")
        print(f"{'='*80}\n")
        
        pkl_files = list(self.directory.glob("*.pkl"))
        
        if not pkl_files:
            print(f"No PKL files found")
            return {}
        
        for file_path in pkl_files:
            name = file_path.stem
            try:
                model = self._load_pickle(str(file_path))
                if model is not None:
                    self.models[name] = model
                    print(f"Loaded: {name}")
            except Exception as e:
                print(f"Error loading {name}: {str(e)}")
        
        print(f"\n{'='*80}")
        print(f"Total loaded: {len(self.models)} model(s)")
        print(f"{'='*80}\n")
        
        return self.models
    
    def _load_pickle(self, file_path: str) -> Optional[Any]:
        """Load pickle file with compatibility handling"""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except:
            pass
        
        try:
            class CompatibilityUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if 'numpy._core' in module:
                        module = module.replace('numpy._core', 'numpy.core')
                    return super().find_class(module, name)
            
            with open(file_path, 'rb') as f:
                return CompatibilityUnpickler(f).load()
        except:
            return None
    
    def get(self, name: str) -> Optional[Any]:
        """Retrieve model by name"""
        return self.models.get(name)


# ============================================================================
# FILE SELECTORS
# ============================================================================

def select_csv_file() -> Optional[str]:
    """Interactive CSV file selection"""
    if TKINTER_AVAILABLE:
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            file_path = filedialog.askopenfilename(
                title="Select CSV File",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
            )
            
            root.destroy()
            
            if file_path and os.path.exists(file_path):
                return file_path
        except:
            pass
    
    file_path = input("\nCSV file path: ").strip()
    if file_path and os.path.exists(file_path):
        return file_path
    return None


def select_directory() -> Optional[str]:
    """Interactive directory selection"""
    if TKINTER_AVAILABLE:
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            directory = filedialog.askdirectory(
                title="Select Directory"
            )
            
            root.destroy()
            
            if directory and os.path.exists(directory):
                return directory
        except:
            pass
    
    directory = input("\nDirectory path: ").strip()
    if directory and os.path.exists(directory):
        return directory
    return None


# ============================================================================
# TCP SERVER - SYNTHETIC DATA
# ============================================================================

class SyntheticDataServer:
    """
    TCP server for synthetic data classification
    
    Accepts connections for testing with generated synthetic velocity sequences,
    providing ground truth validation capabilities.
    """
    
    def __init__(self, system, host: str = '0.0.0.0', port: int = 5556):
        self.system = system
        self.host = host
        self.port = port
        self.server_socket = None
        self.active = False
        self.server_thread = None
        
        self.total_received = 0
        self.total_classified = 0
        self.total_errors = 0
        self.avg_time_ms = 0.0
        self.stats_lock = threading.Lock()
    
    def start(self):
        """Start TCP server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.active = True
            
            print(f"\n{'='*80}")
            print(f"TCP SERVER - SYNTHETIC DATA MODE")
            print(f"{'='*80}")
            print(f"Listening on: {self.host}:{self.port}")
            print(f"Waiting for connections...")
            print(f"Press Ctrl+C to stop\n")
            
            self.server_thread = threading.Thread(target=self._accept_connections, daemon=True)
            self.server_thread.start()
            
            try:
                while self.active:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n\nShutting down server...")
                self.stop()
        
        except Exception as e:
            print(f"Server error: {e}")
            self.stop()
    
    def _accept_connections(self):
        """Accept incoming connections"""
        while self.active:
            try:
                client_socket, address = self.server_socket.accept()
                
                try:
                    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                    client_socket.settimeout(30.0)
                except:
                    pass
                
                print(f"\nConnection established: {address[0]}:{address[1]}")
                
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
                
            except Exception as e:
                if self.active:
                    print(f"Connection error: {e}")
    
    def _handle_client(self, client_socket: socket.socket, address: Tuple):
        """Handle individual client connection"""
        print(f"[{address[0]}] Client connected")
        
        try:
            client_socket.settimeout(30.0)
        except:
            pass
        
        try:
            while self.active:
                try:
                    length_bytes = self._receive_exact(client_socket, 4)
                    if not length_bytes:
                        break
                    
                    length = struct.unpack('I', length_bytes)[0]
                    
                    if length <= 0 or length > 1048576:
                        print(f"[{address[0]}] Invalid message length: {length}")
                        break
                    
                    data_bytes = self._receive_exact(client_socket, length)
                    if not data_bytes:
                        break
                    
                    json_str = data_bytes.decode('utf-8')
                    
                    self._process_and_respond(client_socket, json_str, address)
                    
                except socket.timeout:
                    print(f"[{address[0]}] Timeout, maintaining connection...")
                    continue
                except Exception as e:
                    print(f"[{address[0]}] Error: {e}")
                    break
                    
        except Exception as e:
            print(f"[{address[0]}] Error: {e}")
        finally:
            print(f"[{address[0]}] Client disconnected")
            try:
                client_socket.close()
            except:
                pass
    
    def _receive_exact(self, sock: socket.socket, n_bytes: int) -> Optional[bytes]:
        """Receive exact number of bytes"""
        data = b''
        attempts = 0
        max_attempts = 50
        
        while len(data) < n_bytes and attempts < max_attempts:
            try:
                remaining = n_bytes - len(data)
                packet = sock.recv(min(remaining, 4096))
                
                if not packet:
                    return None
                
                data += packet
                attempts = 0
                
            except socket.timeout:
                attempts += 1
                if attempts >= max_attempts:
                    return None
                time.sleep(0.01)
                continue
            except:
                return None
                
        return data if len(data) == n_bytes else None
    
    def _process_and_respond(self, client_socket: socket.socket, json_str: str, 
                            address: Tuple):
        """Process received data and send classification result"""
        try:
            data = json.loads(json_str)
            
            vel_x = np.array(data.get('HeadVelocityX', []), dtype=np.float32)
            vel_y = np.array(data.get('HeadVelocityY', []), dtype=np.float32)
            ground_truth = data.get('ground_truth', 'UNKNOWN')
            
            with self.stats_lock:
                self.total_received += 1
                sequence_num = self.total_received
            
            magnitude = np.mean(np.sqrt(vel_x**2 + vel_y**2))
            
            print(f"\n{'='*80}")
            print(f"[{address[0]}] SYNTHETIC SEQUENCE #{sequence_num}")
            print(f"{'='*80}")
            print(f"Ground truth: {ground_truth}")
            print(f"Sequence length: {len(vel_x)} samples")
            print(f"Average magnitude: {magnitude:.2f}")
            print(f"VelX range: [{np.min(vel_x):.2f}, {np.max(vel_x):.2f}]")
            print(f"VelY range: [{np.min(vel_y):.2f}, {np.max(vel_y):.2f}]")
            print(f"\nClassifying...")
            
            start_time = time.time()
            
            prediction = None
            confidence = 0.0
            
            if self.system.classifier and len(vel_x) > 0:
                try:
                    prediction, confidence = self.system.classifier.predict(vel_x, vel_y)
                except Exception as e:
                    print(f"Classification error: {e}")
                    prediction = "ERROR"
                    confidence = 0.0
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            with self.stats_lock:
                self.total_classified += 1
                self.avg_time_ms = ((self.avg_time_ms * (self.total_classified - 1)) + elapsed_ms) / self.total_classified
            
            is_correct = prediction == ground_truth
            status = "CORRECT" if is_correct else "INCORRECT"
            
            print(f"\nRESULT:")
            print(f"  Prediction: {prediction}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Time: {elapsed_ms:.1f}ms")
            print(f"  Status: {status}")
            print(f"{'='*80}\n")
            
            response = {
                'prediction': prediction or 'UNKNOWN',
                'confidence': float(confidence),
                'time_ms': float(elapsed_ms)
            }
            
            response_json = json.dumps(response)
            response_bytes = response_json.encode('utf-8')
            
            length_bytes = struct.pack('I', len(response_bytes))
            client_socket.sendall(length_bytes)
            client_socket.sendall(response_bytes)
            
        except Exception as e:
            with self.stats_lock:
                self.total_errors += 1
            print(f"[{address[0]}] Processing error: {e}")
    
    def stop(self):
        """Stop server and print statistics"""
        self.active = False
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        print(f"\n{'='*80}")
        print(f"SERVER STATISTICS")
        print(f"{'='*80}")
        print(f"Total received: {self.total_received}")
        print(f"Total classified: {self.total_classified}")
        print(f"Total errors: {self.total_errors}")
        print(f"Average time: {self.avg_time_ms:.1f}ms")
        print(f"{'='*80}\n")


# ============================================================================
# TCP SERVER - REAL DATA
# ============================================================================

class RealDataServer:
    """
    TCP server for real data classification
    
    Accepts connections from Unity or other production systems for
    real-time wheelchair velocity classification.
    """
    
    def __init__(self, system, host: str = '0.0.0.0', port: int = 5558):
        self.system = system
        self.host = host
        self.port = port
        self.server_socket = None
        self.active = False
        self.server_thread = None
        
        self.total_received = 0
        self.total_classified = 0
        self.total_errors = 0
        self.avg_time_ms = 0.0
        self.stats_lock = threading.Lock()
        
        self.category_distribution = defaultdict(int)
    
    def start(self):
        """Start TCP server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.active = True
            
            print(f"\n{'='*80}")
            print(f"TCP SERVER - REAL DATA MODE")
            print(f"{'='*80}")
            print(f"Listening on: {self.host}:{self.port}")
            print(f"Waiting for Unity connections...")
            print(f"Press Ctrl+C to stop\n")
            
            self.server_thread = threading.Thread(target=self._accept_connections, daemon=True)
            self.server_thread.start()
            
            try:
                while self.active:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n\nShutting down server...")
                self.stop()
        
        except Exception as e:
            print(f"Server error: {e}")
            self.stop()
    
    def _accept_connections(self):
        """Accept incoming connections"""
        while self.active:
            try:
                client_socket, address = self.server_socket.accept()
                
                try:
                    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                    client_socket.settimeout(30.0)
                except:
                    pass
                
                print(f"\nConnection established: {address[0]}:{address[1]}")
                
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
                
            except Exception as e:
                if self.active:
                    print(f"Connection error: {e}")
    
    def _handle_client(self, client_socket: socket.socket, address: Tuple):
        """Handle individual client connection"""
        print(f"[{address[0]}] Client connected")
        
        try:
            client_socket.settimeout(30.0)
        except:
            pass
        
        try:
            while self.active:
                try:
                    length_bytes = self._receive_exact(client_socket, 4)
                    if not length_bytes:
                        break
                    
                    length = struct.unpack('I', length_bytes)[0]
                    
                    if length <= 0 or length > 1048576:
                        print(f"[{address[0]}] Invalid message length: {length}")
                        break
                    
                    data_bytes = self._receive_exact(client_socket, length)
                    if not data_bytes:
                        break
                    
                    json_str = data_bytes.decode('utf-8')
                    
                    self._process_and_respond(client_socket, json_str, address)
                    
                except socket.timeout:
                    print(f"[{address[0]}] Timeout, maintaining connection...")
                    continue
                except Exception as e:
                    print(f"[{address[0]}] Error: {e}")
                    break
                    
        except Exception as e:
            print(f"[{address[0]}] Error: {e}")
        finally:
            print(f"[{address[0]}] Client disconnected")
            try:
                client_socket.close()
            except:
                pass
    
    def _receive_exact(self, sock: socket.socket, n_bytes: int) -> Optional[bytes]:
        """Receive exact number of bytes"""
        data = b''
        attempts = 0
        max_attempts = 50
        
        while len(data) < n_bytes and attempts < max_attempts:
            try:
                remaining = n_bytes - len(data)
                packet = sock.recv(min(remaining, 4096))
                
                if not packet:
                    return None
                
                data += packet
                attempts = 0
                
            except socket.timeout:
                attempts += 1
                if attempts >= max_attempts:
                    return None
                time.sleep(0.01)
                continue
            except:
                return None
                
        return data if len(data) == n_bytes else None
    
    def _process_and_respond(self, client_socket: socket.socket, json_str: str, 
                            address: Tuple):
        """Process received data and send classification result"""
        try:
            data = json.loads(json_str)
            
            vel_x = np.array(data.get('HeadVelocityX', []), dtype=np.float32)
            vel_y = np.array(data.get('HeadVelocityY', []), dtype=np.float32)
            
            with self.stats_lock:
                self.total_received += 1
                sequence_num = self.total_received
            
            magnitude = np.mean(np.sqrt(vel_x**2 + vel_y**2))
            
            print(f"\n{'='*80}")
            print(f"[{address[0]}] REAL DATA SEQUENCE #{sequence_num}")
            print(f"{'='*80}")
            print(f"Sequence length: {len(vel_x)} samples")
            print(f"Average magnitude: {magnitude:.2f}")
            print(f"VelX range: [{np.min(vel_x):.2f}, {np.max(vel_x):.2f}]")
            print(f"VelY range: [{np.min(vel_y):.2f}, {np.max(vel_y):.2f}]")
            print(f"\nClassifying...")
            
            start_time = time.time()
            
            prediction = None
            confidence = 0.0
            
            if self.system.classifier and len(vel_x) > 0:
                try:
                    prediction, confidence = self.system.classifier.predict(vel_x, vel_y)
                except Exception as e:
                    print(f"Classification error: {e}")
                    prediction = "ERROR"
                    confidence = 0.0
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            with self.stats_lock:
                self.total_classified += 1
                self.avg_time_ms = ((self.avg_time_ms * (self.total_classified - 1)) + elapsed_ms) / self.total_classified
                if prediction:
                    self.category_distribution[prediction] += 1
            
            print(f"\nRESULT:")
            print(f"  Prediction: {prediction}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Time: {elapsed_ms:.1f}ms")
            print(f"  Total classified: {self.total_classified}")
            print(f"{'='*80}\n")
            
            response = {
                'prediction': prediction or 'UNKNOWN',
                'confidence': float(confidence),
                'time_ms': float(elapsed_ms)
            }
            
            response_json = json.dumps(response)
            response_bytes = response_json.encode('utf-8')
            
            length_bytes = struct.pack('I', len(response_bytes))
            client_socket.sendall(length_bytes)
            client_socket.sendall(response_bytes)
            
        except Exception as e:
            with self.stats_lock:
                self.total_errors += 1
            print(f"[{address[0]}] Processing error: {e}")
    
    def stop(self):
        """Stop server and print statistics"""
        self.active = False
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        print(f"\n{'='*80}")
        print(f"SERVER STATISTICS - REAL DATA")
        print(f"{'='*80}")
        print(f"Total received: {self.total_received}")
        print(f"Total classified: {self.total_classified}")
        print(f"Total errors: {self.total_errors}")
        print(f"Average time: {self.avg_time_ms:.1f}ms")
        
        if self.category_distribution:
            print(f"\nVELOCITY DISTRIBUTION:")
            for category, count in sorted(self.category_distribution.items()):
                percentage = (count / self.total_classified * 100) if self.total_classified > 0 else 0
                print(f"  {category:15} {count:4} ({percentage:5.1f}%)")
        
        print(f"{'='*80}\n")


# ============================================================================
# CLASSIFICATION SYSTEM
# ============================================================================

class VelocityClassificationSystem:
    """
    Main classification system orchestrator
    
    Manages model loading, classification operations, and system coordination.
    """
    
    def __init__(self, model_manager: ModelManager, config: SystemConfiguration):
        self.manager = model_manager
        self.config = config
        
        velocity_model = model_manager.get('velocity_model')
        
        self.classifier = None
        if velocity_model:
            try:
                model_dir = model_manager.directory
                self.classifier = InferencePipeline(str(model_dir))
            except Exception as e:
                print(f"Error loading classifier: {e}")
        
        self._display_system_info()
    
    def _display_system_info(self):
        """Display system configuration"""
        print(f"\n{'='*80}")
        print(f"VELOCITY CLASSIFICATION SYSTEM")
        print(f"{'='*80}\n")
        
        print(f"Components:")
        if self.classifier:
            print(f"  Classifier: Proximity Forest 2.0")
        else:
            print(f"  Classifier: Not loaded")
        
        print(f"\nConfiguration:")
        print(f"  Workers: {self.config.n_workers}")
        print(f"  TCP Ports: {self.config.tcp_port_synthetic} (synthetic), {self.config.tcp_port_real} (real)")
        
        print(f"\n{'='*80}\n")
    
    def classify_csv(self, csv_path: str, show_details: bool = True) -> Optional[Dict]:
        """Classify data from CSV file"""
        if show_details:
            print(f"\n{'='*80}")
            print(f"CLASSIFYING: {os.path.basename(csv_path)}")
            print(f"{'='*80}")
        
        try:
            df = pd.read_csv(csv_path)
            
            if 'HeadVelocityX' not in df.columns or 'HeadVelocityY' not in df.columns:
                if show_details:
                    print("Error: Required columns not found")
                return None
            
            vel_x = df['HeadVelocityX'].values
            vel_y = df['HeadVelocityY'].values
            
            start_time = time.time()
            
            if self.classifier:
                prediction, confidence = self.classifier.predict(vel_x, vel_y)
            else:
                prediction = "NO_CLASSIFIER"
                confidence = 0.0
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            if show_details:
                print(f"\nResult:")
                print(f"  Category: {prediction}")
                print(f"  Confidence: {confidence:.2%}")
                print(f"  Time: {elapsed_ms:.1f}ms")
                print(f"\n{'='*80}\n")
            
            return {
                'file': os.path.basename(csv_path),
                'prediction': prediction,
                'confidence': confidence,
                'time_ms': elapsed_ms
            }
            
        except Exception as e:
            if show_details:
                print(f"Error: {e}\n")
            return None
    
    def classify_directory_parallel(self, directory: str):
        """Classify all CSV files in directory using parallel processing"""
        csv_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            print(f"No CSV files found in: {directory}")
            return
        
        print(f"\n{'='*80}")
        print(f"PARALLEL BATCH PROCESSING")
        print(f"{'='*80}")
        print(f"Directory: {os.path.basename(directory)}")
        print(f"Files: {len(csv_files)}")
        print(f"Workers: {self.config.n_workers}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
            futures = {executor.submit(self.classify_csv, csv, False): csv for csv in csv_files}
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                try:
                    result = future.result(timeout=120)
                    if result:
                        results.append(result)
                        print(f"[{completed}/{len(csv_files)}] {result['file'][:40]:40} -> {result['prediction']}")
                except:
                    print(f"[{completed}/{len(csv_files)}] Error")
        
        total_time = time.time() - start_time
        
        self._display_batch_report(results, total_time, len(csv_files))
    
    def _display_batch_report(self, results: List[Dict], total_time: float, total_files: int):
        """Display batch processing report"""
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING REPORT")
        print(f"{'='*80}\n")
        
        print(f"Processed: {len(results)}/{total_files}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Throughput: {len(results)/total_time:.1f} files/s")
        print(f"Average per file: {(total_time*1000)/len(results):.1f}ms")
        
        print(f"\nCategory Distribution:")
        distribution = Counter([r['prediction'] for r in results])
        for category, count in distribution.most_common():
            percentage = (count / len(results)) * 100
            print(f"  {category:15} {count:4} ({percentage:5.1f}%)")
        
        print(f"\n{'='*80}\n")
    
    def validate_synthetic_data(self, category: str, n_sequences: int):
        """Validate classifier using synthetic data"""
        print(f"\n{'='*80}")
        print(f"SYNTHETIC DATA VALIDATION")
        print(f"{'='*80}")
        print(f"Category: {category}")
        print(f"Sequences: {n_sequences}")
        print(f"{'='*80}\n")
        
        if not CLASSIFIER_AVAILABLE:
            print("Error: Synthetic data generator not available")
            return
        
        generator = SyntheticVelocityGenerator(sequence_length=50)
        results = []
        
        for i in range(1, n_sequences + 1):
            try:
                sequences = generator.generate_category_dataset(category, 1)
                if not sequences:
                    continue
                
                vel_x, vel_y = sequences[0]
                
                if self.classifier:
                    prediction, confidence = self.classifier.predict(vel_x, vel_y)
                else:
                    prediction = "NO_CLASSIFIER"
                    confidence = 0.0
                
                is_correct = prediction == category
                results.append(is_correct)
                
                status = "CORRECT" if is_correct else "INCORRECT"
                print(f"[{i:3d}/{n_sequences}] Ground truth: {category:15} | Prediction: {prediction:15} | {status}")
                
            except Exception as e:
                print(f"[{i:3d}/{n_sequences}] Error: {e}")
        
        if results:
            accuracy = (sum(results) / len(results)) * 100
            print(f"\n{'='*80}")
            print(f"Accuracy: {sum(results)}/{len(results)} ({accuracy:.1f}%)")
            print(f"{'='*80}\n")
    
    def start_synthetic_server(self):
        """Start TCP server for synthetic data"""
        print(f"\n{'='*80}")
        print(f"TCP SERVER CONFIGURATION - SYNTHETIC MODE")
        print(f"{'='*80}")
        
        host = input("Host (default=0.0.0.0): ").strip() or '0.0.0.0'
        port_str = input("Port (default=5556): ").strip()
        port = int(port_str) if port_str else 5556
        
        server = SyntheticDataServer(self, host=host, port=port)
        server.start()
    
    def start_real_server(self):
        """Start TCP server for real data"""
        print(f"\n{'='*80}")
        print(f"TCP SERVER CONFIGURATION - REAL DATA MODE")
        print(f"{'='*80}")
        
        host = input("Host (default=0.0.0.0): ").strip() or '0.0.0.0'
        port_str = input("Port (default=5558): ").strip()
        port = int(port_str) if port_str else 5558
        
        server = RealDataServer(self, host=host, port=port)
        server.start()


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def main():
    """Main program interface"""
    print("\n" + "="*80)
    print(" HEAD VELOCITY CLASSIFICATION SYSTEM ".center(80))
    print("="*80)
    
    model_manager = ModelManager()
    
    if not model_manager.directory:
        print("\nNo model directory configured. Exiting...")
        return
    
    model_manager.load_all()
    
    if not model_manager.get('velocity_model'):
        print("\nNo valid models found. Exiting...")
        return
    
    config = SystemConfiguration()
    system = VelocityClassificationSystem(model_manager, config)
    
    while True:
        print("\n" + "="*80)
        print(" MAIN MENU ".center(80))
        print("="*80)
        print("1. Classify Single CSV File")
        print("2. Batch Process Directory")
        print("3. Validate with Synthetic Data")
        print("4. Configure Workers")
        print("5. System Information")
        print("6. List Loaded Models")
        print("7. Change Model Directory")
        print("8. TCP Server - Synthetic Data (port 5556)")
        print("9. TCP Server - Real Data (port 5558)")
        print("0. Exit")
        print("="*80)
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '1':
            csv_path = select_csv_file()
            if csv_path:
                system.classify_csv(csv_path)
        
        elif choice == '2':
            directory = select_directory()
            if directory:
                system.classify_directory_parallel(directory)
        
        elif choice == '3':
            print("\nVelocity Categories:")
            for i, category in enumerate(VELOCITY_CATEGORIES, 1):
                vel_range = VELOCITY_RANGES[category]
                print(f"  {i}. {category:15} [{vel_range['min']:.1f}-{vel_range['max']:.1f}]")
            
            try:
                idx = int(input("\nSelect category (1-7): ")) - 1
                if 0 <= idx < len(VELOCITY_CATEGORIES):
                    category = VELOCITY_CATEGORIES[idx]
                    n_sequences = int(input("Number of sequences: "))
                    if n_sequences > 0:
                        system.validate_synthetic_data(category, n_sequences)
            except:
                print("Invalid input")
        
        elif choice == '4':
            try:
                n = int(input(f"Number of workers [1-{cpu_count()}]: "))
                if 1 <= n <= cpu_count():
                    config.n_workers = n
                    system.config = config
                    print(f"\nWorkers set to: {n}")
            except:
                print("Invalid input")
        
        elif choice == '5':
            system._display_system_info()
        
        elif choice == '6':
            print(f"\nLoaded models: {len(model_manager.models)}")
            for name in model_manager.models.keys():
                print(f"  {name}")
        
        elif choice == '7':
            if model_manager.change_directory():
                model_manager.load_all()
                system = VelocityClassificationSystem(model_manager, config)
        
        elif choice == '8':
            system.start_synthetic_server()
        
        elif choice == '9':
            system.start_real_server()
        
        elif choice == '0':
            print("\nShutting down system...\n")
            break
        
        else:
            print("Invalid option")


if __name__ == "__main__":
    main()
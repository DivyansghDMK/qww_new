"""Serial communication classes for ECG hardware"""
import time
from typing import List, Dict, Optional
from PyQt5.QtWidgets import QMessageBox

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    print(" Serial module not available - ECG hardware features disabled")
    SERIAL_AVAILABLE = False
    # Create dummy serial classes
    class Serial:
        def __init__(self, *args, **kwargs): pass
        def close(self): pass
        def readline(self): return b''
        def read(self, size): return b''
        def write(self, data): pass
        def reset_input_buffer(self): pass
        def reset_output_buffer(self): pass
        @property
        def is_open(self): return False
    class SerialException(Exception): pass
    serial = type('Serial', (), {'Serial': Serial, 'SerialException': SerialException})()
    class MockComports:
        @staticmethod
        def comports(*args, **kwargs):
            return []
    serial.tools = type('Tools', (), {'list_ports': MockComports()})()

from utils.crash_logger import get_crash_logger
from .packet_parser import parse_packet, PACKET_SIZE, START_BYTE, END_BYTE

# Import hardware command handler (only if serial is available)
if SERIAL_AVAILABLE:
    from .hardware_commands import HardwareCommandHandler
else:
    # Dummy class for when serial is not available
    class HardwareCommandHandler:
        def __init__(self, *args, **kwargs): pass
        def send_start_command(self): return (False, None)
        def send_stop_command(self): return (False, None)
        def send_version_command(self): return (False, None, None)
        def send_close_command(self): return (False, None)


class SerialStreamReader:
    """Packet-based serial reader for ECG data - NEW IMPLEMENTATION"""
    
    @staticmethod
    def scan_and_detect_port(baudrate: int = 115200, timeout: float = 0.1) -> Optional[tuple]:
        """
        Scan all available COM ports and detect which one responds to START command
        
        Args:
            baudrate: Baud rate to use for scanning (default: 115200)
            timeout: Timeout per port in seconds (default: 0.1 = 100ms)
            
        Returns:
            tuple: (port_name: str, serial_port: Serial) if found, or None if none found
                   The serial port is already opened and has START command sent
        """
        if not SERIAL_AVAILABLE:
            print("❌ Port scanning failed: Serial module not available")
            return None
        
        print("\n" + "="*70)
        print("🔍 PORT SCANNING: Scanning all COM ports for ECG device...")
        print("="*70)
        
        # Get all available COM ports
        try:
            ports = serial.tools.list_ports.comports()
            port_list = [port.device for port in ports]
            print(f"📋 Found {len(port_list)} COM port(s): {', '.join(port_list)}")
        except Exception as e:
            print(f"❌ Error listing COM ports: {e}")
            return None
        
        if not port_list:
            print("⚠️ No COM ports found")
            print("="*70 + "\n")
            return None
        
        # Try each port
        detected_port = None
        detected_serial = None
        temp_ports = []  # Track opened ports to close them
        
        for port_name in port_list:
            print(f"\n🔌 Testing port: {port_name}")
            temp_ser = None
            try:
                # Open port with short timeout for quick scanning
                temp_ser = serial.Serial(
                    port=port_name,
                    baudrate=baudrate,
                    timeout=timeout,
                    write_timeout=timeout
                )
                temp_ports.append(temp_ser)
                
                # Clear any existing data
                temp_ser.reset_input_buffer()
                temp_ser.reset_output_buffer()
                
                # Create temporary command handler
                temp_handler = HardwareCommandHandler(temp_ser)
                
                # Send START command with reduced timeout for faster scanning (100ms)
                # Use quiet mode during scanning to reduce verbosity
                print(f"   📤 Sending START command to {port_name}...")
                success, response = temp_handler.send_start_command(timeout=timeout, quiet=True)
                
                if success:
                    print(f"   ✅ Port {port_name} responded with ACK!")
                    detected_port = port_name
                    detected_serial = temp_ser
                    # Remove from temp_ports so we don't close it
                    temp_ports.remove(temp_ser)
                    break
                else:
                    print(f"   ❌ Port {port_name} did not respond correctly")
                    temp_ser.close()
                    temp_ports.remove(temp_ser)
                    
            except serial.SerialException as e:
                print(f"   ⚠️ Port {port_name} error: {e}")
                if temp_ser and temp_ser in temp_ports:
                    try:
                        temp_ser.close()
                        temp_ports.remove(temp_ser)
                    except:
                        pass
            except Exception as e:
                print(f"   ⚠️ Port {port_name} unexpected error: {e}")
                if temp_ser and temp_ser in temp_ports:
                    try:
                        temp_ser.close()
                        temp_ports.remove(temp_ser)
                    except:
                        pass
        
        # Close all temporary ports (keep the detected one open)
        for temp_ser in temp_ports:
            try:
                if temp_ser.is_open:
                    temp_ser.close()
            except:
                pass
        
        if detected_port and detected_serial:
            print(f"\n✅ PORT DETECTED: {detected_port}")
        else:
            print(f"\n❌ No port responded to START command")
        
        print("="*70 + "\n")
        
        if detected_port and detected_serial:
            return (detected_port, detected_serial)
        return None
    
    def __init__(self, port: str, baudrate: int, timeout: float = 0.1):
        if not SERIAL_AVAILABLE:
            raise RuntimeError("pyserial is required for serial capture. pip install pyserial")
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        self.buf = bytearray()
        self.running = False
        self.data_count = 0
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_error_time = 0
        self.crash_logger = get_crash_logger()
        self.user_details = {}  # For error reporting compatibility
        # Packet loss tracking
        self.start_time = time.time()
        self.last_packet_time = time.time()
        self.total_packets_expected = 0
        self.total_packets_lost = 0
        self.packet_loss_percent = 0.0
        # Sequence-based packet loss detection
        self._last_packet_counter = None
        self._total_sequence_lost = 0
        self._packet_loss_warnings = 0
        # Initialize hardware command handler
        if SERIAL_AVAILABLE:
            self.command_handler = HardwareCommandHandler(self.ser)
        else:
            self.command_handler = None
        self.device_version = None
        print(f" SerialStreamReader initialized: Port={port}, Baud={baudrate}")

    def close(self) -> None:
        """Close serial connection"""
        try:
            # Stop data acquisition first
            self.running = False
            
            # Send CLOSE command to hardware if command handler is available
            if self.command_handler:
                try:
                    success, response = self.command_handler.send_close_command()
                    if not success:
                        print(" ⚠️ Warning: CLOSE command ACK not received")
                except Exception as e:
                    print(f" ⚠️ Error sending CLOSE command: {e}")
            
            # Flush any remaining data
            if hasattr(self.ser, 'reset_input_buffer'):
                try:
                    self.ser.reset_input_buffer()
                except Exception:
                    pass
            
            if hasattr(self.ser, 'reset_output_buffer'):
                try:
                    self.ser.reset_output_buffer()
                except Exception:
                    pass
            
            # Close the serial port
            if self.ser and self.ser.is_open:
                self.ser.close()
                print(" Serial port closed and released")
            
            # Clear buffer
            self.buf.clear()
        except Exception as e:
            print(f" Error closing serial connection: {e}")
    
    def get_device_version(self) -> Optional[str]:
        """
        Get device version from hardware using VERSION command.
        
        Returns:
            str or None: Version string if available, otherwise None.
        """
        # Only available when serial + HardwareCommandHandler are available
        if not hasattr(self, "command_handler") or not self.command_handler:
            print("❌ VERSION COMMAND: Command handler not available")
            return None

        try:
            print("\n" + "=" * 60)
            print("🔍 VERSION COMMAND: Requesting device version...")
            print("=" * 60)

            success, version, response = self.command_handler.send_version_command()
            if success and version:
                self.device_version = version
                print(f"✅ VERSION COMMAND: Success! Device version: '{version}'")
                print("=" * 60 + "\n")
                return version
            else:
                print("⚠️ VERSION COMMAND: Failed or no version received")
                print(f"   Success: {success}, Version: {version}, Response: {response}")
                print("=" * 60 + "\n")
                return None
        except Exception as e:
            print(f"❌ VERSION COMMAND: Error getting device version: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            print("=" * 60 + "\n")
            return None

    def start(self):
        """Start data acquisition using packet‑based protocol + hardware START command."""
        print(" Starting packet-based ECG data acquisition...")

        # Get current port info
        current_port = self.ser.port if hasattr(self.ser, "port") else None
        current_baudrate = self.ser.baudrate if hasattr(self.ser, "baudrate") else 115200
        current_port_open = self.ser.is_open if hasattr(self.ser, "is_open") else False

        # Ensure port is open
        if not current_port_open:
            if current_port:
                try:
                    print(f"   🔄 Opening port {current_port} at {current_baudrate} baud...")
                    self.ser = serial.Serial(
                        port=current_port,
                        baudrate=current_baudrate,
                        timeout=0.1,
                    )
                    print(f"   ✅ Port {current_port} opened")
                except Exception as e:
                    print(f"   ❌ Failed to open port {current_port}: {e}")
                    raise RuntimeError(f"Cannot open port {current_port}: {e}")

        # Verify port is open and ready
        if not self.ser.is_open:
            raise RuntimeError("Serial port is not open. Cannot start data acquisition.")

        # Clear buffers before sending START
        try:
            if hasattr(self.ser, "reset_input_buffer"):
                self.ser.reset_input_buffer()
            self.buf.clear()
        except Exception as e:
            print(f" ⚠️ Warning: Could not clear buffers before START: {e}")

        # First: optionally request and print device version while device is IDLE
        # (send STOP + VERSION, then we will START streaming below)
        try:
            version = self.get_device_version()
            if version:
                print(f" 🧬 ECG Device Version: {version}")
        except Exception as e:
            print(f" ⚠️ VERSION COMMAND skipped due to error: {e}")

        # Now send hardware START command to begin ECG streaming
        if hasattr(self, "command_handler") and self.command_handler:
            try:
                success, response = self.command_handler.send_start_command()
                if not success:
                    print(" ⚠️ Warning: START command ACK not received on current port")
                    print("   Continuing anyway - device may still send data...")
            except Exception as e:
                print(f" ⚠️ Warning: Error sending START command: {e}")
                print("   Continuing anyway - device may still send data...")
        else:
            print(" ⚠️ Warning: HardwareCommandHandler not available – skipping START command")

        # Mark reader as running and reset statistics
        self.running = True
        self.start_time = time.time()
        self.last_packet_time = time.time()
        self.data_count = 0
        self.total_packets_expected = 0
        self.total_packets_lost = 0
        self.packet_loss_percent = 0.0

        port_name = self.ser.port if hasattr(self.ser, "port") else "unknown"
        print(f" ✅ Packet-based ECG device started on port {port_name}")
        print(" 📡 Ready to receive data packets...")

    def stop(self):
        """Stop data acquisition and send hardware STOP command."""
        print(" Stopping packet-based ECG data acquisition...")
        self.running = False

        # Send STOP command to hardware if handler is available
        if hasattr(self, "command_handler") and self.command_handler:
            try:
                success, response = self.command_handler.send_stop_command()
                if not success:
                    print(" ⚠️ Warning: STOP command ACK not received")
            except Exception as e:
                print(f" ⚠️ Warning: Error sending STOP command: {e}")

        # Final packet loss statistics
        if hasattr(self, 'start_time') and self.start_time > 0:
            elapsed_time = time.time() - self.start_time
            expected_packets = int(500 * elapsed_time)  # 500 Hz
            total_lost = max(0, expected_packets - self.data_count)
            loss_percent = (total_lost / expected_packets * 100) if expected_packets > 0 else 0
            print(f" Total data packets received: {self.data_count}")
            if total_lost > 0:
                print(f" Packet loss summary: {total_lost}/{expected_packets} packets lost ({loss_percent:.2f}% loss)")
            else:
                print(f" No packet loss detected - all {expected_packets} expected packets received")
            
            # Report sequence-based packet loss
            if hasattr(self, '_total_sequence_lost') and self._total_sequence_lost > 0:
                print(f" Sequence-based packet loss: {self._total_sequence_lost} packets lost (detected via counter gaps)")
            else:
                print(f" No sequence gaps detected - perfect packet continuity!")
        else:
            print(f" Total data packets received: {self.data_count}")
            if hasattr(self, '_total_sequence_lost') and self._total_sequence_lost > 0:
                print(f" Sequence-based packet loss: {self._total_sequence_lost} packets lost")

    def read_packets(self, max_packets: int = 100) -> List[Dict[str, int]]:
        """Read and parse ECG packets from serial stream
        
        At 500 Hz, hardware sends 500 packets/second = ~16.67 packets per 33ms timer interval.
        We read up to max_packets to prevent buffer overflow and packet loss.
        """
        if not self.running:
            return []
            
        out: List[Dict[str, int]] = []
        
        try:
            bytes_to_read = 0
            if hasattr(self.ser, 'in_waiting'):
                try:
                    bytes_to_read = self.ser.in_waiting
                except Exception:
                    pass
            
            # If no data is available, return immediately to keep UI responsive
            if bytes_to_read == 0:
                return []
            
            # Cap the read size to prevent massive reads if buffer piled up
            # But ensure we read enough to drain the buffer if possible
            # Logic: Read available bytes, but respect our "catch up" logic sizes if they are larger/smaller
            target_read_size = 4096
            if len(self.buf) > 20000:
                target_read_size = 8192
            elif len(self.buf) > 50000:
                target_read_size = 16384
                
            # Read whichever is smaller: what's available or our max chunk size
            # (Actually, we should read all available to clear hardware buffer, but in chunks)
            read_size = min(bytes_to_read, target_read_size)
            
            chunk = self.ser.read(read_size)
            if chunk:
                self.buf.extend(chunk)

            # Extract packets - process ALL available packets to prevent buffer overflow
            # At 500 Hz, we need to process packets quickly to avoid accumulation
            # CRITICAL: Process ALL packets in buffer, not limited by max_packets, to prevent packet loss
            # Use max_iterations as safety limit, but process as many as needed
            max_iterations = max(max_packets * 5, 500)  # Allow catching up significantly if we fell behind
            iteration = 0
            packets_processed = 0
            
            while iteration < max_iterations and len(self.buf) >= PACKET_SIZE:
                iteration += 1
                start_idx = self.buf.find(bytes([START_BYTE]))
                if start_idx == -1:
                    # No start byte found - clear buffer if it's getting too large (>100KB)
                    if len(self.buf) > 100000:
                        print(f" Serial buffer overflow risk: {len(self.buf)} bytes, clearing buffer")
                        self.buf.clear()
                    break
                if start_idx > 10000:
                    # Skip too much garbage data before start byte
                    del self.buf[:start_idx]
                    continue
                if len(self.buf) - start_idx < PACKET_SIZE:
                    # Not enough data for a complete packet - keep what we have
                    if start_idx > 0:
                        del self.buf[:start_idx]
                    break
                    
                candidate = bytes(self.buf[start_idx : start_idx + PACKET_SIZE])
                del self.buf[: start_idx + PACKET_SIZE]

                if candidate[-1] != END_BYTE:
                    continue

                parsed = parse_packet(candidate)
                if parsed:
                    self.data_count += 1
                    self.last_packet_time = time.time()
                    packets_processed += 1
                    
                    # Extract packet counter for sequence tracking
                    packet_counter = candidate[1] & 0x3F  # Counter is in lower 6 bits (0-63)
                    
                    # Detect packet loss by checking sequence continuity
                    if self._last_packet_counter is not None:
                        expected_counter = (self._last_packet_counter + 1) % 64
                        if packet_counter != expected_counter:
                            # Calculate how many packets were lost
                            if packet_counter > expected_counter:
                                lost = packet_counter - expected_counter
                            else:
                                # Wrapped around (e.g., 63 -> 0)
                                lost = (64 - expected_counter) + packet_counter
                            
                            if lost > 0:
                                self._total_sequence_lost += lost
                                self._packet_loss_warnings += 1
                                # Only warn every 50th occurrence to avoid spam (optimized for performance)
                                if self._packet_loss_warnings % 50 == 1:
                                    print(f" ⚠️ Packet loss: {lost} dropped (Total: {self._total_sequence_lost})")
                    
                    self._last_packet_counter = packet_counter
                    
                    # Only log every 500th packet to reduce console spam (optimized for performance)
                    if self.data_count % 500 == 0:
                        loss_info = f" (Lost: {self._total_sequence_lost})" if self._total_sequence_lost > 0 else ""
                        print(f" 📡 Packet #{self.data_count}{loss_info}")
                    out.append(parsed)
            
            # If we processed many packets, we're catching up - this is good
            if packets_processed > max_packets * 2:
                # We're catching up from backlog - this is expected and good
                pass
            
            # Warn if buffer is accumulating too much data (indicates we're falling behind)
            if len(self.buf) > 50000:  # >50KB buffer indicates we're not reading fast enough
                if not hasattr(self, '_buffer_warn_time') or (time.time() - self._buffer_warn_time) > 5.0:
                    print(f" ⚠️ Buffer: {len(self.buf)} bytes")
                    self._buffer_warn_time = time.time()
            
            # Update packet loss statistics
            if self.running and self.data_count > 0:
                elapsed_time = time.time() - self.start_time
                expected_packets = int(500 * elapsed_time)  # 500 Hz = 500 packets/second
                self.total_packets_expected = expected_packets
                self.total_packets_lost = max(0, expected_packets - self.data_count)
                if expected_packets > 0:
                    self.packet_loss_percent = (self.total_packets_lost / expected_packets) * 100
                    
        except Exception as e:
            self.error_count += 1
            self.consecutive_errors += 1
            error_msg = f"Packet parsing error: {e}"
            print(f" {error_msg}")
            self.crash_logger.log_error(
                message=error_msg,
                exception=e,
                category="SERIAL_ERROR"
            )
            
            # If device is disconnected (Errno 6) or too many consecutive errors, stop
            if "Device not configured" in str(e) or "[Errno 6]" in str(e) or self.consecutive_errors > 20:
                print(" Critical serial error - stopping acquisition")
                self.running = False
            
        return out

    def _handle_serial_error(self, error):
        """Handle serial communication errors"""
        current_time = time.time()
        self.error_count += 1
        self.consecutive_errors += 1
        
        error_msg = f"Serial communication error: {error}"
        print(f" {error_msg}")
        
        self.crash_logger.log_error(
            message=error_msg,
            exception=error,
            category="SERIAL_ERROR"
        )
        
        if self.consecutive_errors >= 5 and (current_time - self.last_error_time) > 10:
            self.last_error_time = current_time
            self.consecutive_errors = 0


class SerialECGReader:
    """Legacy serial reader for line-based ECG data"""
    def __init__(self, port, baudrate):
        if not SERIAL_AVAILABLE:
            raise ImportError("Serial module not available - cannot create ECG reader")
        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.running = False
        self.data_count = 0
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_error_time = 0
        self.crash_logger = get_crash_logger()
        print(f" SerialECGReader initialized: Port={port}, Baud={baudrate}")

    def start(self):
        print(" Starting ECG data acquisition...")
        self.ser.reset_input_buffer()
        self.ser.write(b'1\r\n')
        # INSTANT START: Reduced delay for immediate wave display
        time.sleep(0.1)  # Reduced from 0.5s to 0.1s for faster startup
        self.running = True
        print(" ECG device started - waiting for data...")

    def stop(self):
        print(" Stopping ECG data acquisition...")
        self.ser.write(b'0\r\n')
        self.running = False
        print(f" Total data packets received: {self.data_count}")

    def read_value(self):
        if not self.running:
            return None
        try:
            line_raw = self.ser.readline()
            line_data = line_raw.decode('utf-8', errors='replace').strip()

            if line_data:
                self.data_count += 1
                # Print detailed data information
                print(f" [Packet #{self.data_count}] Raw data: '{line_data}' (Length: {len(line_data)})")
                
                # Parse and display ECG value
                if line_data.isdigit():
                    ecg_value = int(line_data[-3:])
                    print(f" ECG Value: {ecg_value} mV")
                    return ecg_value
                else:
                    # Try to parse as multiple values (8-channel data)
                    try:
                        # Clean the line data - remove any non-numeric characters except spaces and minus signs
                        import re
                        cleaned_line = re.sub(r'[^\d\s\-]', ' ', line_data)
                        values = [int(x) for x in cleaned_line.split() if x.strip() and x.replace('-', '').isdigit()]
                        
                        if len(values) >= 8:
                            print(f" 8-Channel ECG Data: {values}")
                            return values  # Return the list of 8 values
                        elif len(values) == 1:
                            print(f" Single ECG Value: {values[0]} mV")
                            return values[0]
                        elif len(values) > 0:
                            print(f" Unexpected number of values: {len(values)} (expected 8)")
                        else:
                            return None
                    except Exception as e:
                        print(f" Error parsing ECG data: {e}")
                        return None
            else:
                print("⏳ No data received (timeout)")
                
        except Exception as e:
            self._handle_serial_error(e)
        return None

    def close(self):
        print(" Closing serial connection...")
        self.ser.close()
        print(" Serial connection closed")

    def _handle_serial_error(self, error):
        """Handle serial communication errors with alert and logging"""
        current_time = time.time()
        self.error_count += 1
        self.consecutive_errors += 1
        
        # Log the error
        error_msg = f"Serial communication error: {error}"
        print(f" {error_msg}")
        
        # Log to crash logger
        self.crash_logger.log_error(
            message=error_msg,
            exception=error,
            category="SERIAL_ERROR"
        )
        
        # Show alert if consecutive errors exceed threshold
        if self.consecutive_errors >= 5 and (current_time - self.last_error_time) > 10:
            self._show_serial_error_alert(error)
            self.last_error_time = current_time
            self.consecutive_errors = 0  # Reset counter after showing alert
    
    def _show_serial_error_alert(self, error):
        """Show alert dialog for serial communication errors"""
        try:
            # Get user details from main application
            user_details = getattr(self, 'user_details', {})
            username = user_details.get('full_name', 'Unknown User')
            phone = user_details.get('phone', 'N/A')
            email = user_details.get('email', 'N/A')
            serial_id = user_details.get('serial_id', 'N/A')
            
            # Create detailed error message
            error_details = f"""
Serial Communication Error Detected!

Error: {str(error)}
User: {username}
Phone: {phone}
Email: {email}
Serial ID: {serial_id}
Machine Serial: {self.crash_logger.machine_serial_id or 'N/A'}
Time: {time.strftime('%Y-%m-%d %H:%M:%S')}

This error has been logged and an email notification will be sent to the support team.
            """
            
            # Show alert dialog
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Serial Communication Error")
            msg_box.setText("ECG Device Connection Lost")
            msg_box.setDetailedText(error_details)
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()
            
            # Send email notification
            self._send_error_email(error, user_details)
            
        except Exception as e:
            print(f" Error showing serial error alert: {e}")
    
    def _send_error_email(self, error, user_details):
        """Send email notification for serial errors"""
        try:
            # Create error data for email
            error_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error_type': 'Serial Communication Error',
                'error_message': str(error),
                'user_details': user_details,
                'machine_serial': self.crash_logger.machine_serial_id or 'N/A',
                'consecutive_errors': self.consecutive_errors,
                'total_errors': self.error_count
            }
            
            # Send email using crash logger
            self.crash_logger._send_crash_email(error_data)
            print(" Serial error email notification sent")
            
        except Exception as e:
            print(f" Error sending serial error email: {e}")

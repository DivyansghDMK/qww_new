"""
ECG Hardware Command Protocol Handler

Protocol Specification:
- Start Byte: 0xE8
- Counter: 0x00 (byte 2)
- Length: 0x11 (17 bytes, byte 3)
- OpCode/Code: Operation code (byte 4)
- Checksum: 0x00 (byte 5)
- Data: Bytes 6-21 (16 bytes)
- End Byte: 0x8E (byte 22)

Command OpCodes:
- 0x10: Start command
- 0x11: Stop command
- 0x14: Version check
- 0x15: Software close

Response Codes:
- 0x21: ACK (acknowledgment)
- 0x24: Actual data (for version response)
"""

import time
from typing import Optional, Dict, Tuple

# Protocol constants
START_BYTE = 0xE8
END_BYTE = 0x8E
PACKET_LENGTH = 0x11  # 17 bytes
FRAME_LEN = 22  # Total frame length (22 bytes)
ACK_CODE = 0x21
DATA_CODE_VERSION = 0x24  # Version data response

# Command OpCodes
OPCODE_START = 0x10
OPCODE_STOP = 0x11
OPCODE_VERSION = 0x14
OPCODE_CLOSE = 0x15

# Timeout for waiting for responses (seconds)
RESPONSE_TIMEOUT = 0.5  # Reduced from 2.0 for instant connection


class HardwareCommandHandler:
    """Handle hardware command protocol for ECG device"""
    
    def __init__(self, serial_port):
        """
        Initialize command handler
        
        Args:
            serial_port: pyserial Serial object
        """
        self.ser = serial_port
        self.counter = 0x00
    
    def _format_packet_details(self, packet: bytes, direction: str, command_name: str) -> str:
        """
        Format packet details for logging
        
        Args:
            packet: Packet bytes (22 bytes)
            direction: "SEND" or "RECV"
            command_name: Name of the command
            
        Returns:
            str: Formatted packet details
        """
        if len(packet) != 22:
            return f"Invalid packet length: {len(packet)} bytes"
        
        details = []
        details.append(f"\n{'='*70}")
        details.append(f"{direction}: {command_name} Packet")
        details.append(f"{'='*70}")
        details.append(f"Raw Hex: {packet.hex().upper()}")
        details.append(f"Byte-by-byte breakdown:")
        details.append(f"  [0] Start Byte:     0x{packet[0]:02X} ({'✓' if packet[0] == START_BYTE else '✗'})")
        details.append(f"  [1] Counter:       0x{packet[1]:02X} ({packet[1]})")
        details.append(f"  [2] Length:        0x{packet[2]:02X} ({packet[2]} bytes)")
        details.append(f"  [3] OpCode/Code:    0x{packet[3]:02X} ({self._get_code_name(packet[3])})")
        details.append(f"  [4] Checksum:       0x{packet[4]:02X}")
        details.append(f"  [5-20] Data:        {packet[5:21].hex().upper()}")
        if packet[3] == ACK_CODE and len(packet) > 5:
            details.append(f"      └─ Echoed OpCode: 0x{packet[5]:02X} ({self._get_code_name(packet[5])})")
        details.append(f"  [21] End Byte:      0x{packet[21]:02X} ({'✓' if packet[21] == END_BYTE else '✗'})")
        details.append(f"{'='*70}\n")
        return "\n".join(details)
    
    def _get_code_name(self, code: int) -> str:
        """Get human-readable name for OpCode/Response code"""
        code_map = {
            0x10: "START",
            0x11: "STOP",
            0x14: "VERSION",
            0x15: "CLOSE",
            0x21: "ACK",
            0x24: "VERSION_DATA"
        }
        return code_map.get(code, f"UNKNOWN(0x{code:02X})")
    
    def _build_command_packet(self, opcode: int) -> bytes:
        """
        Build a command packet according to protocol
        
        Args:
            opcode: Operation code (0x10, 0x11, 0x14, 0x15)
            
        Returns:
            bytes: Complete command packet (22 bytes)
        """
        packet = bytearray(22)
        packet[0] = START_BYTE      # Byte 0: Start byte
        packet[1] = self.counter    # Byte 1: Counter
        packet[2] = PACKET_LENGTH   # Byte 2: Length (0x11 = 17)
        packet[3] = opcode          # Byte 3: OpCode
        packet[4] = 0x00            # Byte 4: Checksum (currently 0x00)
        # Bytes 5-20: Data (all zeros for commands)
        for i in range(5, 21):
            packet[i] = 0x00
        packet[21] = END_BYTE       # Byte 21: End byte
        
        # Increment counter (wrap at 0x3F = 63)
        self.counter = (self.counter + 1) & 0x3F
        
        return bytes(packet)
    
    def _read_response(self, timeout: float = RESPONSE_TIMEOUT) -> Optional[bytes]:
        """
        Read response packet from device
        
        Args:
            timeout: Maximum time to wait for response (seconds)
            
        Returns:
            bytes: Response packet or None if timeout/error
        """
        start_time = time.time()
        buffer = bytearray()
        
        while (time.time() - start_time) < timeout:
            if self.ser.in_waiting > 0:
                chunk = self.ser.read(self.ser.in_waiting)
                buffer.extend(chunk)
                
                # Look for complete packet (START_BYTE ... END_BYTE, 22 bytes)
                start_idx = buffer.find(START_BYTE)
                if start_idx >= 0:
                    # Check if we have enough bytes for a complete packet
                    if len(buffer) >= start_idx + 22:
                        packet = bytes(buffer[start_idx:start_idx + 22])
                        if packet[-1] == END_BYTE and len(packet) == 22:
                            # Remove processed packet from buffer
                            buffer = buffer[start_idx + 22:]
                            return packet
            
            time.sleep(0.01)  # Small delay to avoid CPU spinning
        
        return None
    
    def _parse_response(self, packet: bytes) -> Dict[str, any]:
        """
        Parse response packet
        
        Args:
            packet: Response packet bytes (22 bytes)
            
        Returns:
            dict: Parsed response with keys: type, counter, length, code, opcode, data
        """
        if len(packet) != 22 or packet[0] != START_BYTE or packet[21] != END_BYTE:
            return {"type": "invalid", "error": "Invalid packet format"}
        
        response = {
            "type": "unknown",
            "counter": packet[1],
            "length": packet[2],
            "code": packet[3],
            "checksum": packet[4],
            "data": packet[5:21],  # Bytes 5-20 (16 bytes)
        }
        
        # Handle standard ACK code (0x21)
        if packet[3] == ACK_CODE:
            response["type"] = "ack"
            response["opcode"] = packet[5]  # Echoed OpCode in byte 5
        # Handle Version DATA code (0x24)
        elif packet[3] == DATA_CODE_VERSION:
            response["type"] = "version_data"
        # Handle device-specific code (0x20) - device uses this for both ACK and DATA
        elif packet[3] == 0x20:
            # Check byte 5 to determine if it's ACK (contains echoed OpCode) or DATA
            # For ACK: byte 5 should contain the echoed OpCode (0x10, 0x11, 0x14, 0x15)
            # For DATA: byte 5 might be 0x00 or start of data
            if packet[5] in [OPCODE_START, OPCODE_STOP, OPCODE_VERSION, OPCODE_CLOSE]:
                response["type"] = "ack"
                response["opcode"] = packet[5]  # Echoed OpCode in byte 5
            else:
                # Likely a DATA packet
                response["type"] = "data"
        else:
            response["type"] = "unknown"
            response["opcode"] = packet[3]
        
        return response
    
    def send_start_command(self, timeout: Optional[float] = None, quiet: bool = False) -> Tuple[bool, Optional[Dict]]:
        """
        Send Start command (OpCode 0x10)
        
        Args:
            timeout: Optional timeout in seconds (default: RESPONSE_TIMEOUT)
            quiet: If True, reduce verbosity (useful for port scanning)
        
        Returns:
            tuple: (success: bool, response: dict or None)
        """
        if timeout is None:
            timeout = RESPONSE_TIMEOUT
        
        if not quiet:
            print("\n" + "="*70)
            print("🚀 START COMMAND: Initiating communication with device")
            print("="*70)
        
        try:
            # Build and send command
            cmd_packet = self._build_command_packet(OPCODE_START)
            
            # Log what SOFTWARE is SENDING
            if not quiet:
                print(self._format_packet_details(cmd_packet, "SOFTWARE → DEVICE", "START Command"))
                print(f"📤 Software sending START command to device...")
            
            self.ser.write(cmd_packet)
            self.ser.flush()
            
            if not quiet:
                print(f"✅ Command packet written to serial port ({len(cmd_packet)} bytes)")
                print(f"⏳ Waiting for device response (timeout: {timeout}s)...")
            
            # Wait for ACK
            response_packet = self._read_response(timeout=timeout)
            if response_packet:
                # Log what DEVICE is SENDING BACK
                if not quiet:
                    print(self._format_packet_details(response_packet, "DEVICE → SOFTWARE", "START ACK Response"))
                    print(f"📥 Device responded with packet ({len(response_packet)} bytes)")
                
                response = self._parse_response(response_packet)
                if not quiet:
                    print(f"📋 Parsed response: {response}")
                elif quiet:
                    # In quiet mode, still show minimal info for debugging
                    print(f"   📥 Response: Code=0x{response.get('code', 0):02X}, Type={response.get('type')}, OpCode=0x{response.get('opcode', 0):02X}")
                
                is_ack = False
                if response["type"] == "ack" and response.get("opcode") == OPCODE_START:
                    is_ack = True
                elif response.get("code") == 0x20:
                    is_ack = True
                    
                if is_ack:
                    if not quiet:
                        print(f"✅ START COMMAND: Success! Device acknowledged START command")
                        print(f"   ACK OpCode: 0x{response.get('opcode', 0):02X} ({self._get_code_name(response.get('opcode', 0))})")
                        print("="*70 + "\n")
                    return True, response
                else:
                    if not quiet:
                        print(f"⚠️ START COMMAND: Unexpected response type or OpCode mismatch")
                        print(f"   Expected: ACK with OpCode 0x{OPCODE_START:02X}")
                        print(f"   Received: {response}")
                        print("="*70 + "\n")
                    elif quiet:
                        # In quiet mode, show why it failed
                        print(f"   ❌ Failed: Expected ACK with OpCode 0x{OPCODE_START:02X}, got {response.get('type')} with OpCode 0x{response.get('opcode', 0):02X}")
                    return False, response
            else:
                if not quiet:
                    print("❌ START COMMAND: No response received from device (timeout)")
                    print(f"   Device did not respond within {timeout} seconds")
                    print("="*70 + "\n")
                return False, None
                
        except Exception as e:
            if not quiet:
                print(f"❌ START COMMAND: Error occurred: {e}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
                print("="*70 + "\n")
            return False, None
    
    def send_stop_command(self) -> Tuple[bool, Optional[Dict]]:
        """
        Send Stop command (OpCode 0x11)
        
        Returns:
            tuple: (success: bool, response: dict or None)
        """
        print("\n" + "="*70)
        print("🛑 STOP COMMAND: Requesting device to stop")
        print("="*70)
        
        try:
            # Build and send command
            cmd_packet = self._build_command_packet(OPCODE_STOP)
            
            # Log what SOFTWARE is SENDING
            print(self._format_packet_details(cmd_packet, "SOFTWARE → DEVICE", "STOP Command"))
            print(f"📤 Software sending STOP command to device...")
            
            self.ser.write(cmd_packet)
            self.ser.flush()
            
            print(f"✅ Command packet written to serial port ({len(cmd_packet)} bytes)")
            print(f"⏳ Waiting for device response (timeout: {RESPONSE_TIMEOUT}s)...")
            
            # Wait for ACK
            response_packet = self._read_response()
            if response_packet:
                # Log what DEVICE is SENDING BACK
                print(self._format_packet_details(response_packet, "DEVICE → SOFTWARE", "STOP ACK Response"))
                print(f"📥 Device responded with packet ({len(response_packet)} bytes)")
                
                response = self._parse_response(response_packet)
                print(f"📋 Parsed response: {response}")
                
                is_ack = False
                if response["type"] == "ack" and response.get("opcode") == OPCODE_STOP:
                    is_ack = True
                elif response.get("code") == 0x20:
                    is_ack = True
                    
                if is_ack:
                    print(f"✅ STOP COMMAND: Success! Device acknowledged STOP command")
                    print(f"   ACK OpCode: 0x{response.get('opcode', 0):02X} ({self._get_code_name(response.get('opcode', 0))})")
                    print("="*70 + "\n")
                    return True, response
                else:
                    print(f"⚠️ STOP COMMAND: Unexpected response type or OpCode mismatch")
                    print(f"   Expected: ACK with OpCode 0x{OPCODE_STOP:02X}")
                    print(f"   Received: {response}")
                    print("="*70 + "\n")
                    return False, response
            else:
                print("❌ STOP COMMAND: No response received from device (timeout)")
                print(f"   Device did not respond within {RESPONSE_TIMEOUT} seconds")
                print("="*70 + "\n")
                return False, None
                
        except Exception as e:
            print(f"❌ STOP COMMAND: Error occurred: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            print("="*70 + "\n")
            return False, None
    
    def _read_packet(self, timeout: float = RESPONSE_TIMEOUT) -> bytes:
        """
        Read a complete packet (22 bytes) from serial port
        Reads one byte at a time, looks for START_BYTE and END_BYTE
        
        Args:
            timeout: Maximum time to wait for packet (seconds)
            
        Returns:
            bytes: Complete packet (22 bytes)
            
        Raises:
            TimeoutError: If packet not received within timeout
        """
        start_time = time.time()
        buffer = bytearray()
        
        while (time.time() - start_time) < timeout:
            if self.ser.in_waiting:
                b = self.ser.read(1)
                if not b:
                    continue
                
                if not buffer:
                    if b[0] == START_BYTE:
                        buffer.append(b[0])
                else:
                    buffer.append(b[0])
                    if len(buffer) == FRAME_LEN:
                        if buffer[-1] == END_BYTE:
                            return bytes(buffer)
                        buffer.clear()
            
            time.sleep(0.01)  # Small delay to avoid CPU spinning
        
        raise TimeoutError("Timeout waiting for packet")
    
    def _wait_for_ack(self, expected_opcode: int, timeout: float = 3.0) -> bytes:
        """
        Wait for ACK frame, filtering out ECG streaming frames (0x20)
        
        Args:
            expected_opcode: Expected opcode in ACK response (byte 5)
            timeout: Maximum time to wait (seconds)
            
        Returns:
            bytes: ACK frame
            
        Raises:
            TimeoutError: If ACK not received within timeout
        """
        ECG_STREAM = 0x20
        ACK_CODE = 0x21
        
        t0 = time.time()
        while time.time() - t0 < timeout:
            frame = self._read_packet(timeout=0.5)
            code = frame[3]
            
            # Ignore ECG streaming frames
            if code == ECG_STREAM:
                continue
            
            if code != ACK_CODE:
                continue
            
            if frame[5] != expected_opcode:
                continue
            
            return frame
        
        raise TimeoutError(f"No ACK for opcode 0x{expected_opcode:02X}")
    
    def _wait_for_data(self, timeout: float = 3.0) -> bytes:
        """
        Wait for DATA frame, filtering out ECG streaming frames (0x20)
        
        Args:
            timeout: Maximum time to wait (seconds)
            
        Returns:
            bytes: DATA frame
            
        Raises:
            TimeoutError: If DATA frame not received within timeout
        """
        ECG_STREAM = 0x20
        DATA_CODE = 0x24
        
        t0 = time.time()
        while time.time() - t0 < timeout:
            frame = self._read_packet(timeout=0.5)
            code = frame[3]
            
            # Ignore ECG streaming frames
            if code == ECG_STREAM:
                continue
            
            if code == DATA_CODE:
                return frame
        
        raise TimeoutError("No DATA frame received")
    
    def _calc_checksum(self, pkt: bytes) -> int:
        """
        Calculate checksum for packet (sum of bytes 1-20, masked to 8 bits)
        
        Args:
            pkt: Packet bytes (22 bytes)
            
        Returns:
            int: Checksum value (0-255)
        """
        return sum(pkt[1:21]) & 0xFF
    
    def _decode_version_payload(self, payload: bytes) -> str:
        """
        Decode version payload as hex string
        Version data is NOT ASCII, best reliable representation = hex string
        
        Args:
            payload: 16 bytes of version data (bytes 5-20 of the frame)
            
        Returns:
            str: Hex string representation of version (uppercase)
        """
        return payload.hex().upper()
    
    def _parse_packet(self, pkt: bytes) -> Dict:
        """
        Parse packet into dictionary
        
        Args:
            pkt: Packet bytes (22 bytes)
            
        Returns:
            dict: Parsed packet with keys: counter, length, code, checksum, data, raw
        """
        return {
            "counter": pkt[1],
            "length": pkt[2],
            "code": pkt[3],
            "checksum": pkt[4],
            "data": pkt[5:21],
            "raw": pkt
        }
    
    def _build_simple_packet(self, counter: int, opcode: int) -> bytes:
        """
        Build a simple command packet
        
        Args:
            counter: Packet counter
            opcode: Command opcode
            
        Returns:
            bytes: Complete packet (22 bytes)
        """
        pkt = bytearray(22)
        pkt[0] = START_BYTE
        pkt[1] = counter & 0xFF
        pkt[2] = 0x11  # LEN_BYTE
        pkt[3] = opcode
        pkt[4] = 0x00
        pkt[5:21] = bytes(16)
        pkt[21] = END_BYTE
        return bytes(pkt)
    
    def _send_stop(self, timeout: float = 3.0) -> bool:
        """
        Send STOP command to put device in IDLE state

        Args:
            timeout: Timeout in seconds
        
        Returns:
            bool: True if STOP ACK received successfully
        """
        self.ser.reset_input_buffer()
        pkt = self._build_simple_packet(0, OPCODE_STOP)
        print("📤 STOP →", pkt.hex(" ").upper())
        self.ser.write(pkt)
        self.ser.flush()
        
        ack = self._wait_for_ack(OPCODE_STOP, timeout=timeout)
        print("📥 STOP ACK ←", ack.hex(" ").upper())
        return True
    
    def send_version_command(self, counter: int = 0, timeout: float = 1.0, retries: int = 2) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Hardware protocol:
        App → 0x14
        Device → 0x21 (ACK, byte[5]=0x14)
        Device → 0x24 (DATA, bytes[5:21]=ASCII version)
        
        First sends STOP to put device in IDLE, then requests version.
        Filters out ECG streaming frames (0x20) that might interfere.
        
        Args:
            counter: Packet counter (default: 0, but uses 1 for VERSION packet)
            timeout: Timeout in seconds (increased for stability)
            retries: Number of retry attempts
        
        Returns:
            tuple: (success: bool, version_string: str or None, response: dict or None)
        """
        for attempt in range(retries + 1):
            if attempt > 0:
                print(f"🔄 Retrying VERSION command (attempt {attempt + 1}/{retries + 1})...")
                time.sleep(0.2) # Wait before retry

            print("\n" + "="*60)
            print(f"🔍 VERSION COMMAND (Attempt {attempt + 1}): Requesting device version...")
            print("="*60)
            
            CMD_VERSION = OPCODE_VERSION  # 0x14
            
            try:
                # First, stop the device if it's streaming
                try:
                    self._send_stop(timeout=timeout)
                    print("✅ STOP confirmed")
                    time.sleep(0.1)
                except TimeoutError:
                    # Device might already be stopped, continue anyway
                    print("⚠️ STOP ACK timeout (device may already be stopped)")
                
                # Reset buffer before sending VERSION
                self.ser.reset_input_buffer()
                
                # Send VERSION command
                pkt = self._build_simple_packet(1, CMD_VERSION)
                print("📤 VERSION →", pkt.hex(" ").upper())
                self.ser.write(pkt)
                self.ser.flush()
                
                # Wait for ACK (filters out ECG_STREAM frames)
                ack = self._wait_for_ack(CMD_VERSION, timeout=timeout)
                print("📥 VERSION ACK ←", ack.hex(" ").upper())
                
                # Wait for DATA (filters out ECG_STREAM frames)
                data = self._wait_for_data(timeout=timeout)
                print("📥 VERSION DATA ←", data.hex(" ").upper())
                
                # Decode version from bytes 5-21
                try:
                    version = data[5:21].decode("ascii").rstrip("\x00").strip()
                except UnicodeDecodeError:
                    version = data[5:21].hex().upper()
                
                print("\n" + "="*60)
                print("✅ VERSION COMMAND SUCCESS")
                print("="*60)
                print("✅ DEVICE VERSION:", version)
                print("="*60 + "\n")
                
                # Create response dict for compatibility
                data_response = {
                    "type": "version_data",
                    "counter": data[1],
                    "length": data[2],
                    "code": data[3],
                    "checksum": data[4],
                    "data": data[5:21],
                }
                
                return True, version, data_response
                    
            except TimeoutError as e:
                print(f"❌ VERSION COMMAND: {e}")
                if attempt == retries:
                    print("="*60 + "\n")
                    return False, None, None
            except Exception as e:
                print(f"❌ VERSION COMMAND: Error occurred: {e}")
                if attempt == retries:
                    import traceback
                    print(f"   Traceback: {traceback.format_exc()}")
                    print("="*60 + "\n")
                    return False, None, None
        
        return False, None, None
    
    def send_close_command(self) -> Tuple[bool, Optional[Dict]]:
        """
        Send Software close command (OpCode 0x15)
        
        Returns:
            tuple: (success: bool, response: dict or None)
        """
        print("\n" + "="*70)
        print("🔒 CLOSE COMMAND: Requesting device to close connection")
        print("="*70)
        
        try:
            # Build and send command
            cmd_packet = self._build_command_packet(OPCODE_CLOSE)
            
            # Log what SOFTWARE is SENDING
            print(self._format_packet_details(cmd_packet, "SOFTWARE → DEVICE", "CLOSE Command"))
            print(f"📤 Software sending CLOSE command to device...")
            
            self.ser.write(cmd_packet)
            self.ser.flush()
            
            print(f"✅ Command packet written to serial port ({len(cmd_packet)} bytes)")
            print(f"⏳ Waiting for device response (timeout: {RESPONSE_TIMEOUT}s)...")
            
            # Wait for ACK
            response_packet = self._read_response()
            if response_packet:
                # Log what DEVICE is SENDING BACK
                print(self._format_packet_details(response_packet, "DEVICE → SOFTWARE", "CLOSE ACK Response"))
                print(f"📥 Device responded with packet ({len(response_packet)} bytes)")
                
                response = self._parse_response(response_packet)
                print(f"📋 Parsed response: {response}")
                
                is_ack = False
                if response["type"] == "ack" and response.get("opcode") == OPCODE_CLOSE:
                    is_ack = True
                elif response.get("code") == 0x20:
                    is_ack = True
                    
                if is_ack:
                    print(f"✅ CLOSE COMMAND: Success! Device acknowledged CLOSE command")
                    print(f"   ACK OpCode: 0x{response.get('opcode', 0):02X} ({self._get_code_name(response.get('opcode', 0))})")
                    print("="*70 + "\n")
                    return True, response
                else:
                    print(f"⚠️ CLOSE COMMAND: Unexpected response type or OpCode mismatch")
                    print(f"   Expected: ACK with OpCode 0x{OPCODE_CLOSE:02X}")
                    print(f"   Received: {response}")
                    print("="*70 + "\n")
                    return False, response
            else:
                print("❌ CLOSE COMMAND: No response received from device (timeout)")
                print(f"   Device did not respond within {RESPONSE_TIMEOUT} seconds")
                print("="*70 + "\n")
                return False, None
                
        except Exception as e:
            print(f"❌ CLOSE COMMAND: Error occurred: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            print("="*70 + "\n")
            return False, None


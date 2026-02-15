 
#!/usr/bin/env python3

'''
Total Battle Mercenary Exchange Detector
Monitors your screen and alerts when a mercenary exchange is detected
Enhanced with purple roof color detection to ignore terrain
'''

import cv2
import numpy as np
import mss
import time
import threading
import os
from pathlib import Path

# Try to import audio library
try:
    import pygame
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: pygame not available. Install with: pip install pygame")

class MercExchangeDetector:
    def __init__(self, template_paths, threshold=0.7, scan_interval=0.5, purple_only=True):
        """
        Initialize the detector
        
        Args:
            template_paths: Single path or list of paths to mercenary exchange screenshots
            threshold: Detection confidence threshold (0.0-1.0)
            scan_interval: Seconds between screen scans
            purple_only: If True, only detect purple roof (ignore terrain). If False, use hybrid method.
        """
        # Handle single template or list of templates
        if isinstance(template_paths, str):
            template_paths = [template_paths]
        
        self.template_paths = template_paths
        self.threshold = threshold
        self.scan_interval = scan_interval
        self.purple_only = purple_only
        self.running = False
        self.last_detection_time = 0
        self.cooldown = 3.0  # Seconds between alerts to avoid spam
        
        # Purple color range in HSV
        # Adjust these if detection is too loose or too strict
        self.lower_purple = np.array([120, 50, 50])
        self.upper_purple = np.array([160, 255, 255])
        
        # Load all template images
        self.templates = []
        for path in template_paths:
            template = cv2.imread(path)
            if template is None:
                print(f"Warning: Could not load template image: {path}")
                continue
            
            # Store original, grayscale, and HSV versions
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
            
            # Create purple mask for this template
            template_purple_mask = cv2.inRange(template_hsv, self.lower_purple, self.upper_purple)
            
            self.templates.append({
                'path': path,
                'original': template,
                'gray': template_gray,
                'hsv': template_hsv,
                'purple_mask': template_purple_mask,
                'width': template.shape[1],
                'height': template.shape[0]
            })
            
            print(f"Template loaded: {path} ({template.shape[1]}x{template.shape[0]} pixels)")
        
        if not self.templates:
            raise ValueError("No valid template images could be loaded!")
        
        print(f"Total templates loaded: {len(self.templates)}")
        print(f"Detection mode: {'Purple-only' if purple_only else 'Hybrid (purple + grayscale)'}")
        
        # Setup screen capture
        self.sct = mss.mss()
        
        # Generate alert sound if audio available
        self.setup_audio()
        
    def setup_audio(self):
        """Generate a simple beep sound for alerts"""
        if not AUDIO_AVAILABLE:
            return
            
        # Generate a 440Hz beep (A note)
        duration = 0.3  # seconds
        sample_rate = 44100
        frequency = 440
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        wave = np.sin(2 * np.pi * frequency * t)
        
        # Add envelope to avoid clicks
        envelope = np.exp(-3 * t / duration)
        wave = wave * envelope
        
        # Convert to 16-bit audio
        audio = wave * (2**15 - 1) / np.max(np.abs(wave))
        audio = audio.astype(np.int16)
        
        # Make stereo by duplicating the channel
        self.beep_audio = np.column_stack((audio, audio))
        
    def play_alert_sound(self):
        """Play alert sound"""
        if not AUDIO_AVAILABLE:
            return
            
        try:
            # Create sound from numpy array
            sound = pygame.sndarray.make_sound(self.beep_audio)
            sound.play()
        except Exception as e:
            print(f"Audio error: {e}")
    
    def capture_screen(self):
        """Capture the current screen"""
        # Capture primary monitor
        monitor = self.sct.monitors[1]
        screenshot = self.sct.grab(monitor)
        
        # Convert to numpy array
        img = np.array(screenshot)
        
        # Convert BGRA to BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img
    
    def detect_exchange(self, screen):
        """
        Detect mercenary exchange in the screen image using all templates
        Uses purple roof color detection to ignore terrain
        
        Returns:
            list of (x, y, confidence, template_index) tuples for detections
        """
        # Convert to HSV for color detection
        screen_hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        
        # Create a mask for purple colors (the roof)
        purple_mask = cv2.inRange(screen_hsv, self.lower_purple, self.upper_purple)
        
        # Clean up the mask (remove noise)
        kernel = np.ones((3,3), np.uint8)
        purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel)
        purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, kernel)
        
        all_detections = []
        
        if self.purple_only:
            # Purple-only detection (ignores terrain completely)
            for idx, template in enumerate(self.templates):
                # Match using ONLY the purple roof
                result = cv2.matchTemplate(purple_mask, template['purple_mask'], cv2.TM_CCOEFF_NORMED)
                
                # Find locations where match exceeds threshold
                locations = np.where(result >= self.threshold)
                
                for pt in zip(*locations[::-1]):
                    confidence = result[pt[1], pt[0]]
                    all_detections.append((pt[0], pt[1], confidence, idx))
        else:
            # Hybrid detection (purple + grayscale for verification)
            screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            
            for idx, template in enumerate(self.templates):
                # Match using purple mask
                result_purple = cv2.matchTemplate(purple_mask, template['purple_mask'], cv2.TM_CCOEFF_NORMED)
                
                # Also do grayscale matching for verification
                result_gray = cv2.matchTemplate(screen_gray, template['gray'], cv2.TM_CCOEFF_NORMED)
                
                # Combine both results (weighted average - purple weighted more)
                combined_result = 0.7 * result_purple + 0.3 * result_gray
                
                # Find locations where match exceeds threshold
                locations = np.where(combined_result >= self.threshold)
                
                for pt in zip(*locations[::-1]):
                    confidence = combined_result[pt[1], pt[0]]
                    all_detections.append((pt[0], pt[1], confidence, idx))
        
        # Remove duplicate detections
        all_detections = self.nms_multi_template(all_detections)
        
        return all_detections
    
    def nms_multi_template(self, detections, overlap_threshold=0.3):
        """Non-maximum suppression for multiple templates"""
        if len(detections) == 0:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x[2], reverse=True)
        
        keep = []
        
        for det in detections:
            x, y, conf, template_idx = det
            template = self.templates[template_idx]
            
            # Check if this overlaps with any kept detection
            overlaps = False
            for kept_x, kept_y, _, kept_idx in keep:
                kept_template = self.templates[kept_idx]
                
                # Calculate overlap using average dimensions
                avg_width = (template['width'] + kept_template['width']) / 2
                avg_height = (template['height'] + kept_template['height']) / 2
                
                dx = abs(x - kept_x)
                dy = abs(y - kept_y)
                
                if dx < avg_width * overlap_threshold and dy < avg_height * overlap_threshold:
                    overlaps = True
                    break
            
            if not overlaps:
                keep.append(det)
        
        return keep
    
    def draw_detections(self, screen, detections):
        """Draw bounding boxes around detections"""
        result = screen.copy()
        
        for x, y, conf, template_idx in detections:
            template = self.templates[template_idx]
            h, w = template['height'], template['width']
            
            # Draw red rectangle
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 3)
            
            # Add confidence text
            text = f"MERC EXCHANGE! {conf:.2%} (T{template_idx+1})"
            cv2.putText(result, text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result
    
    def show_purple_detection(self, screen, save_path='purple_detection_test.png'):
        """Debug tool - shows what purple areas are being detected"""
        screen_hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        
        purple_mask = cv2.inRange(screen_hsv, self.lower_purple, self.upper_purple)
        
        # Create visualization
        purple_highlight = screen.copy()
        purple_highlight[purple_mask > 0] = [255, 0, 255]  # Magenta highlight
        
        # Blend with original
        result = cv2.addWeighted(screen, 0.7, purple_highlight, 0.3, 0)
        
        cv2.imwrite(save_path, result)
        print(f"Saved purple detection visualization to: {save_path}")
        return result
    
    def alert(self, detections):
        """Trigger visual and audio alerts"""
        current_time = time.time()
        
        # Check cooldown to avoid spam
        if current_time - self.last_detection_time < self.cooldown:
            return
        
        self.last_detection_time = current_time
        
        # Console alert
        print(f"\n{'='*60}")
        print(f"🎯 MERCENARY EXCHANGE DETECTED! 🎯")
        print(f"Found {len(detections)} exchange(s) on screen")
        for i, (x, y, conf, template_idx) in enumerate(detections, 1):
            print(f"  Location {i}: ({x}, {y}) - Confidence: {conf:.2%} - Template: {template_idx+1}")
        print(f"{'='*60}\n")
        
        # Play sound alert
        if AUDIO_AVAILABLE:
            # Play 3 beeps
            for _ in range(3):
                self.play_alert_sound()
                time.sleep(0.15)
    
    def monitor_loop(self):
        """Main monitoring loop"""
        print("\n" + "="*60)
        print("MERCENARY EXCHANGE DETECTOR - ACTIVE")
        print("="*60)
        print(f"Scanning every {self.scan_interval}s")
        print(f"Detection threshold: {self.threshold:.0%}")
        print(f"Using {len(self.templates)} template(s)")
        print(f"Mode: {'Purple-only' if self.purple_only else 'Hybrid'}")
        print(f"Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        scan_count = 0
        
        while self.running:
            try:
                # Capture screen
                screen = self.capture_screen()
                
                # Detect exchanges
                detections = self.detect_exchange(screen)
                
                scan_count += 1
                
                if detections:
                    # Alert user
                    self.alert(detections)
                    
                    # Save screenshot with detection
                    detected_img = self.draw_detections(screen, detections)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"merc_detected_{timestamp}.png"
                    cv2.imwrite(filename, detected_img)
                    print(f"Screenshot saved: {filename}")
                    
                    # Also save purple detection visualization for debugging
                    debug_filename = f"purple_debug_{timestamp}.png"
                    self.show_purple_detection(screen, debug_filename)
                    print(f"Debug visualization saved: {debug_filename}")
                else:
                    # Status update every 20 scans
                    if scan_count % 20 == 0:
                        print(f"Scanning... ({scan_count} scans completed, no exchanges detected)")
                
                # Wait before next scan
                time.sleep(self.scan_interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                import traceback
                traceback.print_exc()
    
    def start(self):
        """Start monitoring in background thread"""
        if self.running:
            print("Detector is already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)
        print("\nDetector stopped")


def main():
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description='Detect mercenary exchanges in Total Battle')
    parser.add_argument('--template', '-t', nargs='+',
                       help='Path(s) to mercenary exchange template image(s). Can specify multiple.')
    parser.add_argument('--template-dir', '-d', default='templates',
                       help='Directory containing template images (default: templates/)')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Detection confidence threshold (0.0-1.0)')
    parser.add_argument('--interval', type=float, default=0.5,
                       help='Scan interval in seconds')
    parser.add_argument('--hybrid', action='store_true',
                       help='Use hybrid detection (purple + grayscale) instead of purple-only')
    parser.add_argument('--purple-range', nargs=6, type=int, metavar=('H_MIN', 'S_MIN', 'V_MIN', 'H_MAX', 'S_MAX', 'V_MAX'),
                       help='Custom HSV range for purple detection (default: 120 50 50 160 255 255)')
    
    args = parser.parse_args()
    
    # Collect template paths
    template_paths = []
    
    # Add templates from --template argument if specified
    if args.template:
        template_paths.extend(args.template)
    
    # Add templates from directory (default or specified)
    if os.path.isdir(args.template_dir):
        dir_templates = glob.glob(os.path.join(args.template_dir, '*.png'))
        template_paths.extend(dir_templates)
    
    # Fallback: try current directory for merc_exchange.png
    if not template_paths:
        if os.path.exists('merc_exchange.png'):
            template_paths.append('merc_exchange.png')
    
    # Verify at least one template exists
    existing_templates = [t for t in template_paths if os.path.exists(t)]
    
    if not existing_templates:
        print("Error: No template images found!")
        print("\nSearched in:")
        print(f"  1. Directory: {args.template_dir}/")
        print(f"  2. Current directory: merc_exchange.png")
        print("\nPlease either:")
        print("  - Create a 'templates/' folder and add .png images")
        print("  - Place merc_exchange.png in the current directory")
        print("  - Use: --template image1.png image2.png")
        print("  - Use: --template-dir /path/to/templates/")
        return
    
    print(f"Found {len(existing_templates)} template(s):")
    for t in existing_templates:
        print(f"  - {t}")
    print()
    
    # Create detector
    detector = MercExchangeDetector(
        template_paths=existing_templates,
        threshold=args.threshold,
        scan_interval=args.interval,
        purple_only=not args.hybrid
    )
    
    # Apply custom purple range if specified
    if args.purple_range:
        detector.lower_purple = np.array(args.purple_range[:3])
        detector.upper_purple = np.array(args.purple_range[3:])
        print(f"Using custom purple range: {args.purple_range}")
    
    # Start monitoring
    try:
        detector.running = True
        detector.monitor_loop()
    except KeyboardInterrupt:
        print("\n\nStopping detector...")
        detector.stop()


if __name__ == "__main__":
    main()

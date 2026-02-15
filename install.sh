#!/bin/bash
# Installation script for Total Battle Mercenary Exchange Detector

echo "=========================================="
echo "Total Battle Mercenary Exchange Detector"
echo "Installation Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "Error: Python 3 is required but not found"
    exit 1
fi

echo ""
echo "Installing required Python packages..."
echo ""

# Install dependencies
pip3 install opencv-python numpy mss simpleaudio pillow

if [ $? -ne 0 ]; then
    echo ""
    echo "Warning: Some packages may have failed to install"
    echo "You can try installing them manually:"
    echo "  pip3 install --user opencv-python numpy mss simpleaudio pillow"
    echo ""
fi

# Make the main script executable
chmod +x merc_exchange_detector.py

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Usage:"
echo "  python3 merc_exchange_detector.py --template merc_exchange.png"
echo ""
echo "Options:"
echo "  --threshold 0.7    Detection sensitivity (0.0-1.0, higher = more strict)"
echo "  --interval 0.5     Scan interval in seconds"
echo ""
echo "Make sure merc_exchange.png is in the same directory!"
echo "=========================================="

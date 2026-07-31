#!/bin/bash
# simulate_attacks.sh
# Simulates various network attacks to test Network Guardian
# Run with: sudo bash simulate_attacks.sh

TARGET=$(hostname -I | awk '{print $1}')
echo "============================================"
echo " NeuraShield Attack Simulation"
echo " Target IP: $TARGET"
echo "============================================"
echo ""

# Give monitor.py time to start if needed
sleep 2

echo "[1/7] SYN Port Scan (nmap)..."
echo "      Scanning ports 1-1000 on $TARGET"
sudo nmap -sS -p 1-1000 --host-timeout 30s $TARGET 2>/dev/null
echo "      Done."
sleep 3

echo ""
echo "[2/7] UDP Port Scan..."
sudo nmap -sU -p 1-200 --host-timeout 30s $TARGET 2>/dev/null
echo "      Done."
sleep 3

echo ""
echo "[3/7] ICMP Flood (ping flood)..."
echo "      Sending 1000 rapid pings..."
sudo hping3 -1 --flood -c 1000 $TARGET 2>/dev/null &
HPING_PID=$!
sleep 5
kill $HPING_PID 2>/dev/null
echo "      Done."
sleep 2

echo ""
echo "[4/7] SYN Flood (DoS simulation)..."
echo "      Sending 500 SYN packets to port 80..."
sudo hping3 -S -p 80 -c 500 --flood $TARGET 2>/dev/null &
HPING_PID=$!
sleep 5
kill $HPING_PID 2>/dev/null
echo "      Done."
sleep 2

echo ""
echo "[5/7] Connection to suspicious ports..."
echo "      Probing known malware ports..."
for port in 4444 6667 31337 9001 1337; do
    echo "      Probing port $port..."
    timeout 2 bash -c "echo '' > /dev/tcp/$TARGET/$port" 2>/dev/null
    sleep 1
done
echo "      Done."
sleep 2

echo ""
echo "[6/7] Rapid connection attempts (brute force simulation)..."
echo "      Rapid connections to port 22..."
for i in $(seq 1 20); do
    timeout 1 bash -c "echo '' > /dev/tcp/$TARGET/22" 2>/dev/null &
done
wait
echo "      Done."
sleep 2

echo ""
echo "[7/7] Large data transfer simulation..."
echo "      Generating large outbound traffic..."
dd if=/dev/urandom bs=1M count=5 2>/dev/null | \
    timeout 10 nc -w 5 8.8.8.8 443 2>/dev/null || true
echo "      Done."
sleep 2

echo ""
echo "============================================"
echo " Simulation complete!"
echo " Check alerts with:"
echo " cat ~/neurashield/alerts.json | python3 -m json.tool | head -100"
echo "============================================"

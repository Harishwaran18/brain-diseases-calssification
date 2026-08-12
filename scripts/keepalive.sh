#!/usr/bin/env bash
# NeuroCure keepalive supervisor.
# Runs the Streamlit app in a loop so it auto-restarts if it ever crashes,
# and fully detaches from the controlling terminal (setsid + nohup) so it
# survives session disconnects and runs 24/7.
set -u

cd /workspace/project/brain-diseases-calssification
PY=/workspace/project/brain-diseases-calssification/.venv/bin/python
LOG=/tmp/neurocure.log
PIDFILE=/tmp/neurocure.pid

echo "[$(date -Is)] keepalive supervisor starting" >> "$LOG"

while true; do
    echo "[$(date -Is)] launching streamlit..." >> "$LOG"
    "$PY" -m streamlit run neurocure_app/app.py \
        --server.port 12000 \
        --server.headless true \
        --server.address 0.0.0.0 \
        --server.fileWatcherType none \
        --browser.gatherUsageStats false \
        >> "$LOG" 2>&1
    EXIT=$?
    echo "[$(date -Is)] streamlit exited with code $EXIT; restarting in 3s" >> "$LOG"
    sleep 3
done

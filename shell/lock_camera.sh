#!/usr/bin/env bash
# Lock camera exposure / white balance / gain to constant values so brightness
# does not drift between recording and eval.
#
# IMPORTANT: OpenCV (cv2.VideoCapture) resets these controls when it opens the
# device. Run this AFTER your teleop/record/eval script has opened the camera.
#
# Usage:
#   bash shell/lock_camera.sh                 # lock using current auto-chosen values
#   EXPOSURE=156 WB=4600 GAIN=10 bash shell/lock_camera.sh
#   DEVICE=/dev/video0 bash shell/lock_camera.sh   # override the default node
#   bash shell/lock_camera.sh --list          # just print available controls and exit

set -euo pipefail

# The teleop/record/eval camera is /dev/video2 on this machine, so always act on
# it by default. An explicit DEVICE override still wins.
DEVICE="${DEVICE:-/dev/video2}"

if ! command -v v4l2-ctl >/dev/null 2>&1; then
    echo "v4l2-ctl not found. Install it with: sudo apt-get install -y v4l-utils" >&2
    exit 1
fi

if [ "${1:-}" = "--list" ]; then
    v4l2-ctl -d "$DEVICE" --list-ctrls
    exit 0
fi

# Names returned by --list-ctrls (one per line, e.g. "exposure_time_absolute").
CTRLS="$(v4l2-ctl -d "$DEVICE" --list-ctrls | sed -E 's/^[[:space:]]*([a-z_]+).*/\1/' | grep -E '^[a-z_]+$' || true)"

has_ctrl() { echo "$CTRLS" | grep -qx "$1"; }

# Read the current value of a control so we can keep what auto-exposure settled on
# unless the user overrode it.
current() { v4l2-ctl -d "$DEVICE" --get-ctrl "$1" 2>/dev/null | sed -E 's/.*:[[:space:]]*//'; }

set_ctrl() {
    local name="$1" value="$2"
    if v4l2-ctl -d "$DEVICE" --set-ctrl="${name}=${value}" 2>/dev/null; then
        echo "  set ${name}=${value}"
    else
        echo "  WARN: failed to set ${name}=${value}" >&2
    fi
}

echo "Locking controls on ${DEVICE}"

# --- Exposure -------------------------------------------------------------
# UVC: auto_exposure uses 1=manual, 3=aperture-priority(auto).
#      legacy exposure_auto uses 1=manual, 3=auto.
if has_ctrl auto_exposure; then
    AUTO_EXP_CTRL=auto_exposure; EXP_CTRL=exposure_time_absolute
elif has_ctrl exposure_auto; then
    AUTO_EXP_CTRL=exposure_auto; EXP_CTRL=exposure_absolute
else
    AUTO_EXP_CTRL=""; EXP_CTRL=""
fi

if [ -n "$AUTO_EXP_CTRL" ]; then
    EXP_VALUE="${EXPOSURE:-$(current "$EXP_CTRL")}"
    set_ctrl "$AUTO_EXP_CTRL" 1            # 1 = manual mode
    [ -n "$EXP_VALUE" ] && set_ctrl "$EXP_CTRL" "$EXP_VALUE"
else
    echo "  (no exposure control found)"
fi

# --- White balance --------------------------------------------------------
if has_ctrl white_balance_automatic; then
    AUTO_WB_CTRL=white_balance_automatic
elif has_ctrl white_balance_temperature_auto; then
    AUTO_WB_CTRL=white_balance_temperature_auto
else
    AUTO_WB_CTRL=""
fi

if [ -n "$AUTO_WB_CTRL" ]; then
    WB_VALUE="${WB:-$(current white_balance_temperature)}"
    set_ctrl "$AUTO_WB_CTRL" 0             # 0 = manual
    [ -n "$WB_VALUE" ] && has_ctrl white_balance_temperature && set_ctrl white_balance_temperature "$WB_VALUE"
else
    echo "  (no white-balance control found)"
fi

# --- Gain -----------------------------------------------------------------
if has_ctrl gain; then
    GAIN_VALUE="${GAIN:-$(current gain)}"
    [ -n "$GAIN_VALUE" ] && set_ctrl gain "$GAIN_VALUE"
fi

echo "Done. Verify with: v4l2-ctl -d ${DEVICE} --list-ctrls"

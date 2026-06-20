#!/usr/bin/env bash
# Lock camera exposure / gain to constant values so brightness
# does not drift between recording and eval.
#
# IMPORTANT: OpenCV (cv2.VideoCapture) resets these controls when it opens the
# device. Run this AFTER your teleop/record/eval script has opened the camera.
#
# Usage:
#   bash shell/lock_camera.sh                      # lock all available /dev/video* cameras
#   EXPOSURE=156 GAIN=10 bash shell/lock_camera.sh
#   DEVICE=/dev/video0 bash shell/lock_camera.sh   # lock one camera only
#   bash shell/lock_camera.sh --list               # print available controls and exit

set -euo pipefail

if ! command -v v4l2-ctl >/dev/null 2>&1; then
    echo "v4l2-ctl not found. Install it with: sudo apt-get install -y v4l-utils" >&2
    exit 1
fi

devices() {
    if [ -n "${DEVICE:-}" ]; then
        echo "$DEVICE"
    else
        v4l2-ctl --list-devices \
            | grep -Eo '/dev/video[0-9]+' \
            | while read -r device; do
                v4l2-ctl -d "$device" --list-ctrls >/dev/null 2>&1 && echo "$device"
            done
    fi
}

lock_device() {
    local device="$1"

    # Names returned by --list-ctrls (one per line, e.g. "exposure_time_absolute").
    local ctrls
    ctrls="$(v4l2-ctl -d "$device" --list-ctrls | sed -E 's/^[[:space:]]*([a-z_]+).*/\1/' | grep -E '^[a-z_]+$' || true)"

    has_ctrl() { echo "$ctrls" | grep -qx "$1"; }

    # Read the current value of a control so we can keep what auto-exposure settled on
    # unless the user overrode it.
    current() { v4l2-ctl -d "$device" --get-ctrl "$1" 2>/dev/null | sed -E 's/.*:[[:space:]]*//'; }

    set_ctrl() {
        local name="$1" value="$2"
        if v4l2-ctl -d "$device" --set-ctrl="${name}=${value}" 2>/dev/null; then
            echo "  set ${name}=${value}"
        else
            echo "  WARN: failed to set ${name}=${value}" >&2
        fi
    }

    echo "Locking controls on ${device}"

    # --- Exposure -------------------------------------------------------------
    # UVC: auto_exposure uses 1=manual, 3=aperture-priority(auto).
    #      legacy exposure_auto uses 1=manual, 3=auto.
    if has_ctrl auto_exposure; then
        auto_exp_ctrl=auto_exposure; exp_ctrl=exposure_time_absolute
    elif has_ctrl exposure_auto; then
        auto_exp_ctrl=exposure_auto; exp_ctrl=exposure_absolute
    else
        auto_exp_ctrl=""; exp_ctrl=""
    fi

    if [ -n "$auto_exp_ctrl" ]; then
        exp_value="${EXPOSURE:-$(current "$exp_ctrl")}"
        set_ctrl "$auto_exp_ctrl" 1            # 1 = manual mode
        [ -n "$exp_value" ] && set_ctrl "$exp_ctrl" "$exp_value"
    else
        echo "  (no exposure control found)"
    fi

    # --- Gain -----------------------------------------------------------------
    if has_ctrl gain; then
        gain_value="${GAIN:-$(current gain)}"
        [ -n "$gain_value" ] && set_ctrl gain "$gain_value"
    fi
}

mapfile -t DEVICES < <(devices)

if [ "${1:-}" = "--list" ]; then
    for device in "${DEVICES[@]}"; do
        echo "${device}:"
        v4l2-ctl -d "$device" --list-ctrls
    done
    exit 0
fi

if [ "${#DEVICES[@]}" -eq 0 ]; then
    echo "No controllable /dev/video* devices found." >&2
    exit 1
fi

for device in "${DEVICES[@]}"; do
    lock_device "$device"
done

echo "Done. Verify with: bash shell/lock_camera.sh --list"

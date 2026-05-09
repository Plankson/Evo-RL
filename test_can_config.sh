#!/usr/bin/env bash
set -euo pipefail

# Configure only the follower-arm CAN interfaces.
# This is a narrow test variant of can_config.sh:
#   1-13:1.0 -> can_left  @ 1000000
#   1-12:1.0 -> can_right @ 1000000
#
# Extra CAN interfaces, such as leader/master arms or can0, are ignored.

declare -A USB_PORTS
USB_PORTS["1-13:1.0"]="can_left:1000000"
USB_PORTS["1-12:1.0"]="can_right:1000000"

declare -A PORT_IFACES

echo "Loading gs_usb module..."
sudo modprobe gs_usb

CURRENT_CAN_COUNT=$(ip link show type can | grep -c "link/can" || true)
if [ "${CURRENT_CAN_COUNT}" -eq 0 ]; then
  echo "Error: no CAN interfaces detected."
  exit 1
fi

echo "Detected ${CURRENT_CAN_COUNT} CAN interface(s). Only follower arms will be configured."

configure_interface() {
  local iface="$1"
  local target_name="$2"
  local target_bitrate="$3"

  local current_bitrate
  current_bitrate="$(ip -details link show "${iface}" | grep -oP 'bitrate \K\d+' || true)"

  if [ "${iface}" != "${target_name}" ] && ip link show "${target_name}" >/dev/null 2>&1; then
    echo "Error: target interface ${target_name} already exists while configuring ${iface}."
    exit 1
  fi

  echo "Configuring ${iface} as ${target_name} @ ${target_bitrate}"
  if [ "${current_bitrate}" != "${target_bitrate}" ]; then
    sudo ip link set "${iface}" down
    sudo ip link set "${iface}" type can bitrate "${target_bitrate}"
  else
    sudo ip link set "${iface}" down
  fi

  if [ "${iface}" != "${target_name}" ]; then
    sudo ip link set "${iface}" name "${target_name}"
  fi

  sudo ip link set "${target_name}" up
  ip -details link show "${target_name}" | sed -n '1,3p'
}

for iface in $(ip -br link show type can | awk '{print $1}'); do
  BUS_INFO="$(sudo ethtool -i "${iface}" 2>/dev/null | awk '/bus-info/ {print $2}' || true)"

  if [ -z "${BUS_INFO}" ]; then
    echo "Warning: unable to read bus-info for ${iface}; skipping."
    continue
  fi

  echo "Interface ${iface} is on USB port ${BUS_INFO}"

  if [ -z "${USB_PORTS[${BUS_INFO}]+set}" ]; then
    echo "Skipping ${iface}: not a follower-arm CAN port."
    continue
  fi

  PORT_IFACES["${BUS_INFO}"]="${iface}"
done

for bus_info in "${!USB_PORTS[@]}"; do
  if [ -z "${PORT_IFACES[${bus_info}]+set}" ]; then
    echo "Error: follower-arm CAN port ${bus_info} was not found."
    exit 1
  fi

  iface="${PORT_IFACES[${bus_info}]}"
  IFS=':' read -r TARGET_NAME TARGET_BITRATE <<< "${USB_PORTS[${bus_info}]}"
  configure_interface "${iface}" "${TARGET_NAME}" "${TARGET_BITRATE}"
done

echo "Follower CAN interfaces are configured and active: can_left, can_right"

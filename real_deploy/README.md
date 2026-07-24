# Franka OpenPI Deployment

Minimal real-robot client for running an OpenPI policy on a Franka FR3 with two Intel RealSense cameras.

## Environment

Use Python 3.10 on the robot workstation:

```bash
conda create -n robosnap-deploy python=3.10 -y
conda activate robosnap-deploy
pip install -r requirements.txt
```

Install the `panda-python` wheel that matches the robot firmware and libfranka version. FR3 compatibility depends on this wheel; use the release specified by the robot workstation configuration.

Install the OpenPI client from the same OpenPI revision used by the policy server:

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
pip install -e openpi/packages/openpi-client
```

Alternatively, point the client at an existing source checkout:

```bash
export OPENPI_CLIENT_ROOT=/path/to/openpi/packages/openpi-client/src
```

The workstation also needs Intel RealSense SDK access and permission to use the Franka network interface and the gripper serial device.

## Franka and ROS 2

This client controls Franka directly through `panda-python` and [`libfranka`](https://github.com/frankarobotics/libfranka); ROS 2 is optional. Keep the robot system version, `libfranka` version, and `panda-python` wheel compatible. The wheel must also match the Python ABI and platform.

If ROS 2 is added, select a [ROS 2](https://github.com/ros2) distribution supported by the host Ubuntu release, then use compatible `franka_ros2` and `libfranka` versions for that distribution and robot system version.

## Run

Start the OpenPI websocket policy server, then run:

```bash
python playground/pipeline/inference_franka_openpi_sdk.py \
  --openpi-host <policy-server-ip> \
  --robot-hostname <franka-ip> \
  --exterior-serial <camera-serial> \
  --wrist-serial <camera-serial> \
  --prompt "<task>" \
  --auto-start
```

Pass `--exterior-usb-port` and `--wrist-usb-port` to enforce physical camera-port assignments. Use `--gripper-type panda_hand` or `--gripper-type none` when Robotiq is not used. Run `python playground/pipeline/inference_franka_openpi_sdk.py --help` for all controls.

# Boot process

The image builder installs one matched Ubuntu kernel and stages its modules in the final root. It rebuilds the initramfs with Casper, checks for the Casper script and disables resume from the build host’s devices. GRUB identifies the live filesystem by `boot=casper live-media-path=/casper` and disables swap, hibernation resume and persistence. Casper mounts the compressed root and calls the configured Rust init.

PID 1 checks its real process ID and reports the root filesystem type. It brings up the loopback interface, requires the firewall before external network setup, and starts the service with UID/GID 61000. The service is supervised using a loopback health check and a bounded restart policy. PID 1 reaps orphan children and terminates the service process group before powering off. The boot-validation GRUB entry also exercises one deliberate service restart in a diskless guest.

The stock image includes Ubuntu’s userspace tools. For an operator-maintained GPU image, obtain NVIDIA drivers and firmware from the official source, match the installed kernel ABI, and use matching Toolkit components. Confirm which individual binaries may be redistributed before bundling any NVIDIA packages. [NVIDIA CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

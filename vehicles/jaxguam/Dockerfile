#FROM ros:noetic
FROM osrf/ros:noetic-desktop-focal

# Set environment variables for NVIDIA and non-interactive apt
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/bin:${PATH}"


# Install tools and fix ROS GPG key issue
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 lsb-release ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Add the new ROS key properly
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add ROS repo (signed with new key)
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" \
    > /etc/apt/sources.list.d/ros1.list

# Update and install essential tools in one layer to minimize image size
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    curl \
    eog \
    x11-apps \
    ros-noetic-tf \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 and set it as the default
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.10 python3.10-distutils python3.10-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies for JAX and other tools
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install extensisq jax jaxlib pandas

# Copy the jax_guam source and install it
COPY jax_guam /jax_guam
WORKDIR /jax_guam
RUN python3.10 -m pip install -e .

# Set up ROS environment
ENV ROS_DISTRO=noetic
ENV ROS_ROOT=/opt/ros/noetic
ENV ROS_PACKAGE_PATH=$ROS_ROOT/share
ENV PATH=$ROS_ROOT/bin:$PATH
ENV PYTHONPATH=$ROS_ROOT/lib/python3/dist-packages:$PYTHONPATH
ENV LD_LIBRARY_PATH=$ROS_ROOT/lib:$LD_LIBRARY_PATH

RUN mkdir -p /catkin_ws/src


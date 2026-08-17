ARG ROS_DISTRO=humble
FROM osrf/ros:${ROS_DISTRO}-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /ws


# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-pip \
    x11-apps \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    ros-${ROS_DISTRO}-joint-state-publisher-gui \
    ros-${ROS_DISTRO}-xacro \
    vim \
    tmux \
    && rm -rf /var/lib/apt/lists/*


COPY ./requirements /tmp/requirements

# Install pip requirements
RUN pip install -r /tmp/requirements/requirements.txt

# Copy entrypoint
COPY ./docker/entrypoint.sh /tmp/entrypoint.sh
RUN chmod +x /tmp/entrypoint.sh

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash" >> /root/.bashrc

ENTRYPOINT ["/tmp/entrypoint.sh"]
CMD ["bash"]




# Dockerfile for fedora-demos
FROM docker.io/debian:bookworm

# Install necessary packages for OpenGL
RUN apt update -y
RUN apt install -y \
    python3 \
    python3-dev \
    python3-pip \
    libglfw3 \
    python3-opengl \
    python3-pyglfw \
    gcc \
    g++ \
    mesa-va-drivers \
    mesa-vdpau-drivers \
    texlive-latex-base texlive-latex-recommended texlive-science 


# Set a default command to keep the container running for interaction
CMD ["bash"]

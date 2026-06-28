FROM docker.io/debian:trixie

ARG USE_JUPYTER=1
ARG USE_SPYDER=1


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
    texlive-latex-base texlive-latex-recommended texlive-science  \
    emacs \
    tmux \
    which


RUN apt install -y  python3-venv

RUN echo FOO && python3 -m venv /venv --system-site-packages  && \
    . /venv/bin/activate && \
    python -m pip install --upgrade pip setuptools



COPY entrypoint/dotfiles/ /root/

RUN echo "/usr/local/bin/jupyter.sh # JupyterLab on http://127.0.0.1:8888/lab" >> ~/.bash_history

RUN apt install -y  git


RUN export VIRTUAL_ENV_DISABLE_PROMPT=1 && \
       . /venv/bin/activate && \
        cd ~/ && \
        git clone https://github.com/billsix/pyimgui.git && \
        cd pyimgui && \
        git submodule init && git submodule update && \
        python3 -m pip install . --root-user-action=ignore

RUN export VIRTUAL_ENV_DISABLE_PROMPT=1 && \
       . /venv/bin/activate && \
        python3 -m pip install ty --root-user-action=ignore


RUN apt install -y jupyter \
             jupyterlab \
             python3-jupytext \
             fonts-mathjax \
             python3-jupyter-server-mathjax






ENTRYPOINT ["/entrypoint.sh"]

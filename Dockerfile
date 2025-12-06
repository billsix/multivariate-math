FROM registry.fedoraproject.org/fedora:43

ARG USE_JUPYTER=1
ARG USE_SPYDER=1


RUN --mount=type=cache,target=/var/cache/libdnf5 \
    --mount=type=cache,target=/var/lib/dnf \
    echo "keepcache=True" >> /etc/dnf/dnf.conf && \
    dnf upgrade -y

RUN --mount=type=cache,target=/var/cache/libdnf5 \
    --mount=type=cache,target=/var/lib/dnf \
    dnf install -y \
                   emacs \
                   glfw \
                   npm \
                   python3 \
                   python3-pip \
                   python3-setuptools \
                   python3-sympy \
                   python3-pytest \
                   python3-wheel \
                   python3-devel \
                   python3-glfw \
		   python3-numpy \
                   python3-pip \
                   python3-pyopengl \
                   python3-pytest \
		   python3-pytest-lsp \
                   python3-sympy \
                   ruff \
                   emacs-gtk+x11 \
                   emacs-pgtk \
                   tmux \
                   which

RUN --mount=type=cache,target=/var/cache/libdnf5 \
    --mount=type=cache,target=/var/lib/dnf \
     export VIRTUAL_ENV_DISABLE_PROMPT=1 && \
     python3 -m venv /venv --system-site-packages  && \
     source /venv/bin/activate && \
     python -m pip install --upgrade pip setuptools && \
     # install pyright for lsp \
     npm install -g pyright


COPY entrypoint/dotfiles/ /root/

RUN emacs --batch --load /root/.emacs.d/install-melpa-packages.el && \
    echo "alias ls='ls --color=auto'" >> ~/.bashrc


RUN source /venv/bin/activate && \
    python -m pip install ty

RUN --mount=type=cache,target=/var/cache/libdnf5 \
    --mount=type=cache,target=/var/lib/dnf \
    if [ "$USE_JUPYTER" = "1" ]; then \
       dnf install -y \
                   jupyter \
                   jupyterlab  \
                   jupytext \
                   mathjax \
                   mathjax-main-fonts \
                   mathjax-math-fonts \
                   python3-jupyterlab-jupytext \
        	   python3-jupyter-lsp  ; \
    fi;

RUN --mount=type=cache,target=/var/cache/libdnf5 \
    --mount=type=cache,target=/var/lib/dnf \
    if [ "$USE_SPYDER" = "1" ]; then \
      dnf install -y   \
                   mesa-dri-drivers  \
                   mesa-libGLU-devel && \
      dnf install -y python3-spyder && \
      mkdir -p ~/.config/spyder-py3/config && \
      echo "[editor]" >> ~/.config/spyder-py3/config/spyder.ini && \
      echo "font/family = Source Code Pro" >> ~/.config/spyder-py3/config/spyder.ini && \
      echo "font/size = 24" >> ~/.config/spyder-py3/config/spyder.ini && \
      echo "[file_explorer]" >> ~/.config/spyder-py3/config/spyder.ini && \
      echo "visible = False" >> ~/.config/spyder-py3/config/spyder.ini && \
      echo "[tours]" >> ~/.config/spyder-py3/config/spyder.ini && \
      echo "show_tour_message = False" >> ~/.config/spyder-py3/config/spyder.ini && \
      echo "[appearance]" >> ~/.config/spyder-py3/config/spyder.ini && \
      echo "font/family = Adwaita Mono" >> ~/.config/spyder-py3/config/spyder.ini && \
      echo "font/size = 18" >> ~/.config/spyder-py3/config/spyder.ini; \
    fi ;

RUN echo "/usr/local/bin/jupyter.sh" >> ~/.bash_history


RUN --mount=type=cache,target=/var/cache/libdnf5 \
    --mount=type=cache,target=/var/lib/dnf \
    dnf install -y gcc \
                   gcc-c++

RUN --mount=type=cache,target=/var/cache/libdnf5 \
    --mount=type=cache,target=/var/lib/dnf \
       dnf install -y \
                   autoconf \
                   automake \
                   g++ \
        	   gcc \
                   python3-devel && \
       export VIRTUAL_ENV_DISABLE_PROMPT=1 && \
       source /venv/bin/activate && \
        cd ~/ && \
        git clone https://github.com/billsix/pyimgui.git && \
        cd pyimgui && \
        git submodule init && git submodule update && \
        python3 -m pip install . --root-user-action=ignore



ENTRYPOINT ["/entrypoint.sh"]

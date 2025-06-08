FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget curl ca-certificates vim pkg-config \
        python3.11 python3.11-venv python3-pip \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev libffi-dev liblzma-dev tk-dev uuid-dev \
        openssh-server \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash jaxuser

ENV VENV_DIR=/opt/venv
RUN python3.11 -m venv $VENV_DIR
ENV PATH="$VENV_DIR/bin:$PATH" \
    VIRTUAL_ENV=$VENV_DIR

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      jax[cuda12] \
      flax \
      transformers \
      datasets \
      optax

RUN mkdir /var/run/sshd

COPY id_rsa.pub /home/jaxuser/.ssh/authorized_keys
RUN chown -R jaxuser:jaxuser /home/jaxuser/.ssh && \
    chmod 700 /home/jaxuser/.ssh && \
    chmod 600 /home/jaxuser/.ssh/authorized_keys

RUN sed -i 's/^
    sed -i 's/^UsePAM yes/UsePAM no/' /etc/ssh/sshd_config

WORKDIR /workspace/app
RUN mkdir -p /workspace/app && chown jaxuser:jaxuser /workspace/app

USER jaxuser

EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]

FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget curl ca-certificates vim pkg-config \
        python3.11 python3.11-venv python3-pip \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev libffi-dev liblzma-dev tk-dev uuid-dev \
        openssh-server \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash jaxuser

ENV VENV_DIR=/opt/venv
RUN python3.11 -m venv $VENV_DIR
ENV PATH="$VENV_DIR/bin:$PATH" \
    VIRTUAL_ENV=$VENV_DIR

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      jax[cuda12] \
      flax \
      transformers \
      datasets \
      optax

RUN mkdir /var/run/sshd

COPY id_rsa.pub /home/jaxuser/.ssh/authorized_keys
RUN chown -R jaxuser:jaxuser /home/jaxuser/.ssh && \
    chmod 700 /home/jaxuser/.ssh && \
    chmod 600 /home/jaxuser/.ssh/authorized_keys

RUN sed -i 's/^
    sed -i 's/^UsePAM yes/UsePAM no/' /etc/ssh/sshd_config

WORKDIR /workspace/app
RUN mkdir -p /workspace/app && chown jaxuser:jaxuser /workspace/app

USER jaxuser

EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]

FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget curl ca-certificates vim pkg-config \
        python3.11 python3.11-venv python3-pip \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev libffi-dev liblzma-dev tk-dev uuid-dev \
        openssh-server \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash jaxuser

ENV VENV_DIR=/opt/venv
RUN python3.11 -m venv $VENV_DIR
ENV PATH="$VENV_DIR/bin:$PATH" \
    VIRTUAL_ENV=$VENV_DIR

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      jax[cuda12] \
      flax \
      transformers \
      datasets \
      optax

RUN mkdir /var/run/sshd

COPY id_rsa.pub /home/jaxuser/.ssh/authorized_keys
RUN chown -R jaxuser:jaxuser /home/jaxuser/.ssh && \
    chmod 700 /home/jaxuser/.ssh && \
    chmod 600 /home/jaxuser/.ssh/authorized_keys

RUN sed -i 's/^
    sed -i 's/^UsePAM yes/UsePAM no/' /etc/ssh/sshd_config

WORKDIR /workspace/app
RUN mkdir -p /workspace/app && chown jaxuser:jaxuser /workspace/app

USER jaxuser

EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]

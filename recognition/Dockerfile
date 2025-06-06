# Usar una imagen base ligera de Ubuntu 20.04
FROM ubuntu:20.04

# Establecer variables de entorno para evitar interacciones durante la construcción
ENV DEBIAN_FRONTEND=noninteractive

# Actualizar paquetes e instalar dependencias del sistema
RUN apt-get update && \
    apt-get install -y git zip wget libgl1 libglib2.0-0 vim tmux build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Instalar Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Configurar Conda en el PATH
ENV PATH=/opt/conda/bin:$PATH

# Crear y activar un entorno de Conda llamado 'myenv'
RUN conda create -n myenv python=3.10 -y

# Establecer el entorno como predeterminado
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Copiar el archivo de dependencias para instalar paquetes
COPY requirements.txt /tmp/requirements.txt

# Instalar las librerías de Python requeridas desde el requirements.txt
RUN pip install -r /tmp/requirements.txt

# Set the working directory to /app
WORKDIR /app

# Configurar el entorno por defecto en bash
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate myenv" >> ~/.bashrc

# Comando por defecto
CMD ["bash"]

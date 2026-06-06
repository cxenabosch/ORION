# Base OS
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, Python, and FSL via neurodebian
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    neurodebian \
    fsl-core \
    && rm -rf /var/lib/apt/lists/*

# Configure FSL environment variables
ENV FSLDIR=/usr/share/fsl/5.0
ENV PATH=${FSLDIR}/bin:${PATH}
ENV FSLOUTPUTTYPE=NIFTI_GZ

# Create data mount points and application directory
RUN mkdir /input /output /weights /app

# Copy application files and template resources
COPY requirements.txt /app/
COPY scripts/ /app/scripts/
COPY resources/ /app/resources/

# Install Python requirements
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Set locale
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Define working directory and entry point
WORKDIR /app
CMD ["python3", "scripts/main.py"]
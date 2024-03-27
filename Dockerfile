# Use a Debian-based image as Python 3.10 isn't directly available with all dependencies for Octave
FROM debian:bullseye-slim

# Install Python 3.10
RUN apt-get update && apt-get install -y python3.10 python3-pip && \
    ln -s /usr/bin/python3.10 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

# Install Octave and the octave-image package
RUN apt-get update && \
    apt-get install -y octave octave-image && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --verbose


# Make port 80 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV NAME World

# Default command
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0" , "--port=5000"]

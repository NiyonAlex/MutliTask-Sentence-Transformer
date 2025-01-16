# Use Python 3.10-slim as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt to the container
COPY requirements.txt /app/requirements.txt

# Install the Python dependen
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy your project’s source code to the container
COPY src /app/src

# Set the default command to run Task 4, but this can be changed
CMD ["python", "/app/src/task2_multitask_learning_model.py"]
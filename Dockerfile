# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# ---- UPDATE THIS SECTION ----
# Install system dependencies required by RDKit for drawing
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*
# -------------------------

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Command to run the app
CMD ["streamlit", "run", "app.py"]
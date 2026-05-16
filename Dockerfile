# Use the official lightweight Python 3.12 image
FROM python:3.12-slim

# Prevent Python from writing pyc files and keep stdout/stderr unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

# Set the working directory
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .

# CRITICAL FOR PYTHON 3.12: Explicitly install setuptools and wheel first 
# to replace the deprecated distutils before installing the rest of the RAG stack.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Make the run script executable (if you choose to use it)
RUN chmod +x bin/run

# Expose the port FastAPI will run on
# Hugging Face requires port 7860
EXPOSE 7860

# OPTION A: Run directly via Uvicorn (No bin/run required)
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "4"]

# OPTION B: Run via the entrypoint script (Recommended)
CMD ["./bin/run"]
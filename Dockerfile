# Use the official Python lightweight image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first (for caching)
COPY requirements.txt .

# Install dependencies (and fix potential build errors)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Healthcheck to keep the container running on cloud
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
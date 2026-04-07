FROM python:3.11-slim

WORKDIR /app

# Upgrade pip and setuptools first
RUN pip install --upgrade pip setuptools wheel

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source
COPY . .

# Expose the port HF Spaces expects
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
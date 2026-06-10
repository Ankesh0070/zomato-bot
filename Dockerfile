FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app

# Expose Streamlit port
EXPOSE 8501

ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "src/main.py", "--server.port=8501", "--server.headless=true"]

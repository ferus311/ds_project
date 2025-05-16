FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsnappy-dev \
    libatlas-base-dev \
    gcc \
    g++ \
    make \
    curl \
    gnupg \
    git \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Install browser dependencies for Playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    xdg-utils \
    libu2f-udev \
    libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# Install Chrome for Selenium
# RUN curl -sS -o - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
#     && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
#     && apt-get -y update \
#     && apt-get -y install google-chrome-stable \
#     && rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    pyarrow \
    python-snappy \
    hdfs \
    fastavro \
    kafka-python \
    selenium \
    webdriver-manager \
    playwright \
    requests \
    beautifulsoup4 \
    lxml \
    pyhdfs \
    scrapy

# Install Playwright browsers
RUN pip install playwright && python -m playwright install chromium

RUN apt-get update && apt-get install -y \
    librdkafka-dev \
    build-essential \
    python3-dev

RUN pip install confluent-kafka

# Copy the crawler code
COPY . .

# Make the entrypoint script executable
# RUN chmod +x /app/entrypoint.sh

# Set the entrypoint
# ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (can be overridden from docker-compose)
# CMD ["python", "-m", "services.list_crawler.main"]

FROM python:3.10-slim

# Ensures that Python output to stdout/stderr is not buffered
ENV PYTHONUNBUFFERED=1

# Create user
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# Set working directory
WORKDIR /opt/app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model directory
RUN mkdir -p /opt/app/model

# Copy N2V_Vanilla package
COPY N2V_Vanilla/ ./N2V_Vanilla/

# Copy model checkpoint
COPY N2V_Vanilla/best_model_NafNet_Noise2Void.pth /opt/app/model/best_model_NafNet_Noise2Void.pth

# Copy inference script
COPY predict.py .

# Create input/output directories with proper permissions
RUN mkdir -p /input /output && chown -R user:user /opt/app /input /output

# Switch to user
USER user

# Set entrypoint
ENTRYPOINT ["python", "predict.py"]
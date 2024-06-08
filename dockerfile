
# FROM python:3.9.11

# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY app/ .

# EXPOSE 8080
# CMD ["python", "main.py"]


FROM python:3.9.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "ipykernel_launcher", "-f", "{connection_file}"]
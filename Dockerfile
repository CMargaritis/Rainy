FROM zauberzeug/nicegui:latest
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && python -m pip install --no-cache-dir -r requirements.txt
RUN python -c "import importlib; importlib.import_module('openmeteo_requests'); print('openmeteo_requests import OK')"
COPY . .
EXPOSE 8080
CMD ["python", "rainy.py"]
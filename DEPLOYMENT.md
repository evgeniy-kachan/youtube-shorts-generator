# 🚀 Deployment Guide

Подробное руководство по развертыванию YouTube Shorts Generator на сервере с GPU A4000.

## 📋 Предварительные требования

### Аппаратура
- **GPU**: NVIDIA A4000 (или аналог с 16GB+ VRAM)
- **CPU**: 8+ cores
- **RAM**: 32GB+
- **Storage**: 100GB+ SSD

### Операционная система
- Ubuntu 22.04 LTS (рекомендуется)
- Ubuntu 20.04 LTS
- Debian 11+

## 🔧 Пошаговая установка

### Шаг 1: Подготовка сервера

```bash
# Обновите систему
sudo apt update && sudo apt upgrade -y

# Установите основные инструменты
sudo apt install -y build-essential git curl wget
```

### Шаг 2: Установка NVIDIA драйверов и CUDA

```bash
# Проверьте текущие драйверы
nvidia-smi

# Если драйверов нет, установите
sudo apt install -y nvidia-driver-535
sudo reboot

# После перезагрузки проверьте
nvidia-smi

# Установка CUDA Toolkit 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Добавьте CUDA в PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Проверьте установку
nvcc --version
```

### Шаг 3: Установка Python 3.10+

```bash
# Ubuntu 22.04 уже имеет Python 3.10
python3 --version

# Если нужна установка
sudo apt install -y python3.10 python3.10-venv python3-pip

# Обновите pip
python3 -m pip install --upgrade pip
```

### Шаг 4: Установка FFmpeg

```bash
sudo apt install -y ffmpeg

# Проверьте установку
ffmpeg -version
```

### Шаг 5: Установка Node.js (для frontend)

```bash
# Установка Node.js 18.x
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Проверьте установку
node --version
npm --version
```

### Шаг 6: Клонирование и установка проекта

```bash
# Клонируйте репозиторий
cd /opt
sudo git clone https://github.com/yourusername/youtube-shorts-generator.git
cd youtube-shorts-generator

# Дайте права текущему пользователю
sudo chown -R $USER:$USER /opt/youtube-shorts-generator

# Запустите автоматическую установку
chmod +x install.sh
./install.sh
```

### Шаг 7: Настройка конфигурации

```bash
# Отредактируйте .env файл
nano .env
```

Рекомендуемые настройки для A4000:

```env
HOST=0.0.0.0
PORT=8000

OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b  # Или qwen2.5:7b

MAX_VIDEO_DURATION=7200
TEMP_DIR=./temp
OUTPUT_DIR=./output

CUDA_VISIBLE_DEVICES=0
```

### Шаг 8: Сборка Frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

## 🚀 Запуск сервиса

### Вариант 1: Простой запуск

```bash
./run.sh
```

### Вариант 2: Production с Systemd

Создайте systemd service файл:

```bash
sudo nano /etc/systemd/system/youtube-shorts.service
```

Содержимое:

```ini
[Unit]
Description=YouTube Shorts Generator
After=network.target

[Service]
Type=simple
User=your-username
Group=your-username
WorkingDirectory=/opt/youtube-shorts-generator
Environment="PATH=/opt/youtube-shorts-generator/venv/bin:/usr/local/cuda-11.8/bin:/usr/bin"
Environment="LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64"
Environment="PYTHONPATH=/opt/youtube-shorts-generator"
ExecStartPre=/bin/bash -c 'ollama serve > /dev/null 2>&1 &'
ExecStart=/opt/youtube-shorts-generator/venv/bin/python backend/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Активируйте сервис:

```bash
sudo systemctl daemon-reload
sudo systemctl enable youtube-shorts
sudo systemctl start youtube-shorts
sudo systemctl status youtube-shorts
```

Проверка логов:

```bash
sudo journalctl -u youtube-shorts -f
```

### Вариант 3: С использованием Gunicorn

```bash
# Установите gunicorn
source venv/bin/activate
pip install gunicorn

# Запустите
gunicorn backend.main:app \
    -w 1 \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300 \
    --access-logfile - \
    --error-logfile -
```

## 🌐 Настройка Nginx (опционально)

Для production рекомендуется использовать Nginx как reverse proxy:

```bash
sudo apt install -y nginx

# Создайте конфигурацию
sudo nano /etc/nginx/sites-available/youtube-shorts
```

Содержимое:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 500M;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts for long-running requests
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

Активируйте конфигурацию:

```bash
sudo ln -s /etc/nginx/sites-available/youtube-shorts /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 🔒 Настройка SSL (Let's Encrypt)

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
sudo systemctl reload nginx
```

## 🐳 Docker Deployment (альтернатива)

### Установка Docker и NVIDIA Container Toolkit

```bash
# Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Проверка
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Запуск с Docker Compose

```bash
# Сборка образа
docker-compose build

# Запуск
docker-compose up -d

# Проверка логов
docker-compose logs -f

# Остановка
docker-compose down
```

## 📊 Мониторинг

### GPU Monitoring

```bash
# Установите gpustat
pip install gpustat

# Мониторинг в реальном времени
watch -n 1 gpustat -cpu

# Или nvidia-smi
watch -n 1 nvidia-smi
```

### Logs

```bash
# Systemd service logs
sudo journalctl -u youtube-shorts -f

# Docker logs
docker-compose logs -f

# Manual run logs
tail -f logs/app.log
```

## 🔧 Troubleshooting

### CUDA не обнаружена

```bash
# Проверьте переменные окружения
echo $PATH
echo $LD_LIBRARY_PATH

# Добавьте в .bashrc
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

### Out of Memory

Уменьшите размер моделей в `backend/config.py`:

```python
WHISPER_MODEL = "medium"  # вместо large-v3
WHISPER_COMPUTE_TYPE = "int8"  # вместо float16
```

Используйте меньшую LLM:

```bash
ollama pull llama3.1:8b  # вместо 70b
```

### Ollama не запускается

```bash
# Переустановите Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Запустите вручную
ollama serve

# Проверьте порт
netstat -tlnp | grep 11434
```

### FFmpeg ошибки

```bash
# Переустановите FFmpeg
sudo apt remove ffmpeg
sudo apt install ffmpeg

# Или соберите из исходников для лучшей поддержки кодеков
```

## 🔄 Обновление

```bash
cd /opt/youtube-shorts-generator
git pull
source venv/bin/activate
pip install -r requirements.txt --upgrade
cd frontend && npm install && npm run build && cd ..
sudo systemctl restart youtube-shorts
```

## 💾 Backup

```bash
# Создайте скрипт бэкапа
cat > backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/youtube-shorts"
mkdir -p $BACKUP_DIR

# Backup configuration
tar -czf $BACKUP_DIR/config_$DATE.tar.gz .env backend/config.py

# Backup models (optional, they can be re-downloaded)
# tar -czf $BACKUP_DIR/models_$DATE.tar.gz ~/.cache/huggingface

echo "Backup completed: $BACKUP_DIR"
EOF

chmod +x backup.sh
```

## 📈 Performance Tuning

### Для максимальной производительности:

1. **Используйте SSD** для temp и output директорий
2. **Увеличьте swap** если RAM недостаточно:
   ```bash
   sudo fallocate -l 16G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

3. **Оптимизируйте модели**:
   - Используйте quantized версии (int8, int4)
   - Кэшируйте модели на SSD

4. **Настройте limits**:
   ```bash
   sudo nano /etc/security/limits.conf
   ```
   Добавьте:
   ```
   * soft nofile 65536
   * hard nofile 65536
   ```

## 🎯 Production Checklist

- [ ] CUDA и драйверы установлены
- [ ] Все зависимости установлены
- [ ] .env файл настроен
- [ ] Модели загружены
- [ ] Systemd service создан и активен
- [ ] Nginx настроен (если используется)
- [ ] SSL сертификат установлен
- [ ] Мониторинг настроен
- [ ] Backup скрипт создан
- [ ] Firewall настроен
- [ ] Logs rotation настроен

## 📞 Поддержка

При возникновении проблем:
1. Проверьте логи
2. Проверьте GPU статус
3. Проверьте доступность моделей
4. Создайте Issue в репозитории

---

Удачного развертывания! 🚀


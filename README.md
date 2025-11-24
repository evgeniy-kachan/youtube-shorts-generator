# 🎬 YouTube Shorts Generator

AI-powered сервис для автоматического создания вирусных Shorts/Reels из длинных YouTube видео.

## 🌟 Возможности

- **Автоматическая транскрипция** с использованием Whisper (large-v3)
- **AI-анализ** интересных моментов по 12 критериям с помощью DeepSeek (reasoner)
- **Автоматический перевод и адаптация** на русский язык (DeepSeek)
- **Генерация озвучки** на русском (Silero TTS)
- **Стильные субтитры** в стиле TikTok/Instagram с анимацией
- **Конвертация в вертикальный формат 9:16** (Reels/Shorts) с 3 методами:
  - 🌟 Размытый фон (рекомендуется) - видео по центру + blur
  - ✂️ Обрезка по центру - простая обрезка
  - 🤖 Умная обрезка - с детекцией объектов
- **Обработка видео** до 2 часов
- **Клипы** от 20 секунд до 3 минут

## 🏗️ Архитектура

### Backend
- **FastAPI** - современный асинхронный веб-фреймворк
- **faster-whisper** - транскрипция с GPU ускорением
- **DeepSeek API** - удалённая LLM для анализа/перевода/подготовки текста
- **Silero TTS** - генерация русской речи
- **FFmpeg** - обработка видео и субтитров

### Frontend
- **React** - UI фреймворк
- **TailwindCSS** - современные стили
- **Axios** - HTTP клиент

## 📋 Требования

### Сервер
- **GPU**: NVIDIA A4000 или аналог (минимум 16GB VRAM)
- **RAM**: минимум 32GB
- **Диск**: минимум 100GB свободного места
- **OS**: Ubuntu 20.04+ или аналог

### ПО
- Python 3.10+
- Node.js 18+
- CUDA 11.8+
- FFmpeg
- DeepSeek API key (https://api.deepseek.com/)

## 🚀 Установка

### 1. Установка системных зависимостей

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.10 python3-pip python3-venv
sudo apt install -y ffmpeg
sudo apt install -y nodejs npm

# Установка CUDA (если еще не установлена)
# Следуйте инструкциям: https://developer.nvidia.com/cuda-downloads
```

### 2. DeepSeek API

1. Создайте ключ в [DeepSeek Platform](https://platform.deepseek.com/api_keys).
2. Убедитесь, что у ключа есть доступ к модели `deepseek-reasoner`.
3. Сохраните ключ как `DEEPSEEK_API_KEY` в `.env` (см. ниже). Ключ **нельзя** коммитить в репозиторий.

### 3. Клонирование и настройка проекта

```bash
# Клонируйте репозиторий (или используйте существующий)
cd /path/to/project_blog

# Создайте виртуальное окружение
python3 -m venv venv
source venv/bin/activate

# Установите Python зависимости
pip install --upgrade pip
pip install -r requirements.txt

# Скачайте Whisper модель (автоматически при первом запуске)
# Или предварительно:
python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cuda')"
```

### 4. Настройка конфигурации

```bash
# Создайте файл .env
cp .env.example .env

# Отредактируйте .env при необходимости
nano .env
```

Пример `.env`:
```env
HOST=0.0.0.0
PORT=8000

DEEPSEEK_API_KEY=sk-XXXX
DEEPSEEK_MODEL=deepseek-reasoner
DEEPSEEK_BASE_URL=https://api.deepseek.com

WHISPER_MODEL=large-v3
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16

MAX_VIDEO_DURATION=7200
TEMP_DIR=./temp
OUTPUT_DIR=./output
CUDA_VISIBLE_DEVICES=0

TTS_ENABLE_MARKUP=true
TTS_MARKUP_MAX_TOKENS=200
```

### 5. Установка Frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

## 🎯 Запуск

### Вариант 1: Разработка (Dev Mode)

```bash
# Терминал 1: Backend
source venv/bin/activate
python backend/main.py

# Терминал 2: Frontend (в режиме разработки)
cd frontend
npm start
```

Frontend будет доступен на `http://localhost:3000`  
Backend API на `http://localhost:8000`

### Вариант 2: Production

```bash
# 1. Соберите frontend
cd frontend
npm run build
cd ..

# 2. Запустите backend (он будет отдавать и frontend)
source venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1

# Или с помощью gunicorn
gunicorn backend.main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

Откройте `http://your-server-ip:8000`

### Вариант 3: С использованием systemd (автозапуск)

Создайте файл `/etc/systemd/system/youtube-shorts.service`:

```ini
[Unit]
Description=YouTube Shorts Generator
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/project_blog
Environment="PATH=/path/to/project_blog/venv/bin"
ExecStart=/path/to/project_blog/venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Затем:
```bash
sudo systemctl daemon-reload
sudo systemctl enable youtube-shorts
sudo systemctl start youtube-shorts
sudo systemctl status youtube-shorts
```

## 📖 Использование

### Вариант 1: С локальными файлами (для России, где YouTube заблокирован)

#### 1. Загрузить видео на сервер:

```bash
scp your_video.mp4 root@SERVER_IP:/opt/youtube-shorts-generator/temp/my_video.mp4
```

#### 2. Через Swagger UI:

- Откройте `http://SERVER_IP:8000/docs`
- Найдите `POST /api/video/analyze-local`
- Нажмите "Try it out"
- Введите `filename`: `my_video.mp4`
- Нажмите "Execute"
- Скопируйте `task_id`
- Проверяйте статус через `GET /api/video/task/{task_id}`
- Когда completed - обрабатывайте сегменты через `POST /api/video/process`

#### 3. Или через API:

```bash
curl -X POST "http://SERVER_IP:8000/api/video/analyze-local?filename=my_video.mp4"
```

**Требования к видео:**
- Формат: MP4, AVI, MKV (любой что поддерживает FFmpeg)
- Длительность: до 2 часов
- Язык: английский (для транскрипции и анализа)
- Контент: интервью, подкасты, лекции (НЕ музыкальные клипы)

### Вариант 2: С YouTube (для стран где YouTube доступен)

1. Откройте веб-интерфейс в браузере
2. Вставьте URL YouTube видео (до 2 часов)
3. Дождитесь анализа (показывается прогресс)
4. Выберите интересные сегменты из предложенных
5. Нажмите "Создать клипы"
6. Скачайте готовые клипы с русской озвучкой и субтитрами

### API

#### 1. Анализ видео

```bash
curl -X POST http://localhost:8000/api/video/analyze \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID"}'
```

Ответ:
```json
{
  "task_id": "uuid",
  "status": "pending",
  "message": "Analysis started"
}
```

#### 2. Проверка статуса

```bash
curl http://localhost:8000/api/video/task/TASK_ID
```

#### 3. Обработка сегментов

```bash
curl -X POST http://localhost:8000/api/video/process \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "VIDEO_ID",
    "segment_ids": ["segment_0", "segment_1"]
  }'
```

#### 4. Скачивание

```bash
curl -O http://localhost:8000/api/video/download/VIDEO_ID/SEGMENT_ID
```

## 🎨 Критерии анализа

Система анализирует видео по 12 критериям:

1. **Information Density** - плотность информации (идеи, выводы, факты)
2. **Emotional Intensity** - эмоциональные всплески
3. **Topic Transition** - смена темы
4. **Key Value** - ценные советы и takeaway
5. **Hook Potential** - захватывающие зацепки
6. **Tension** - конфликт и напряжение
7. **Story Moment** - истории и примеры
8. **Humor** - юмор и смех
9. **Cadence Shift** - изменение темпа речи
10. **Keyword Density** - плотность ключевых слов
11. **Multimodal Score** - комплексная оценка
12. **Audience Appeal** - привлекательность для аудитории

Итоговый score рассчитывается по формуле:
```
highlight_score = 0.4 * semantic_value 
                + 0.25 * emotional_intensity
                + 0.15 * hook_probability
                + 0.1 * keyword_density
                + 0.1 * story_probability
```

## 🔧 Настройка моделей

### Изменение LLM модели

Отредактируйте `.env`:
```env
DEEPSEEK_MODEL=deepseek-reasoner   # или deepseek-chat для экономии
```

### Изменение Whisper модели

Отредактируйте `backend/config.py`:
```python
WHISPER_MODEL = "large-v3"  # или "large-v2", "medium"
```

## 📊 Производительность

На сервере с NVIDIA A4000:

| Этап | Время (для 1 часа видео) |
|------|------------------------|
| Транскрипция (Whisper large-v3) | ~10-15 мин |
| Анализ LLM (20 сегментов, DeepSeek) | ~5-10 мин |
| Перевод + адаптация (DeepSeek) | ~1-2 мин |
| Обработка 1 клипа (TTS + субтитры) | ~30-60 сек |

**Общее время**: 20-30 минут для полной обработки 1-часового видео

## 🐛 Troubleshooting

### Ошибка CUDA Out of Memory

Уменьшите размер моделей:
```python
# В config.py
WHISPER_MODEL = "medium"  # вместо large-v3
WHISPER_COMPUTE_TYPE = "int8"  # вместо float16
```

### FFmpeg ошибки

Убедитесь что установлена последняя версия:
```bash
ffmpeg -version
sudo apt install --reinstall ffmpeg
```

### Медленная обработка

1. Убедитесь что используется GPU:
```python
import torch
print(torch.cuda.is_available())  # должно быть True
```

2. Проверьте загрузку GPU:
```bash
nvidia-smi
```

## 📝 Лицензия

MIT License

## 🤝 Вклад

Приветствуются Pull Requests и Issues!

## 📞 Поддержка

Для вопросов и предложений создавайте Issues в репозитории.

---

**Powered by:**
- OpenAI Whisper (via faster-whisper)
- DeepSeek API (reasoner)
- Silero Models
- FFmpeg


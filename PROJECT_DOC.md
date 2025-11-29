# youtube-shorts-generator – Project Documentation

## 1. Overview

AI-пайплайн, который из исходного ролика (локального файла или YouTube) автоматически строит клипы формата Shorts/Reels:

1. Загружает или скачивает видео.
2. Транскрибирует Whisper large-v3 (CUDA).
3. Находит интересные отрезки через DeepSeek (reasoner).
4. Переводит и адаптирует на русский (DeepSeek).
5. Легко полирует текст для озвучки (LLM → естественная речь без форматирования).
6. Синтезирует озвучку Silero TTS (голос Eugene).
7. Генерирует вертикальное видео (FFmpeg + кастомный рендер).
8. Выдаёт клипы для скачивания через UI.

## 2. Repo / Services

```
backend/             FastAPI сервер (uvicorn)
├─ routers/video.py  API / пайплайн
├─ services          Whisper, DeepSeek, TTS, subtitles, рендер
├─ utils             хелперы
frontend/            React + Tailwind (Nginx)
docker-compose.yml   backend + frontend
Dockerfile           CUDA образ для backend
Dockerfile.nginx     сборка фронта
nginx.conf           прокси + статический фронт
PROJECT_DOC.md       текущий документ
```

### Backend key services

| Service              | File                                     | Notes                                    |
| -------------------- | ---------------------------------------- | ---------------------------------------- |
| TranscriptionService | `backend/services/transcription.py`      | Whisper large-v3, CUDA                   |
| HighlightAnalyzer    | `backend/services/highlight_analyzer.py` | DeepSeek reasoner (анализ сегментов)     |
| Translator           | `backend/services/translation.py`        | DeepSeek перевод                         |
| TextMarkupService    | `backend/services/text_markup.py`        | DeepSeek постобработка текста (plain)    |
| TTSService           | `backend/services/tts.py`                | Silero (`language=ru`, `speaker=eugene`) |
| VideoProcessor       | `backend/services/video_processor.py`    | FFmpeg пайплайн, субтитры CapCut         |

## 3. Configuration

Управляется через `backend/config.py` + `.env`:

| Variable                                                         | Description                   |
| ---------------------------------------------------------------- | ----------------------------- |
| `WHISPER_MODEL`, `WHISPER_DEVICE`                                | модель и устройство Whisper   |
| `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`, `DEEPSEEK_BASE_URL`        | доступ к DeepSeek             |
| `DEEPSEEK_TRANSLATION_CHUNK_SIZE`, `DEEPSEEK_TIMEOUT`            | параметры батчей/таймаут      |
| `SILERO_LANGUAGE`, `SILERO_SPEAKER`, `SILERO_MODEL_VERSION`      | Silero TTS                    |
| `TTS_ENABLE_MARKUP`, `TTS_MARKUP_MODEL`, `TTS_MARKUP_MAX_TOKENS` | LLM-полировка текста (без формат.) |
| `VERTICAL_CONVERSION_METHOD`                                     | default метод рендера         |
| `TEMP_DIR`, `OUTPUT_DIR`                                         | каталоги данных               |

## 4. Workflow

1. **Upload/Analyze**  
   UI: пользователь загружает MP4 → `/api/video/upload-video`.  
   Затем `/api/video/analyze-local?filename=...` запускает фоновый таск.

2. **Analysis Pipeline** (`_run_analysis_pipeline`):

   - Whisper -> сегменты (англ. текст + слова)
   - HighlightAnalyzer -> кандидатные окна (30–180 сек), оценка LLM
   - `_filter_overlapping_segments` -> убираем дубликаты
   - Translator -> `text_ru`
   - TextMarkupService -> `text_ru_tts` (любой текст остаётся без форматирования)
   - Результаты кешируются (`analysis_results_cache`), статус таска — `completed`

3. **Processing** (`/api/video/process`):

   - Тянем кешированный `video_id`
   - Для выбранных `segment_ids`:
     - TTS по `text_ru_tts`
     - VideoProcessor: вырезать отрезок, сделать вертикальный кадр (blur/center/smart), нарисовать субтитры (CapCut стиль)
   - Возвращаем `output_videos` (относительные пути)

4. **Download**
   - `/api/video/download/{video_id}/{segment_id}` → `FileResponse`

## 5. Frontend UX

- **VideoInput**: загрузка файла (только `.mp4`), прогресс.
- **ProgressBar**: отображение статуса анализа/рендера (данные из `/api/video/task/{task_id}`).
- **SegmentsList**:
  - Чекбоксы для сегментов, быстрые фильтры (выбрать/снять все).
  - Карточки выбора vertical метода.
  - Кнопка "Создать клипы" запускает `/api/video/process`.
- **DownloadList**:
  - Показывает список готовых клипов, кнопки `Скачать`/`Скачать все`.
  - Кнопка “Назад к выбору моментов” возвращает на экран сегментов (без повторного анализа).
  - “Обработать новое видео” — полный сброс.

## 6. Subtitles & TTS

- Субтитры: `VideoProcessor._create_stylized_subtitles` + CapCut стиль (`Montserrat`, fontsize 86, позиция `\pos(540,1250)`, мягкая тень).
- Анимация по словам: `\t` + fade, каждое слово появляется независимо, строка не подпрыгивает.
- TTS text prep:
  - DeepSeek аккуратно перефразирует русский текст для плавной озвучки.
  - Выход всегда без Markdown/`...`/подчёркиваний.
  - Silero голос — `eugene`. Можно сменить через env или `config.SILERO_SPEAKER`.

## 7. Commands

```bash
# локальная разработка
uvicorn backend.main:app --reload
cd frontend && npm start

# docker / gpu
docker compose up --build

# на сервере (из README/UPDATE_SERVER.md):
git pull origin main
docker compose build --no-cache backend frontend
docker compose up -d --force-recreate backend frontend
```

## 8. Future ideas

- UI для настройки субтитров (шрифт/позиция drag&drop).
- (legacy) Расширенный `smart_crop` (детекция лица/объекта с OpenCV).
- Личный пресет стилей (хранить в backend + UI-переключатель).
- Поддержка SSML или другого TTS для эмоциональной озвучки.
- Экспорт метаданных (json) для передачи в соцсети.
- Интеграция ElevenLabs TTS как облачной опции.



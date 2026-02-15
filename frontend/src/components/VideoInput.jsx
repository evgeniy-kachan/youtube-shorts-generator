import React, { useState, useEffect } from 'react';
import { getUploadedVideos, getCachedVideo } from '../services/api';

const VideoInput = ({ onSubmit, loading }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState('');
  const [analysisMode, setAnalysisMode] = useState('fast'); // 'fast' or 'deep'
  const [diarizer, setDiarizer] = useState('pyannote'); // 'pyannote' or 'nemo'
  const [cachedVideos, setCachedVideos] = useState([]);
  const [showCachedList, setShowCachedList] = useState(false);

  // Load cached videos on mount
  useEffect(() => {
    const loadCachedVideos = async () => {
      try {
        const data = await getUploadedVideos();
        setCachedVideos(data.videos || []);
      } catch (error) {
        console.error('Failed to load cached videos:', error);
      }
    };
    loadCachedVideos();
  }, []);

  const handleUseCachedVideo = async (videoId, filename) => {
    try {
      setError('');
      const data = await getCachedVideo(videoId);
      // Create a fake file object for compatibility
      const fakeFile = {
        name: data.filename,
        size: data.size_mb * 1024 * 1024,
        cached: true,
        cachedPath: data.path,
        cachedVideoId: videoId
      };
      onSubmit(fakeFile, analysisMode, diarizer);
    } catch (error) {
      setError('Ошибка при использовании кэшированного видео: ' + error.message);
    }
  };

  const handleFileChange = (event) => {
    setError('');
    const file = event.target.files?.[0];

    if (!file) {
      setSelectedFile(null);
      return;
    }

    const normalizedName = file.name?.toLowerCase() || '';
    const isSupported =
      file.type === 'video/mp4' ||
      file.type === 'video/quicktime' ||
      file.type === 'video/webm' ||
      normalizedName.endsWith('.mp4') ||
      normalizedName.endsWith('.mov') ||
      normalizedName.endsWith('.webm');

    if (!isSupported) {
      setError('Поддерживаются только файлы MP4, MOV или WEBM');
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setError('');

    if (!selectedFile) {
      setError('Пожалуйста, выберите видеофайл MP4, MOV или WEBM');
      return;
    }

    onSubmit(selectedFile, analysisMode, diarizer);
  };

  return (
    <div className="card max-w-3xl mx-auto">
      <div className="text-center mb-6">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">
          Загрузите видео — AI сделает клипы
        </h2>
        <p className="text-gray-600">
          Выберите видеофайл (MP4, MOV или WEBM) на своём компьютере, мы загрузим его на сервер и
          найдём самые интересные моменты автоматически.
        </p>
      </div>

      {/* Development: Cached videos list */}
      {cachedVideos.length > 0 && (
        <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-xl">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-blue-900">
              💾 Загруженные ранее (для разработки)
            </h3>
            <button
              type="button"
              onClick={() => setShowCachedList(!showCachedList)}
              className="text-xs text-blue-600 hover:text-blue-800"
            >
              {showCachedList ? 'Скрыть' : 'Показать'}
            </button>
          </div>
          {showCachedList && (
            <div className="space-y-2">
              {cachedVideos.map((video) => (
                <button
                  key={video.video_id}
                  type="button"
                  onClick={() => handleUseCachedVideo(video.video_id, video.filename)}
                  disabled={loading}
                  className="w-full text-left p-3 bg-white border border-blue-200 rounded-lg hover:bg-blue-50 transition disabled:opacity-50"
                >
                  <div className="font-medium text-gray-900">{video.filename}</div>
                  <div className="text-xs text-gray-500 mt-1">
                    {video.size_mb.toFixed(2)} MB • {new Date(video.uploaded_at).toLocaleString('ru-RU')}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label
            htmlFor="video-upload"
            className="block text-sm font-medium text-gray-700 mb-2"
          >
            Видеофайл (MP4 / MOV / WEBM)
          </label>
          <input
            id="video-upload"
            type="file"
            accept="video/mp4,video/quicktime,video/webm,.mov,.webm"
            onChange={handleFileChange}
            className="input-field"
            disabled={loading}
          />
          <p className="mt-2 text-sm text-gray-500">
            Поддерживаются файлы MP4, MOV или WEBM до 2 ГБ.
          </p>
          {selectedFile && !error && (
            <div className="mt-3 text-sm text-gray-700 bg-gray-50 border border-gray-200 rounded-lg p-3">
              <p className="font-medium">{selectedFile.name}</p>
              <p>Размер: {(selectedFile.size / (1024 * 1024)).toFixed(2)} МБ</p>
            </div>
          )}
          {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
        </div>

        {/* Analysis Mode Selector */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Режим анализа
          </label>
          <div className="grid grid-cols-2 gap-3">
            <button
              type="button"
              disabled={loading}
              onClick={() => setAnalysisMode('fast')}
              className={`p-4 border rounded-xl text-left transition ${
                analysisMode === 'fast'
                  ? 'border-purple-600 bg-purple-50 ring-1 ring-purple-300'
                  : 'border-gray-200 hover:border-purple-400'
              }`}
            >
              <div className="font-semibold text-gray-900">⚡ Быстрый</div>
              <div className="text-xs text-gray-500 mt-1">
                ~30 сек, хорошее качество
              </div>
            </button>
            <button
              type="button"
              disabled={loading}
              onClick={() => setAnalysisMode('deep')}
              className={`p-4 border rounded-xl text-left transition ${
                analysisMode === 'deep'
                  ? 'border-purple-600 bg-purple-50 ring-1 ring-purple-300'
                  : 'border-gray-200 hover:border-purple-400'
              }`}
            >
              <div className="font-semibold text-gray-900">🧠 Глубокий</div>
              <div className="text-xs text-gray-500 mt-1">
                ~3-4 мин, максимум качества
              </div>
            </button>
          </div>
        </div>

        {/* Speaker Diarization Selector */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Диаризация спикеров
          </label>
          <div className="grid grid-cols-2 gap-3">
            <button
              type="button"
              disabled={loading}
              onClick={() => setDiarizer('pyannote')}
              className={`p-4 border rounded-xl text-left transition ${
                diarizer === 'pyannote'
                  ? 'border-green-600 bg-green-50 ring-1 ring-green-300'
                  : 'border-gray-200 hover:border-green-400'
              }`}
            >
              <div className="font-semibold text-gray-900">🎙️ Pyannote</div>
              <div className="text-xs text-gray-500 mt-1">
                Быстрая, стабильная
              </div>
            </button>
            <button
              type="button"
              disabled={loading}
              onClick={() => setDiarizer('nemo')}
              className={`p-4 border rounded-xl text-left transition ${
                diarizer === 'nemo'
                  ? 'border-green-600 bg-green-50 ring-1 ring-green-300'
                  : 'border-gray-200 hover:border-green-400'
              }`}
            >
              <div className="font-semibold text-gray-900">🧠 NeMo MSDD</div>
              <div className="text-xs text-gray-500 mt-1">
                Точная, для похожих голосов
              </div>
            </button>
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className={`w-full btn-primary ${
            loading ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <svg
                className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
              Анализируем видео...
            </span>
          ) : (
            'Загрузить и проанализировать'
          )}
        </button>
      </form>

      <div className="mt-6 border-t pt-6">
        <h3 className="text-sm font-semibold text-gray-900 mb-3">
          Возможности:
        </h3>
        <ul className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {[
            'AI анализ по 12 критериям',
            'Автоматический перевод на русский',
            'Генерация озвучки',
            'Стильные субтитры как в TikTok',
            'Видео до 2 часов',
            'Клипы от 20 сек до 3 минут',
          ].map((feature, idx) => (
            <li key={idx} className="flex items-center text-sm text-gray-600">
              <svg
                className="h-5 w-5 text-green-500 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M5 13l4 4L19 7"
                />
              </svg>
              {feature}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default VideoInput;

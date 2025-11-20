import React, { useState } from 'react';

const VideoInput = ({ onSubmit, loading }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState('');

  const handleFileChange = (event) => {
    setError('');
    const file = event.target.files?.[0];

    if (!file) {
      setSelectedFile(null);
      return;
    }

    const isMp4 =
      file.type === 'video/mp4' ||
      file.name?.toLowerCase().endsWith('.mp4');

    if (!isMp4) {
      setError('Поддерживаются только файлы MP4');
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setError('');

    if (!selectedFile) {
      setError('Пожалуйста, выберите видеофайл MP4');
      return;
    }

    onSubmit(selectedFile);
  };

  return (
    <div className="card max-w-3xl mx-auto">
      <div className="text-center mb-6">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">
          Загрузите видео — AI сделает клипы
        </h2>
        <p className="text-gray-600">
          Выберите MP4-файл на своём компьютере, мы загрузим его на сервер и найдём самые интересные моменты автоматически.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="video-upload" className="block text-sm font-medium text-gray-700 mb-2">
            Видеофайл (MP4)
          </label>
          <input
            id="video-upload"
            type="file"
            accept="video/mp4"
            onChange={handleFileChange}
            className="input-field"
            disabled={loading}
          />
          <p className="mt-2 text-sm text-gray-500">
            Поддерживаются файлы MP4 до 2 ГБ.
          </p>
          {selectedFile && !error && (
            <div className="mt-3 text-sm text-gray-700 bg-gray-50 border border-gray-200 rounded-lg p-3">
              <p className="font-medium">{selectedFile.name}</p>
              <p>
                Размер:{' '}
                {(selectedFile.size / (1024 * 1024)).toFixed(2)} МБ
              </p>
            </div>
          )}
          {error && (
            <p className="mt-2 text-sm text-red-600">{error}</p>
          )}
        </div>

        <button
          type="submit"
          disabled={loading}
          className={`w-full btn-primary ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
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
        <h3 className="text-sm font-semibold text-gray-900 mb-3">Возможности:</h3>
        <ul className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {[
            'AI анализ по 12 критериям',
            'Автоматический перевод на русский',
            'Генерация озвучки',
            'Стильные субтитры как в TikTok',
            'Видео до 2 часов',
            'Клипы от 20 сек до 3 минут'
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

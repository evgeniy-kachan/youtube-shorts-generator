import React, { useState } from 'react';

const VideoInput = ({ onSubmit, loading }) => {
  const [filename, setFilename] = useState('test_video.mp4');
  const [error, setError] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    setError('');

    if (!filename.trim()) {
      setError('Пожалуйста, введите имя файла');
      return;
    }
    
    if (!filename.toLowerCase().endsWith('.mp4')) {
        setError('Имя файла должно заканчиваться на .mp4');
        return;
    }

    onSubmit(filename);
  };

  return (
    <div className="card max-w-3xl mx-auto">
      <div className="text-center mb-6">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">
          Создайте вирусные Shorts из локального видео
        </h2>
        <p className="text-gray-600">
          Введите имя видеофайла (например, test_video.mp4), загруженного в папку `temp`, и AI найдёт самые интересные моменты.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="filename-input" className="block text-sm font-medium text-gray-700 mb-2">
            Имя файла на сервере
          </label>
          <input
            id="filename-input"
            type="text"
            value={filename}
            onChange={(e) => setFilename(e.target.value)}
            placeholder="например, my_video.mp4"
            className="input-field"
            disabled={loading}
          />
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
            'Анализировать локальный файл'
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

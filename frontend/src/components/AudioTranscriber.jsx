import React, { useState, useRef } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * AudioTranscriber - компонент для транскрипции аудио с таймингами слов.
 * Используется для анализа эталонных переводов (например, Яндекс sync).
 */
export default function AudioTranscriber() {
  const [isOpen, setIsOpen] = useState(false);
  const [file, setFile] = useState(null);
  const [language, setLanguage] = useState('ru');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setResult(null);
      setError(null);
    }
  };

  const handleTranscribe = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(
        `${API_BASE_URL}/api/video/transcribe-audio?language=${language}`,
        {
          method: 'POST',
          body: formData,
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Transcription failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(2);
    return `${mins}:${secs.padStart(5, '0')}`;
  };

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-4 right-4 bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-full shadow-lg flex items-center gap-2 z-50 transition-all"
      >
        <span>🎤</span>
        <span className="text-sm font-medium">Транскрипция</span>
      </button>
    );
  }

  return (
    <div className="fixed bottom-4 right-4 w-[500px] max-h-[80vh] bg-white rounded-2xl shadow-2xl border border-gray-200 z-50 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xl">🎤</span>
          <span className="font-semibold">Транскрипция аудио</span>
        </div>
        <button
          onClick={() => setIsOpen(false)}
          className="text-white/80 hover:text-white text-xl"
        >
          ✕
        </button>
      </div>

      {/* Content */}
      <div className="p-4 flex-1 overflow-y-auto">
        {/* File upload */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Загрузите аудио (MP3, WAV, M4A)
          </label>
          <div className="flex gap-2">
            <input
              ref={fileInputRef}
              type="file"
              accept=".mp3,.wav,.m4a,.ogg,.webm,.mp4"
              onChange={handleFileChange}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex-1 px-4 py-2 border-2 border-dashed border-gray-300 rounded-lg hover:border-purple-400 hover:bg-purple-50 transition-colors text-gray-600"
            >
              {file ? (
                <span className="text-purple-600 font-medium">{file.name}</span>
              ) : (
                <span>📁 Выбрать файл...</span>
              )}
            </button>
          </div>
        </div>

        {/* Language selector */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Язык аудио
          </label>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
          >
            <option value="ru">🇷🇺 Русский</option>
            <option value="en">🇬🇧 English</option>
          </select>
        </div>

        {/* Transcribe button */}
        <button
          onClick={handleTranscribe}
          disabled={!file || isLoading}
          className={`w-full py-3 rounded-lg font-semibold transition-all ${
            !file || isLoading
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-purple-600 hover:bg-purple-700 text-white'
          }`}
        >
          {isLoading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                  fill="none"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
              Транскрибирую...
            </span>
          ) : (
            '🎯 Транскрибировать'
          )}
        </button>

        {/* Error */}
        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
            ❌ {error}
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="mt-4 space-y-4">
            {/* Stats */}
            <div className="flex gap-4 text-sm">
              <div className="bg-purple-50 px-3 py-2 rounded-lg">
                <span className="text-purple-600 font-semibold">{result.words.length}</span>
                <span className="text-gray-600 ml-1">слов</span>
              </div>
              <div className="bg-blue-50 px-3 py-2 rounded-lg">
                <span className="text-blue-600 font-semibold">{result.duration.toFixed(2)}с</span>
                <span className="text-gray-600 ml-1">длительность</span>
              </div>
              <div className="bg-green-50 px-3 py-2 rounded-lg">
                <span className="text-green-600 font-semibold">
                  {(result.words.length / result.duration).toFixed(1)}
                </span>
                <span className="text-gray-600 ml-1">слов/сек</span>
              </div>
            </div>

            {/* Full text */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm font-medium text-gray-700">Полный текст</label>
                <button
                  onClick={() => copyToClipboard(result.text)}
                  className="text-xs text-purple-600 hover:text-purple-800"
                >
                  📋 Копировать
                </button>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg text-sm text-gray-800 max-h-32 overflow-y-auto">
                {result.text}
              </div>
            </div>

            {/* Words with timestamps */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm font-medium text-gray-700">
                  Слова с таймингами ({result.words.length})
                </label>
                <button
                  onClick={() =>
                    copyToClipboard(
                      result.words
                        .map((w) => `${w.start.toFixed(2)}-${w.end.toFixed(2)}: ${w.word}`)
                        .join('\n')
                    )
                  }
                  className="text-xs text-purple-600 hover:text-purple-800"
                >
                  📋 Копировать JSON
                </button>
              </div>
              <div className="max-h-48 overflow-y-auto border border-gray-200 rounded-lg">
                <table className="w-full text-xs">
                  <thead className="bg-gray-100 sticky top-0">
                    <tr>
                      <th className="px-2 py-1 text-left text-gray-600">Время</th>
                      <th className="px-2 py-1 text-left text-gray-600">Слово</th>
                      <th className="px-2 py-1 text-right text-gray-600">Длит.</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.words.map((word, idx) => (
                      <tr
                        key={idx}
                        className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}
                      >
                        <td className="px-2 py-1 text-gray-500 font-mono">
                          {formatTime(word.start)}
                        </td>
                        <td className="px-2 py-1 text-gray-800 font-medium">
                          {word.word}
                        </td>
                        <td className="px-2 py-1 text-right text-gray-400">
                          {((word.end - word.start) * 1000).toFixed(0)}ms
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Copy as JSON */}
            <button
              onClick={() => copyToClipboard(JSON.stringify(result, null, 2))}
              className="w-full py-2 text-sm text-purple-600 hover:text-purple-800 border border-purple-200 rounded-lg hover:bg-purple-50 transition-colors"
            >
              📋 Копировать полный JSON
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

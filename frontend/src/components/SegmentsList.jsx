import React, { useState } from 'react';

const SegmentsList = ({ segments, videoTitle, onProcess, loading }) => {
  const [selectedSegments, setSelectedSegments] = useState([]);
  const [verticalMethod, setVerticalMethod] = useState('blur_background');

  const toggleSegment = (segmentId) => {
    setSelectedSegments((prev) =>
      prev.includes(segmentId)
        ? prev.filter((id) => id !== segmentId)
        : [...prev, segmentId]
    );
  };

  const selectAll = () => {
    setSelectedSegments(segments.map((s) => s.id));
  };

  const deselectAll = () => {
    setSelectedSegments([]);
  };

  const handleProcess = () => {
    if (selectedSegments.length > 0) {
      onProcess(selectedSegments, verticalMethod);
    }
  };

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getScoreColor = (score) => {
    if (score >= 0.8) return 'text-green-600 bg-green-100';
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-100';
    return 'text-orange-600 bg-orange-100';
  };

  const getScoreLabel = (score) => {
    if (score >= 0.8) return 'Отлично';
    if (score >= 0.6) return 'Хорошо';
    return 'Норм';
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Найденные моменты</h2>
            <p className="text-sm text-gray-600 mt-1">{videoTitle}</p>
            <p className="text-xs text-gray-500 mt-1">
              Найдено {segments.length} интересных моментов
            </p>
          </div>
          <div className="flex space-x-2">
            <button
              onClick={selectAll}
              className="btn-secondary text-sm"
              disabled={loading}
            >
              Выбрать все
            </button>
            <button
              onClick={deselectAll}
              className="btn-secondary text-sm"
              disabled={loading}
            >
              Снять выбор
            </button>
          </div>
        </div>

        <div className="space-y-3 max-h-96 overflow-y-auto">
          {segments.map((segment, index) => (
            <div
              key={segment.id}
              className={`border rounded-lg p-4 cursor-pointer transition-all duration-200 ${
                selectedSegments.includes(segment.id)
                  ? 'border-purple-600 bg-purple-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
              onClick={() => toggleSegment(segment.id)}
            >
              <div className="flex items-start space-x-4">
                <div className="flex-shrink-0 mt-1">
                  <input
                    type="checkbox"
                    checked={selectedSegments.includes(segment.id)}
                    onChange={() => toggleSegment(segment.id)}
                    className="w-5 h-5 text-purple-600 rounded focus:ring-purple-500"
                  />
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-semibold text-gray-900">
                        Сегмент {index + 1}
                      </span>
                      <span className="text-xs text-gray-500">
                        {formatDuration(segment.start_time)} - {formatDuration(segment.end_time)}
                      </span>
                      <span className="text-xs text-gray-500">
                        ({formatDuration(segment.duration)})
                      </span>
                    </div>
                    <div className={`px-3 py-1 rounded-full text-xs font-semibold ${getScoreColor(segment.highlight_score)}`}>
                      {getScoreLabel(segment.highlight_score)} {(segment.highlight_score * 100).toFixed(0)}%
                    </div>
                  </div>

                  <p className="text-sm text-gray-700 mb-3 line-clamp-2">
                    {segment.text_ru}
                  </p>

                  <div className="flex flex-wrap gap-2">
                    {Object.entries(segment.criteria_scores)
                      .filter(([_, score]) => score > 0.6)
                      .slice(0, 5)
                      .map(([criterion, score]) => (
                        <span
                          key={criterion}
                          className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-gray-100 text-gray-700"
                        >
                          {criterion.replace('_', ' ')}: {(score * 100).toFixed(0)}%
                        </span>
                      ))}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-6 pt-6 border-t space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              📱 Формат для Reels/Shorts (9:16):
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {[
                { id: 'blur_background', label: '🌟 Размытый фон', description: 'Видео по центру + blur' },
                { id: 'center_crop', label: '✂️ Центр-кроп', description: 'Простая обрезка по центру' },
                { id: 'smart_crop', label: '🤖 Smart', description: 'Кроп с учётом объекта (beta)' },
              ].map((method) => (
                <button
                  key={method.id}
                  type="button"
                  disabled={loading}
                  onClick={() => setVerticalMethod(method.id)}
                  className={`p-4 border rounded-xl text-left transition ${
                    verticalMethod === method.id ? 'border-purple-600 bg-purple-50' : 'hover:border-purple-500'
                  } ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <div className="font-semibold text-gray-900">{method.label}</div>
                  <div className="text-sm text-gray-600">{method.description}</div>
                </button>
              ))}
            </div>
            <p className="mt-1 text-xs text-gray-500">
              Видео будет автоматически переведено в вертикальный формат 1080×1920
            </p>
          </div>

          <div className="flex items-center justify-between">
            <p className="text-sm text-gray-600">
              Выбрано сегментов: <span className="font-semibold">{selectedSegments.length}</span>
            </p>
            <button
              onClick={handleProcess}
              disabled={selectedSegments.length === 0 || loading}
              className={`btn-primary ${
                selectedSegments.length === 0 || loading
                  ? 'opacity-50 cursor-not-allowed'
                  : ''
              }`}
            >
              {loading ? (
                <span className="flex items-center">
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
                  Обрабатываем...
                </span>
              ) : (
                `Создать ${selectedSegments.length} клипов`
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SegmentsList;


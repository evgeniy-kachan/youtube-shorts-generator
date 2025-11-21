import React from 'react';
import { getDownloadUrl } from '../services/api';

const DownloadList = ({ processedSegments, videoId, onReset, onBackToSegments }) => {
  const segments = Array.isArray(processedSegments) ? processedSegments : [];
  const segmentCount = segments.length;

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleDownload = (segmentId, filename) => {
    const url = getDownloadUrl(videoId, segmentId);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const downloadAll = () => {
    segments.forEach((segment, index) => {
      setTimeout(() => {
        handleDownload(segment.segment_id, segment.filename);
      }, index * 500); // Delay to avoid browser blocking multiple downloads
    });
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      <div className="card">
        <div className="text-center mb-6">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-green-100 rounded-full mb-4">
            <svg
              className="w-8 h-8 text-green-600"
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
          </div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            Клипы готовы!
          </h2>
          <p className="text-gray-600">
            Ваши {segmentCount} клипов обработаны и готовы к скачиванию
          </p>
        </div>

        <div className="space-y-3 mb-6">
          {segments.map((segment, index) => (
            <div
              key={segment.segment_id}
              className="flex items-center justify-between p-4 bg-gray-50 rounded-lg border border-gray-200"
            >
              <div className="flex items-center space-x-4">
                <div className="flex-shrink-0">
                  <div className="w-12 h-12 bg-gradient-to-r from-pink-500 to-purple-600 rounded-lg flex items-center justify-center">
                    <span className="text-white font-bold text-lg">{index + 1}</span>
                  </div>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-900">{segment.filename}</p>
                  <p className="text-xs text-gray-500">
                    Длительность: {formatDuration(segment.duration)}
                  </p>
                </div>
              </div>

              <button
                onClick={() => handleDownload(segment.segment_id, segment.filename)}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg shadow-sm text-white bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500"
              >
                <svg
                  className="w-5 h-5 mr-2"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                  />
                </svg>
                Скачать
              </button>
            </div>
          ))}
        </div>

        <div className="flex flex-col md:flex-row md:space-x-3 space-y-3 md:space-y-0">
          <button
            onClick={downloadAll}
            className="flex-1 btn-primary"
          >
            <svg
              className="w-5 h-5 mr-2 inline"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
              />
            </svg>
            Скачать все ({segmentCount})
          </button>
          <div className="flex flex-col md:w-1/3 space-y-3">
            <button
              onClick={onBackToSegments}
              className="btn-secondary"
            >
              Назад к выбору моментов
            </button>
            <button
              onClick={onReset}
              className="btn-outline"
            >
              Обработать новое видео
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DownloadList;


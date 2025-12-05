import React from 'react';

const Header = ({
  onNewVideo = () => {},
  canStartOver = false,
  isBusy = false,
}) => {
  return (
    <header className="bg-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="bg-gradient-to-r from-pink-500 to-purple-600 rounded-lg p-3">
              <svg
                className="w-8 h-8 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                />
              </svg>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                YouTube Shorts Generator
              </h1>
              <p className="text-sm text-gray-500">
                AI-powered viral moments extractor
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
              <span className="w-2 h-2 mr-2 bg-green-400 rounded-full animate-pulse"></span>
              Online
            </span>
          </div>
          <div className="flex items-center space-x-3">
            <button
              type="button"
              onClick={onNewVideo}
              disabled={!canStartOver}
              className={`inline-flex items-center px-4 py-2 rounded-lg text-sm font-medium border transition ${
                canStartOver
                  ? 'text-purple-600 border-purple-200 hover:bg-purple-50'
                  : 'text-gray-400 border-gray-200 cursor-not-allowed'
              }`}
            >
              <svg
                className="w-4 h-4 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 8v8m4-4H8m12 0a8 8 0 11-16 0 8 8 0 0116 0z"
                />
              </svg>
              {isBusy ? 'Завершаем...' : 'Сделать новое видео'}
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;


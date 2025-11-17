import React from 'react';

const ProgressBar = ({ progress, message, status }) => {
  const getStatusColor = () => {
    switch (status) {
      case 'completed':
        return 'bg-green-500';
      case 'failed':
        return 'bg-red-500';
      case 'processing':
        return 'bg-purple-600';
      default:
        return 'bg-gray-400';
    }
  };

  return (
    <div className="card max-w-3xl mx-auto">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {status === 'processing' && (
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600"></div>
            )}
            {status === 'completed' && (
              <div className="rounded-full h-8 w-8 bg-green-100 flex items-center justify-center">
                <svg
                  className="h-5 w-5 text-green-600"
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
            )}
            {status === 'failed' && (
              <div className="rounded-full h-8 w-8 bg-red-100 flex items-center justify-center">
                <svg
                  className="h-5 w-5 text-red-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </div>
            )}
            <div>
              <p className="text-sm font-medium text-gray-900">{message}</p>
              <p className="text-xs text-gray-500">{Math.round(progress * 100)}% завершено</p>
            </div>
          </div>
          <span className="text-2xl font-bold text-gray-900">{Math.round(progress * 100)}%</span>
        </div>

        <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
          <div
            className={`h-3 rounded-full transition-all duration-500 ${getStatusColor()}`}
            style={{ width: `${progress * 100}%` }}
          ></div>
        </div>
      </div>
    </div>
  );
};

export default ProgressBar;


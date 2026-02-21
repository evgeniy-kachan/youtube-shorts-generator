import React, { useState } from 'react';
import { getDownloadUrl, getTranscriptionDownloadUrl, generateDescription } from '../services/api';

const DownloadList = ({ processedSegments, videoId, onReset, onBackToSegments }) => {
  const [segments, setSegments] = useState(
    Array.isArray(processedSegments) ? processedSegments : []
  );
  const [copiedField, setCopiedField] = useState(null);
  const [regeneratingId, setRegeneratingId] = useState(null);
  
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
      }, index * 1200);
    });
  };

  const downloadAllDescriptions = () => {
    const lines = segments.map((segment, index) => {
      const desc = segment.description;
      const segLabel = `Segment ${segment.segment_id} — ${segment.filename}`;
      const separator = '='.repeat(segLabel.length);

      if (!desc) {
        return `${separator}\n${segLabel}\n${separator}\n\nОписание не сгенерировано\n`;
      }

      const parts = [
        `${separator}`,
        `${segLabel}`,
        `${separator}`,
        '',
      ];

      if (desc.category) parts.push(`Категория: ${desc.category}`, '');
      if (desc.title)    parts.push(`Заголовок: ${desc.title}`, '');
      if (desc.description) parts.push(`Описание:\n${desc.description}`, '');
      if (desc.guest_bio)   parts.push(`О госте:\n${desc.guest_bio}`, '');
      if (desc.hashtags?.length) parts.push(`Хэштеги: ${desc.hashtags.join(' ')}`, '');

      return parts.join('\n');
    });

    const content = lines.join('\n\n');
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `descriptions_${videoId || 'all'}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleDownloadTranscription = () => {
    if (!videoId) return;
    const url = getTranscriptionDownloadUrl(videoId);
    const link = document.createElement('a');
    link.href = url;
    link.download = `transcription_${videoId}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleCopy = async (text, fieldId) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedField(fieldId);
      setTimeout(() => setCopiedField(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const handleCopyAll = (segment) => {
    const desc = segment.description;
    if (!desc) return;
    const parts = [];
    if (desc.category)    parts.push(`Категория: ${desc.category}`);
    if (desc.title)       parts.push(`\n${desc.title}`);
    if (desc.description) parts.push(`\n${desc.description}`);
    if (desc.guest_bio)   parts.push(`\n${desc.guest_bio}`);
    if (desc.hashtags?.length) parts.push(`\n${desc.hashtags.join(' ')}`);
    handleCopy(parts.join('\n'), `all-${segment.segment_id}`);
  };

  const handleRegenerate = async (segment, index) => {
    setRegeneratingId(segment.segment_id);
    try {
      // We need original text - use segment_id to find it or use existing description context
      const result = await generateDescription(
        '', // text_en not available here, but DeepSeek can work with just Russian
        segment.description?.title || segment.filename, // Use title as context
        segment.duration || 60,
        0
      );
      
      // Update segment with new description
      const updatedSegments = [...segments];
      updatedSegments[index] = {
        ...segment,
        description: result
      };
      setSegments(updatedSegments);
    } catch (error) {
      console.error('Error regenerating description:', error);
      alert('Ошибка при генерации нового описания');
    } finally {
      setRegeneratingId(null);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
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
          <p className="text-gray-600 mb-4">
            {segmentCount} {segmentCount === 1 ? 'клип обработан' : 'клипов обработаны'} и готовы к скачиванию
          </p>
          <button
            onClick={handleDownloadTranscription}
            className="inline-flex items-center px-4 py-2 text-sm font-medium rounded-lg text-white bg-gradient-to-r from-blue-500 to-cyan-600 hover:from-blue-600 hover:to-cyan-700 shadow-sm transition"
          >
            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Скачать JSON с таймингами
          </button>
        </div>

        <div className="space-y-6 mb-6">
          {segments.map((segment, index) => (
            <div
              key={segment.segment_id}
              className="border border-gray-200 rounded-xl overflow-hidden bg-white shadow-sm"
            >
              {/* Header */}
              <div className="flex items-center justify-between p-4 bg-gradient-to-r from-purple-50 to-pink-50 border-b border-gray-100">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-gradient-to-r from-pink-500 to-purple-600 rounded-lg flex items-center justify-center">
                    <span className="text-white font-bold">{index + 1}</span>
                  </div>
                  <div>
                    <p className="font-semibold text-gray-900">{segment.filename}</p>
                    <p className="text-xs text-gray-500">
                      Длительность: {formatDuration(segment.duration)}
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => handleDownload(segment.segment_id, segment.filename)}
                  className="inline-flex items-center px-4 py-2 text-sm font-medium rounded-lg text-white bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700 shadow-sm transition"
                >
                  <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Скачать
                </button>
              </div>

              {/* Description */}
              {segment.description && (
                <div className="p-4">
                  {/* Combined description block */}
                  <div className="bg-gray-50 rounded-lg p-4 space-y-3">
                    {/* Category */}
                    {segment.description.category && (
                      <div>
                        <span className="text-xs font-semibold text-purple-600 uppercase tracking-wide">Категория</span>
                        <p className="text-gray-800 font-medium mt-1">
                          {segment.description.category}
                        </p>
                      </div>
                    )}

                    {/* Title */}
                    <div>
                      <span className="text-xs font-semibold text-purple-600 uppercase tracking-wide">Заголовок</span>
                      <p className="text-gray-900 font-semibold mt-1">
                        {segment.description.title}
                      </p>
                    </div>

                    {/* Description text */}
                    <div>
                      <span className="text-xs font-semibold text-purple-600 uppercase tracking-wide">Описание</span>
                      <p className="text-gray-700 mt-1 whitespace-pre-wrap">
                        {segment.description.description}
                      </p>
                    </div>

                    {/* Guest bio */}
                    {segment.description.guest_bio && (
                      <div>
                        <span className="text-xs font-semibold text-purple-600 uppercase tracking-wide">О госте</span>
                        <p className="text-gray-700 mt-1 whitespace-pre-wrap">
                          {segment.description.guest_bio}
                        </p>
                      </div>
                    )}

                    {/* Hashtags */}
                    {segment.description.hashtags && segment.description.hashtags.length > 0 && (
                      <div>
                        <span className="text-xs font-semibold text-purple-600 uppercase tracking-wide">Хэштеги</span>
                        <p className="text-purple-700 mt-1">
                          {segment.description.hashtags.join(' ')}
                        </p>
                      </div>
                    )}
                  </div>

                  {/* Action buttons */}
                  <div className="flex gap-3 mt-4">
                    <button
                      onClick={() => handleCopyAll(segment)}
                      className="flex-1 py-2.5 px-4 bg-purple-600 text-white rounded-lg font-medium hover:bg-purple-700 transition text-sm flex items-center justify-center"
                    >
                      {copiedField === `all-${segment.segment_id}` ? (
                        <>
                          <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                          Скопировано!
                        </>
                      ) : (
                        <>
                          <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                          </svg>
                          Копировать описание
                        </>
                      )}
                    </button>
                    <button
                      onClick={() => handleRegenerate(segment, index)}
                      disabled={regeneratingId === segment.segment_id}
                      className="py-2.5 px-4 border border-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-50 transition text-sm disabled:opacity-50 flex items-center"
                    >
                      {regeneratingId === segment.segment_id ? (
                        <>
                          <svg className="animate-spin h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
                          </svg>
                          Генерация...
                        </>
                      ) : (
                        <>
                          <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                          </svg>
                          Новое
                        </>
                      )}
                    </button>
                  </div>
                </div>
              )}

              {/* No description fallback */}
              {!segment.description && (
                <div className="p-4 text-center text-gray-500">
                  <p className="text-sm">Описание не сгенерировано</p>
                  <button
                    onClick={() => handleRegenerate(segment, index)}
                    disabled={regeneratingId === segment.segment_id}
                    className="mt-2 text-sm text-purple-600 hover:text-purple-800"
                  >
                    {regeneratingId === segment.segment_id ? 'Генерация...' : '📝 Сгенерировать'}
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Bottom actions */}
        <div className="flex flex-col sm:flex-row gap-3 flex-wrap">
          <button
            onClick={downloadAll}
            className="flex-1 btn-primary"
          >
            <svg className="w-5 h-5 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Скачать все ({segmentCount})
          </button>
          <button
            onClick={downloadAllDescriptions}
            className="flex-1 inline-flex items-center justify-center px-4 py-2.5 text-sm font-medium rounded-lg text-white bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 shadow-sm transition"
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Скачать описания (.txt)
          </button>
          <button
            onClick={onBackToSegments}
            className="btn-secondary"
          >
            Назад к выбору
          </button>
          <button
            onClick={onReset}
            className="btn-outline"
          >
            Новое видео
          </button>
        </div>
      </div>
    </div>
  );
};

export default DownloadList;

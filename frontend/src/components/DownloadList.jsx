import React, { useState } from 'react';
import { getDownloadUrl, getTranscriptionDownloadUrl, generateDescription } from '../services/api';

const DownloadList = ({ processedSegments, videoId, onReset, onBackToSegments }) => {
  const [segments, setSegments] = useState(
    Array.isArray(processedSegments) ? processedSegments : []
  );
  const [copiedField, setCopiedField] = useState(null);
  const [regeneratingId, setRegeneratingId] = useState(null);
  const [showDescMenu, setShowDescMenu] = useState(false);
  
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
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    // Don't remove immediately — give browser time to start the download
    setTimeout(() => {
      try { document.body.removeChild(link); } catch (_) {}
    }, 5000);
  };

  const downloadAll = () => {
    // Stagger downloads with 2.5s delay — browsers throttle rapid programmatic downloads
    segments.forEach((segment, index) => {
      setTimeout(() => {
        handleDownload(segment.segment_id, segment.filename);
      }, index * 2500);
    });
  };

  const buildDescriptionsHtml = () => {
    const cards = segments.map((segment) => {
      const desc = segment.description;
      if (!desc) return `<div class="card"><h2>${segment.filename}</h2><p class="empty">Описание не сгенерировано</p></div>`;

      const alts = (desc.title_alternatives || []).map(a => `<li>${a}</li>`).join('');
      return `<div class="card">
        <h2>${segment.filename}</h2>
        ${desc.category ? `<p class="badge">${desc.category}</p>` : ''}
        <div class="platform-label yt">▶ YouTube</div>
        ${desc.title ? `<h3>${desc.title}</h3>` : ''}
        ${alts ? `<p class="alts"><strong>Альтернативы:</strong><ul>${alts}</ul></p>` : ''}
        ${desc.description ? `<p class="desc">${desc.description.replace(/\n/g, '<br>')}</p>` : ''}
        ${desc.guest_bio ? `<div class="guest"><strong>О госте:</strong><br>${desc.guest_bio.replace(/\n/g, '<br>')}</div>` : ''}
        ${(desc.title_tiktok || desc.description_tiktok) ? `
          <div class="tiktok-block">
            <div class="platform-label tt">🎵 TikTok</div>
            ${desc.title_tiktok ? `<h3>${desc.title_tiktok}</h3>` : ''}
            ${desc.description_tiktok ? `<p class="desc">${desc.description_tiktok.replace(/\n/g, '<br>')}</p>` : ''}
          </div>` : ''}
        ${desc.hashtags?.length ? `<p class="tags">${desc.hashtags.join(' ')}</p>` : ''}
      </div>`;
    }).join('\n');

    return `<!DOCTYPE html>
<html lang="ru"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Описания — ${videoId || 'kachan.cuts'}</title>
<style>
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f5f7;margin:0;padding:16px;color:#1d1d1f}
  .card{background:#fff;border-radius:16px;padding:20px 24px;margin-bottom:20px;box-shadow:0 2px 12px rgba(0,0,0,.08)}
  h2{font-size:14px;color:#888;font-weight:500;margin:0 0 8px}
  h3{font-size:18px;font-weight:700;margin:4px 0 12px;color:#1d1d1f}
  .badge{display:inline-block;background:#f0e6ff;color:#7c3aed;border-radius:20px;padding:2px 12px;font-size:12px;font-weight:600;margin-bottom:10px}
  .desc{font-size:15px;line-height:1.6;color:#333;margin:0 0 12px}
  .guest{font-size:14px;line-height:1.6;color:#555;background:#f9f9fb;border-radius:10px;padding:12px;margin-bottom:12px}
  .tags{font-size:13px;color:#7c3aed;margin:0;font-weight:500}
  .alts{font-size:14px;color:#555;margin:0 0 12px} .alts ul{margin:4px 0 0;padding-left:18px} .alts li{margin-bottom:2px}
  .platform-label{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}
  .yt{color:#dc2626} .tt{color:#fff}
  .tiktok-block{background:#1c1c1e;border-radius:10px;padding:14px;margin:12px 0} .tiktok-block h3{color:#fff} .tiktok-block .desc{color:#ccc}
  .empty{color:#aaa;font-style:italic}
  @media(prefers-color-scheme:dark){body{background:#1c1c1e;color:#f5f5f7}.card{background:#2c2c2e}.desc{color:#e0e0e0}.guest{background:#3a3a3c;color:#ccc}.badge{background:#3d2a5a}}
</style></head><body>
<h1 style="font-size:22px;margin-bottom:20px">🎬 ${videoId || 'kachan.cuts'}</h1>
${cards}
</body></html>`;
  };

  const downloadAllDescriptions = (format = 'html') => {
    if (format === 'html') {
      const html = buildDescriptionsHtml();
      const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `descriptions_${videoId || 'all'}.html`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } else {
      const lines = segments.map((segment) => {
        const desc = segment.description;
        const segLabel = `Segment ${segment.segment_id} — ${segment.filename}`;
        const separator = '='.repeat(segLabel.length);
        if (!desc) return `${separator}\n${segLabel}\n${separator}\n\nОписание не сгенерировано\n`;
        const parts = [separator, segLabel, separator, ''];
        if (desc.category) parts.push(`Категория: ${desc.category}`, '');

        // YouTube section
        parts.push('▶ YOUTUBE', '');
        if (desc.title) parts.push(`Заголовок: ${desc.title}`, '');
        if (desc.title_alternatives?.length) {
          parts.push('Альтернативы:');
          desc.title_alternatives.forEach((alt, i) => {
            const labels = ['Числовой', 'Цитатный', 'Интригующий'];
            parts.push(`  ${labels[i] || i + 1}. ${alt}`);
          });
          parts.push('');
        }
        if (desc.description) {
          parts.push(desc.description, '');
        }
        if (desc.guest_bio) parts.push(`О госте:\n${desc.guest_bio}`, '');

        // TikTok section
        if (desc.title_tiktok || desc.description_tiktok) {
          parts.push('🎵 TIKTOK', '');
          if (desc.title_tiktok) parts.push(`Заголовок: ${desc.title_tiktok}`, '');
          if (desc.description_tiktok) parts.push(desc.description_tiktok, '');
        }

        if (desc.hashtags?.length) parts.push(desc.hashtags.join(' '), '');
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
    }
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

  const handleCopyYoutube = (segment) => {
    const desc = segment.description;
    if (!desc) return;
    const parts = [];
    if (desc.title) parts.push(desc.title);
    if (desc.description) parts.push(`\n${desc.description}`);
    if (desc.guest_bio) parts.push(`\n${desc.guest_bio}`);
    if (desc.hashtags?.length) parts.push(`\n${desc.hashtags.join(' ')}`);
    handleCopy(parts.join('\n'), `yt-${segment.segment_id}`);
  };

  const handleCopyTiktok = (segment) => {
    const desc = segment.description;
    if (!desc) return;
    const parts = [];
    if (desc.title_tiktok) parts.push(desc.title_tiktok);
    else if (desc.title) parts.push(desc.title);
    if (desc.description_tiktok) parts.push(`\n${desc.description_tiktok}`);
    else if (desc.description) parts.push(`\n${desc.description}`);
    if (desc.hashtags?.length) parts.push(`\n${desc.hashtags.join(' ')}`);
    handleCopy(parts.join('\n'), `tt-${segment.segment_id}`);
  };

  const handleRegenerate = async (segment, index) => {
    setRegeneratingId(segment.segment_id);
    try {
      const result = await generateDescription(
        segment.text_en || '',
        segment.text_ru || '',
        segment.duration || 60,
        0,
        segment.guest_name || ''
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
                  {/* YouTube block */}
                  <div className="bg-gray-50 rounded-lg p-4 space-y-3">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-red-500 text-sm">▶</span>
                      <span className="text-xs font-bold text-red-600 uppercase tracking-wide">YouTube</span>
                    </div>

                    {/* Category */}
                    {segment.description.category && (
                      <div>
                        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Категория</span>
                        <p className="text-gray-800 font-medium mt-0.5 text-sm">
                          {segment.description.category}
                        </p>
                      </div>
                    )}

                    {/* Title */}
                    <div>
                      <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Заголовок</span>
                      <p className="text-gray-900 font-semibold mt-0.5">
                        {segment.description.title}
                      </p>
                    </div>

                    {/* Alternative titles */}
                    {segment.description.title_alternatives && segment.description.title_alternatives.length > 0 && (
                      <div>
                        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Альтернативы</span>
                        <ul className="mt-0.5 space-y-0.5">
                          {segment.description.title_alternatives.map((alt, i) => (
                            <li key={i} className="text-gray-600 text-sm flex items-start gap-1.5">
                              <span className="text-gray-400 mt-0.5 text-xs">
                                {i === 0 ? '🔢' : i === 1 ? '💬' : '❓'}
                              </span>
                              {alt}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Description text */}
                    <div>
                      <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Описание</span>
                      <p className="text-gray-700 mt-0.5 whitespace-pre-wrap text-sm">
                        {segment.description.description}
                      </p>
                    </div>

                    {/* Guest bio */}
                    {segment.description.guest_bio && (
                      <div>
                        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">О госте</span>
                        <p className="text-gray-700 mt-0.5 whitespace-pre-wrap text-sm">
                          {segment.description.guest_bio}
                        </p>
                      </div>
                    )}

                    {/* Hashtags */}
                    {segment.description.hashtags && segment.description.hashtags.length > 0 && (
                      <div>
                        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Хэштеги</span>
                        <p className="text-purple-700 mt-0.5 text-sm">
                          {segment.description.hashtags.join(' ')}
                        </p>
                      </div>
                    )}
                  </div>

                  {/* TikTok block */}
                  {(segment.description.title_tiktok || segment.description.description_tiktok) && (
                    <div className="bg-gray-900 rounded-lg p-4 space-y-2 mt-3">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm">🎵</span>
                        <span className="text-xs font-bold text-white uppercase tracking-wide">TikTok</span>
                      </div>
                      {segment.description.title_tiktok && (
                        <div>
                          <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Заголовок</span>
                          <p className="text-white font-semibold mt-0.5">
                            {segment.description.title_tiktok}
                          </p>
                        </div>
                      )}
                      {segment.description.description_tiktok && (
                        <div>
                          <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Описание</span>
                          <p className="text-gray-300 mt-0.5 whitespace-pre-wrap text-sm">
                            {segment.description.description_tiktok}
                          </p>
                        </div>
                      )}
                      {segment.description.hashtags && segment.description.hashtags.length > 0 && (
                        <p className="text-cyan-400 text-sm">
                          {segment.description.hashtags.join(' ')}
                        </p>
                      )}
                    </div>
                  )}

                  {/* Action buttons */}
                  <div className="flex gap-2 mt-4 flex-wrap">
                    {/* Copy for YouTube */}
                    <button
                      onClick={() => handleCopyYoutube(segment)}
                      className="flex-1 py-2 px-3 bg-red-600 text-white rounded-lg font-medium hover:bg-red-700 transition text-sm flex items-center justify-center min-w-[140px]"
                    >
                      {copiedField === `yt-${segment.segment_id}` ? (
                        <>✓ Скопировано!</>
                      ) : (
                        <>▶ Копировать YouTube</>
                      )}
                    </button>
                    {/* Copy for TikTok */}
                    {(segment.description.title_tiktok || segment.description.description_tiktok) && (
                      <button
                        onClick={() => handleCopyTiktok(segment)}
                        className="flex-1 py-2 px-3 bg-gray-900 text-white rounded-lg font-medium hover:bg-gray-800 transition text-sm flex items-center justify-center min-w-[140px]"
                      >
                        {copiedField === `tt-${segment.segment_id}` ? (
                          <>✓ Скопировано!</>
                        ) : (
                          <>🎵 Копировать TikTok</>
                        )}
                      </button>
                    )}
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
          <div className="relative flex-1">
            <div className="flex rounded-lg overflow-hidden shadow-sm">
              <button
                onClick={() => downloadAllDescriptions('html')}
                className="flex-1 inline-flex items-center justify-center px-4 py-2.5 text-sm font-medium text-white bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 transition"
              >
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Описания (.html)
              </button>
              <button
                onClick={() => setShowDescMenu(v => !v)}
                className="px-2 py-2.5 text-white bg-teal-600 hover:bg-teal-700 border-l border-teal-400 transition"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
            </div>
            {showDescMenu && (
              <div className="absolute bottom-full mb-1 right-0 bg-white border border-gray-200 rounded-lg shadow-lg z-10 min-w-[140px]">
                <button
                  onClick={() => { downloadAllDescriptions('html'); setShowDescMenu(false); }}
                  className="w-full text-left px-4 py-2 text-sm hover:bg-gray-50 text-gray-700"
                >
                  📄 Скачать .html
                </button>
                <button
                  onClick={() => { downloadAllDescriptions('txt'); setShowDescMenu(false); }}
                  className="w-full text-left px-4 py-2 text-sm hover:bg-gray-50 text-gray-700"
                >
                  📝 Скачать .txt
                </button>
              </div>
            )}
          </div>
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

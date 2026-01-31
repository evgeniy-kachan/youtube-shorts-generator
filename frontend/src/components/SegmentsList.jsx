import React, { useMemo, useState } from 'react';

const CRITERIA_LABELS = {
  surprise_novelty: 'Неожиданность',
  specificity_score: 'Конкретика',
  personal_connection: 'Личная история',
  actionability_score: 'Практичность',
  clarity_simplicity: 'Понятность',
  completeness_arc: 'Законченность',
  hook_quality: 'Сила хука',
};

const SUBTITLE_POSITIONS = [
  {
    id: 'mid_low',
    label: 'Чуть ниже центра',
    description: 'Сдвигаем текст чуть ниже центральной линии',
    coords: { x: 540, y: 1050 },
  },
  {
    id: 'lower_center',
    label: 'Нижняя треть',
    description: 'Классическая позиция ближе к нижней трети',
    coords: { x: 540, y: 1350 },
  },
  {
    id: 'bottom_center',
    label: 'Самый низ',
    description: 'Максимально низкое размещение',
    coords: { x: 540, y: 1520 },
  },
];

const FONT_OPTIONS = [
  {
    id: 'Montserrat Bold',
    label: 'Montserrat Bold ★',
    css: '"Montserrat", sans-serif',
    weight: 700,
  },
  {
    id: 'Montserrat Light',
    label: 'Montserrat Light',
    css: '"Montserrat", sans-serif',
    weight: 300,
  },
  {
    id: 'Montserrat Medium',
    label: 'Montserrat Medium',
    css: '"Montserrat", sans-serif',
    weight: 500,
  },
  {
    id: 'Montserrat Regular',
    label: 'Montserrat Regular',
    css: '"Montserrat", sans-serif',
    weight: 400,
  },
  {
    id: 'Inter',
    label: 'Inter Regular',
    css: '"Inter", sans-serif',
    weight: 400,
  },
  {
    id: 'Inter ExtraLight',
    label: 'Inter ExtraLight',
    css: '"Inter", sans-serif',
    weight: 200,
  },
  {
    id: 'Open Sans',
    label: 'Open Sans Regular',
    css: '"Open Sans", sans-serif',
    weight: 400,
  },
  {
    id: 'Open Sans Light',
    label: 'Open Sans Light',
    css: '"Open Sans", sans-serif',
    weight: 300,
  },
  {
    id: 'Nunito',
    label: 'Nunito Regular',
    css: '"Nunito", sans-serif',
    weight: 400,
  },
  {
    id: 'Nunito Light',
    label: 'Nunito Light',
    css: '"Nunito", sans-serif',
    weight: 300,
  },
  {
    id: 'Roboto',
    label: 'Roboto Regular',
    css: '"Roboto", sans-serif',
    weight: 400,
  },
  {
    id: 'Roboto Light',
    label: 'Roboto Light',
    css: '"Roboto", sans-serif',
    weight: 300,
  },
  {
    id: 'Rubik',
    label: 'Rubik Regular',
    css: '"Rubik", sans-serif',
    weight: 400,
  },
  {
    id: 'Source Sans 3',
    label: 'Source Sans 3 Regular',
    css: '"Source Sans 3", sans-serif',
    weight: 400,
  },
  {
    id: 'Source Sans 3 Light',
    label: 'Source Sans 3 Light',
    css: '"Source Sans 3", sans-serif',
    weight: 300,
  },
  {
    id: 'IBMPlexSans Light',
    label: 'IBM Plex Sans Light',
    css: '"IBM Plex Sans", sans-serif',
    weight: 300,
  },
  {
    id: 'IBMPlexSans Condensed',
    label: 'IBM Plex Sans Condensed',
    css: '"IBM Plex Sans Condensed", sans-serif',
    weight: 400,
  },
  {
    id: 'Open Sans Condensed Light',
    label: 'Open Sans Condensed Light',
    css: '"Open Sans Condensed", sans-serif',
    weight: 300,
  },
  {
    id: 'Roboto Regular',
    label: 'Roboto Regular',
    css: '"Roboto", sans-serif',
    weight: 400,
  },
];

const FONT_SIZE_OPTIONS = [72, 82, 92, 102];

const TARGET_CANVAS_WIDTH = 1080;
const TARGET_CANVAS_HEIGHT = 1920;
const PHONE_DISPLAY_WIDTH = 220;
const PHONE_DISPLAY_HEIGHT = Math.round(
  (TARGET_CANVAS_HEIGHT / TARGET_CANVAS_WIDTH) * PHONE_DISPLAY_WIDTH
);
const PHONE_SCALE = PHONE_DISPLAY_WIDTH / TARGET_CANVAS_WIDTH;

const SubtitlePreview = ({
  text,
  positionId,
  fontFamily,
  fontSize,
  fontWeight,
  animation,
  thumbnailUrl,
  showBackground = false,
}) => {
  const positionCoords =
    SUBTITLE_POSITIONS.find((preset) => preset.id === positionId)?.coords ||
    SUBTITLE_POSITIONS[0].coords;
  const previewFontSize = Math.round(fontSize * PHONE_SCALE);
  const previewLines = useMemo(() => {
    const words = text.split(/\s+/).filter(Boolean);
    if (words.length === 0) return ['Текст субтитров'];
    const chunks = [];
    let current = [];
    const maxWordsPerLine = 6;
    words.forEach((word) => {
      current.push(word);
      if (current.length >= maxWordsPerLine) {
        chunks.push(current.join(' '));
        current = [];
      }
    });
    if (current.length) {
      chunks.push(current.join(' '));
    }
    return chunks.slice(0, 2);
  }, [text]);

  const containerClass = thumbnailUrl
    ? 'relative mx-auto rounded-[32px] overflow-hidden bg-black shadow-lg border border-black/20'
    : 'relative mx-auto rounded-[32px] overflow-hidden bg-gradient-to-br from-gray-800 via-gray-700 to-gray-900 shadow-lg';

  const containerStyle = {
    width: `${PHONE_DISPLAY_WIDTH}px`,
    height: `${PHONE_DISPLAY_HEIGHT}px`,
    backgroundImage: thumbnailUrl ? `url(${thumbnailUrl})` : undefined,
    backgroundSize: 'cover',
    backgroundPosition: 'center',
  };

  const blockWrapperStyle = {
    position: 'absolute',
    width: '80%',
    left: `${(positionCoords.x / TARGET_CANVAS_WIDTH) * 100}%`,
    top: `${(positionCoords.y / TARGET_CANVAS_HEIGHT) * 100}%`,
    transform: 'translate(-50%, -50%)',
    display: 'flex',
    justifyContent: 'center',
  };

  const lineTextStyle = {
    fontFamily,
    fontSize: `${previewFontSize}px`,
    lineHeight: 1.2,
    fontWeight,
    color: '#fff',
  };

  const blockStyle = (hasBackground) => ({
    fontFamily,
    fontSize: `${previewFontSize}px`,
    lineHeight: 1.2,
    fontWeight,
    width: '100%',
    padding: '10px 16px',
    borderRadius: '24px',
    border: hasBackground ? '2px solid rgba(255,255,255,0.3)' : 'none',
    backgroundColor: hasBackground ? 'rgba(0, 0, 0, 0.55)' : 'transparent',
    color: '#fff',
    textAlign: 'center',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  });

  return (
    <div className="bg-gray-50 border rounded-2xl shadow-inner p-4 sticky top-6 self-start">
      <div className="flex items-center justify-between mb-2">
        <p className="text-sm font-semibold text-gray-800">Превью макета</p>
        <span className="text-xs text-gray-500">9:16</span>
      </div>
      <div className={containerClass} style={containerStyle}>
        <div className="absolute inset-0 opacity-30 bg-[radial-gradient(circle_at_top,_rgba(255,255,255,0.35),_transparent_55%)]" />
        <div style={blockWrapperStyle}>
          <div
            className="subtitle-preview-card flex flex-col items-center"
            style={blockStyle(showBackground)}
          >
            {previewLines.slice(0, 2).map((line, idx) => {
              const words = line.split(' ').filter(Boolean);
              return (
                <span key={idx} className="preview-line block">
                  {words.map((word, wordIdx) => (
                    <span
                      key={`${idx}-${wordIdx}`}
                      className={`preview-word anim-${animation}`}
                      style={{
                        ...lineTextStyle,
                        animationDelay: `${wordIdx * 0.2}s`,
                        marginRight: '0.25em',
                      }}
                    >
                      {word}
                    </span>
                  ))}
                </span>
              );
            })}
          </div>
        </div>
      </div>
      <p className="text-xs text-gray-500 mt-3 text-center">
        Прокрутите вниз для настроек
      </p>
    </div>
  );
};

const SegmentsList = ({
  segments,
  videoTitle,
  onProcess,
  onDubbing,
  loading,
  videoThumbnail,
}) => {
  const [selectedSegments, setSelectedSegments] = useState([]);
  const [expandedSegments, setExpandedSegments] = useState([]);
  const [verticalMethod, setVerticalMethod] = useState('center_crop');

  // Style settings
  const [subtitleAnimation, setSubtitleAnimation] = useState('highlight');
  const [subtitlePosition, setSubtitlePosition] = useState('lower_center');
  const [subtitleFont, setSubtitleFont] = useState('Montserrat Medium');
  const [subtitleFontSize, setSubtitleFontSize] = useState(82);
  const [subtitleBackground, setSubtitleBackground] = useState(false);
  const [subtitleGlow, setSubtitleGlow] = useState(false); // Soft glow effect for readability
  const [subtitleGradient, setSubtitleGradient] = useState(true); // Dark gradient at bottom (enabled by default)
  const speakerColorMode = 'colored'; // Always use colored mode
  
  const [ttsProvider, setTtsProvider] = useState('elevenlabs');
  const [voiceMix, setVoiceMix] = useState('male_duo');
  const [preserveBackgroundAudio, setPreserveBackgroundAudio] = useState(true);
  const [numSpeakers, setNumSpeakers] = useState(0); // 0 = auto-detect, 1-3 = fixed
  const [speakerChangeTime, setSpeakerChangeTime] = useState(''); // Time in seconds when speaker changes (e.g., "15" or "15,30")
  // cropFocus is now always 'face_auto' for center_crop
  const cropFocus = verticalMethod === 'center_crop' ? 'face_auto' : 'center';


  // Tabs: 'style', 'text', 'position'
  const [activeTab, setActiveTab] = useState('style');

  const toggleSegment = (segmentId) => {
    setSelectedSegments((prev) =>
      prev.includes(segmentId)
        ? prev.filter((id) => id !== segmentId)
        : [...prev, segmentId]
    );
  };

  const toggleExpand = (segmentId) => {
    setExpandedSegments((prev) =>
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

  // Group segments by tier: strict, extended, fallback
  const { strictSegments, extendedSegments, fallbackSegments } = useMemo(() => {
    const strict = [];
    const extended = [];
    const fallback = [];
    
    segments.forEach((segment) => {
      const tier = segment.tier || 'extended'; // Default to extended for backward compatibility
      if (tier === 'strict') {
        strict.push(segment);
      } else if (tier === 'fallback') {
        fallback.push(segment);
      } else {
        extended.push(segment);
      }
    });
    
    return { 
      strictSegments: strict, 
      extendedSegments: extended, 
      fallbackSegments: fallback 
    };
  }, [segments]);

  const handleProcess = () => {
    if (selectedSegments.length > 0) {
      console.log('[SegmentsList] Calling onProcess with subtitleAnimation:', subtitleAnimation, 'numSpeakers:', numSpeakers, 'speakerChangeTime:', speakerChangeTime);
      onProcess(
        selectedSegments,
        verticalMethod,
        subtitleAnimation,
        subtitlePosition,
        subtitleFont,
        subtitleFontSize,
        subtitleBackground,
        subtitleGlow,
        subtitleGradient,
        ttsProvider,
        voiceMix,
        preserveBackgroundAudio,
        cropFocus,
        speakerColorMode,
        numSpeakers,
        speakerChangeTime
      );
    }
  };

  const handleDubbing = () => {
    if (selectedSegments.length > 0 && onDubbing) {
      // AI Dubbing processes one segment at a time
      onDubbing(
        selectedSegments[0], // Only first selected segment
        verticalMethod,
        subtitleAnimation,
        subtitlePosition,
        subtitleFont,
        subtitleFontSize,
        subtitleBackground,
        subtitleGlow,
        subtitleGradient,
        cropFocus
      );
    }
  };

  const previewText = useMemo(() => {
    if (!segments || segments.length === 0) {
      return 'Так будут выглядеть субтитры';
    }
    const words = segments[0].text_ru?.split(' ') ?? [];
    return words.slice(0, 12).join(' ') || 'Так будут выглядеть субтитры';
  }, [segments]);

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
    <div className="space-y-6 max-w-6xl mx-auto pb-20">
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">
              Найденные моменты
            </h2>
            <p className="text-sm text-gray-600 mt-1">{videoTitle}</p>
            <p className="text-xs text-gray-500 mt-1">
              Найдено {segments.length} интересных моментов
              {strictSegments.length > 0 && ` (⭐${strictSegments.length} строгих`}
              {extendedSegments.length > 0 && `${strictSegments.length > 0 ? ', ' : ' ('}📋${extendedSegments.length} расширенных`}
              {fallbackSegments.length > 0 && `, 📎${fallbackSegments.length} доп.`}
              {(strictSegments.length > 0 || extendedSegments.length > 0 || fallbackSegments.length > 0) && ')'}
            </p>
            <p className="text-xs text-purple-500 mt-1">
              Нажмите карточку, чтобы раскрыть полный текст (по умолчанию видно
              2 строки).
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

        {/* Helper function to render segment card */}
        {(() => {
          const renderSegmentCard = (segment, globalIndex) => {
            const isSelected = selectedSegments.includes(segment.id);
            const isExpanded = expandedSegments.includes(segment.id);
            return (
              <div
                key={segment.id}
                className={`border rounded-lg p-4 cursor-pointer transition-all duration-200 ${
                  isSelected
                    ? 'border-purple-600 bg-purple-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => toggleExpand(segment.id)}
              >
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0 mt-1">
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={(e) => {
                        e.stopPropagation();
                        toggleSegment(segment.id);
                      }}
                      onClick={(e) => e.stopPropagation()}
                      className="w-5 h-5 text-purple-600 rounded focus:ring-purple-500"
                    />
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-semibold text-gray-900">
                          Сегмент {globalIndex + 1}
                        </span>
                        <span className="text-xs text-gray-500">
                          {formatDuration(segment.start_time)} -{' '}
                          {formatDuration(segment.end_time)}
                        </span>
                        <span className="text-xs text-gray-500">
                          ({formatDuration(segment.duration)})
                        </span>
                      </div>
                      <div
                        className={`px-3 py-1 rounded-full text-xs font-semibold ${getScoreColor(
                          segment.highlight_score
                        )}`}
                      >
                        {getScoreLabel(segment.highlight_score)}{' '}
                        {(segment.highlight_score * 100).toFixed(0)}%
                      </div>
                    </div>

                    <p
                      className={`text-sm text-gray-700 mb-3 ${
                        isExpanded ? '' : 'line-clamp-2'
                      }`}
                    >
                      {segment.text_ru}
                    </p>

                    <div className="flex flex-wrap gap-2">
                      {/* Context flags */}
                      {segment.needs_previous_context && (
                        <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-amber-100 text-amber-800">
                          ⚠️ Нужен контекст до
                        </span>
                      )}
                      {segment.needs_next_context && (
                        <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-amber-100 text-amber-800">
                          ⚠️ Нужен контекст после
                        </span>
                      )}
                      {segment.merged_from_starts && (
                        <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-blue-100 text-blue-800">
                          🔗 Объединён
                        </span>
                      )}
                      {/* Criteria scores */}
                      {Object.entries(segment.criteria_scores || {})
                        .filter(([key, score]) => typeof score === 'number' && score > 0.6)
                        .slice(0, 5)
                        .map(([criterion, score]) => (
                          <span
                            key={criterion}
                            className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-gray-100 text-gray-700"
                          >
                            {CRITERIA_LABELS[criterion] ||
                              criterion.replace(/_/g, ' ')}
                            : {(score * 100).toFixed(0)}%
                          </span>
                        ))}
                    </div>
                  </div>
                </div>
              </div>
            );
          };

          // Calculate global indices for proper numbering
          let globalIndex = 0;
          const strictWithIndex = strictSegments.map(s => ({ segment: s, index: globalIndex++ }));
          const extendedWithIndex = extendedSegments.map(s => ({ segment: s, index: globalIndex++ }));
          const fallbackWithIndex = fallbackSegments.map(s => ({ segment: s, index: globalIndex++ }));

          return (
            <div className="space-y-6 max-h-[600px] overflow-y-auto mb-8">
              {/* STRICT tier - high quality segments */}
              {strictSegments.length > 0 && (
                <div>
                  <div className="flex items-center gap-2 mb-3 sticky top-0 bg-white py-2 z-10">
                    <span className="text-sm font-bold text-green-700">⭐ Строгая выборка</span>
                    <span className="text-xs text-gray-500">
                      ({strictSegments.length} сегментов, score ≥ 35%)
                    </span>
                  </div>
                  <div className="space-y-3 pl-2 border-l-4 border-green-400">
                    {strictWithIndex.map(({ segment, index }) => renderSegmentCard(segment, index))}
                  </div>
                </div>
              )}

              {/* EXTENDED tier - good quality segments */}
              {extendedSegments.length > 0 && (
                <div>
                  <div className="flex items-center gap-2 mb-3 sticky top-0 bg-white py-2 z-10">
                    <span className="text-sm font-bold text-blue-700">📋 Расширенная выборка</span>
                    <span className="text-xs text-gray-500">
                      ({extendedSegments.length} сегментов, score ≥ 25%)
                    </span>
                  </div>
                  <div className="space-y-3 pl-2 border-l-4 border-blue-400">
                    {extendedWithIndex.map(({ segment, index }) => renderSegmentCard(segment, index))}
                  </div>
                </div>
              )}

              {/* FALLBACK tier - acceptable segments */}
              {fallbackSegments.length > 0 && (
                <div>
                  <div className="flex items-center gap-2 mb-3 sticky top-0 bg-white py-2 z-10">
                    <span className="text-sm font-bold text-gray-500">📎 Дополнительные</span>
                    <span className="text-xs text-gray-500">
                      ({fallbackSegments.length} сегментов, score ≥ 15%)
                    </span>
                  </div>
                  <div className="space-y-3 pl-2 border-l-4 border-gray-300">
                    {fallbackWithIndex.map(({ segment, index }) => renderSegmentCard(segment, index))}
                  </div>
                </div>
              )}
            </div>
          );
        })()}

        <div className="pt-6 border-t">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Настройки видео
          </h3>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Left Column: Controls (Tabs) */}
            <div className="lg:col-span-7 space-y-6">
              {/* Format Selection (Always visible) */}
              <div>
                <label className="block text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">
                  Формат (9:16)
                </label>
                <div className="flex space-x-3">
                  {[
                    {
                      id: 'letterbox',
                      label: '⚫️ Вписать (поля)',
                      description: 'Чёрные полосы сверху/снизу',
                    },
                    {
                      id: 'center_crop',
                      label: '🤖 Умный кроп',
                      description: 'Автофокус на лица',
                    },
                  ].map((method) => (
                    <button
                      key={method.id}
                      type="button"
                      disabled={loading}
                      onClick={() => setVerticalMethod(method.id)}
                      className={`flex-1 py-3 px-4 border rounded-xl text-left transition ${
                        verticalMethod === method.id
                          ? 'border-purple-600 bg-purple-50 ring-1 ring-purple-300'
                          : 'border-gray-200 text-gray-600 hover:border-gray-300'
                      }`}
                    >
                      <div className="font-semibold text-gray-900">
                        {method.label}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        {method.description}
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">
                  Озвучка
                </label>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {[
                    {
                      id: 'local',
                      title: 'Локальная',
                      description: 'Silero на сервере, без доп. затрат',
                    },
                    {
                      id: 'elevenlabs',
                      title: 'ElevenLabs',
                      description: 'Облачный голос ElevenLabs',
                    },
                  ].map((option) => (
                    <button
                      key={option.id}
                      type="button"
                      disabled={loading}
                      onClick={() => setTtsProvider(option.id)}
                      className={`p-4 border rounded-xl text-left transition ${
                        ttsProvider === option.id
                          ? 'border-purple-600 bg-purple-50 ring-1 ring-purple-300'
                          : 'border-gray-200 hover:border-purple-400'
                      }`}
                    >
                      <div className="font-semibold text-gray-900">
                        {option.title}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        {option.description}
                      </div>
                    </button>
                  ))}
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  ElevenLabs требует активный интернет и API ключ, но даёт более
                  живой голос.
                </p>
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">
                  Состав голосов (для ElevenLabs)
                </label>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {[
                    {
                      id: 'male_duo',
                      label: '2 мужских',
                      description: 'Интервью с мужчинами',
                    },
                    {
                      id: 'mixed_duo',
                      label: 'Муж + Жен',
                      description: 'Смешанная пара',
                    },
                    {
                      id: 'female_duo',
                      label: '2 женских',
                      description: 'Интервью с женщинами',
                    },
                  ].map((option) => (
                    <button
                      key={option.id}
                      type="button"
                      disabled={loading || ttsProvider !== 'elevenlabs'}
                      onClick={() => setVoiceMix(option.id)}
                      className={`p-4 border rounded-xl text-left transition ${
                        voiceMix === option.id
                          ? 'border-purple-600 bg-purple-50 ring-1 ring-purple-300'
                          : 'border-gray-200 hover:border-purple-400'
                      } ${
                        ttsProvider !== 'elevenlabs'
                          ? 'opacity-60 cursor-not-allowed'
                          : ''
                      }`}
                    >
                      <div className="font-semibold text-gray-900">
                        {option.label}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        {option.description}
                      </div>
                    </button>
                  ))}
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  По умолчанию назначаются мужские голоса. Выберите другой
                  вариант, если есть спикер-женщина.
                </p>
              </div>

              <div>
                <label className="block text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">
                  Фоновый звук
                </label>
                <button
                  type="button"
                  disabled={loading}
                  onClick={() => setPreserveBackgroundAudio((prev) => !prev)}
                  className={`w-full text-left p-4 border rounded-xl transition ${
                    preserveBackgroundAudio
                      ? 'border-purple-600 bg-purple-50 ring-1 ring-purple-300'
                      : 'border-gray-200 hover:border-purple-400'
                  }`}
                >
                  <div className="font-semibold text-gray-900">
                    {preserveBackgroundAudio
                      ? '✅ Сохранять оригинальный фон (-20 dB)'
                      : '⬜️ Только новая озвучка'}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Если включить — подмешаем тихий оригинальный звук (‑20 dB),
                    чтобы оставить атмосферу площадки.
                  </div>
                </button>
              </div>

              {/* Number of Speakers */}
              <div>
                <label className="block text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">
                  Количество спикеров
                </label>
                <div className="grid grid-cols-4 gap-2">
                  {[
                    { id: 0, label: 'Авто', description: 'Определить автоматически' },
                    { id: 1, label: '1', description: 'Монолог' },
                    { id: 2, label: '2', description: 'Диалог' },
                    { id: 3, label: '3', description: 'Три спикера' },
                  ].map((option) => (
                    <button
                      key={option.id}
                      type="button"
                      disabled={loading}
                      onClick={() => setNumSpeakers(option.id)}
                      className={`p-3 border rounded-xl text-center transition ${
                        numSpeakers === option.id
                          ? 'border-purple-600 bg-purple-50 ring-1 ring-purple-300'
                          : 'border-gray-200 hover:border-purple-400'
                      }`}
                    >
                      <div className="font-semibold text-gray-900">
                        {option.label}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        {option.description}
                      </div>
                    </button>
                  ))}
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  Укажите количество спикеров для более точной диаризации. &quot;Авто&quot; — система определит сама.
                </p>
                
                {/* Speaker Change Time - only show when numSpeakers >= 2 */}
                {numSpeakers >= 2 && (
                  <div className="mt-4 p-3 bg-purple-50 rounded-xl border border-purple-200">
                    <label className="block text-xs font-bold text-purple-700 uppercase tracking-wider mb-2">
                      ⏱️ Время смены спикера (сек)
                    </label>
                    <input
                      type="text"
                      disabled={loading}
                      value={speakerChangeTime}
                      onChange={(e) => setSpeakerChangeTime(e.target.value)}
                      placeholder={numSpeakers === 2 ? "Например: 15" : "Например: 15, 30"}
                      className="w-full p-3 border border-purple-300 rounded-lg text-gray-900 placeholder-gray-400 focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                    />
                    <p className="text-xs text-purple-600 mt-2">
                      {numSpeakers === 2 
                        ? "Укажите секунду, когда меняется спикер. До этого времени — спикер 1, после — спикер 2."
                        : "Укажите секунды через запятую. Например: 15, 30 — три спикера."
                      }
                    </p>
                  </div>
                )}
              </div>

              {/* Tabs Navigation */}
              <div className="flex border-b border-gray-200">
                {[
                  { id: 'style', label: '✨ Анимация' },
                  { id: 'text', label: '🅰️ Текст и Фон' },
                  { id: 'position', label: '📍 Позиция' },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`py-2 px-4 text-sm font-medium border-b-2 transition-colors ${
                      activeTab === tab.id
                        ? 'border-purple-600 text-purple-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Tab Content */}
              <div className="min-h-[300px]">
                {/* TAB: STYLE (Animations) */}
                {activeTab === 'style' && (
                  <div className="animate-fadeIn">
                    <div className="grid grid-cols-2 gap-3">
                      {[
                        {
                          id: 'highlight',
                          label: 'Highlight',
                          description: 'Подсветка слова',
                        },
                        {
                          id: 'boxed',
                          label: 'Boxed',
                          description: 'Слова в рамках',
                        },
                        {
                          id: 'bounce_word',
                          label: 'Bounce Word',
                          description: 'Пружинящее',
                        },
                        {
                          id: 'readable',
                          label: 'Readable',
                          description: 'Все слова сразу',
                        },
                        {
                          id: 'fade',
                          label: 'Fade',
                          description: 'Прозрачность',
                        },
                        {
                          id: 'fade_short',
                          label: 'Fade Short',
                          description: '4 слова, 1 строка',
                        },
                        {
                          id: 'word_pop',
                          label: 'Word Pop',
                          description: 'Вылет из центра',
                        },
                        {
                          id: 'slide',
                          label: 'Slide Up',
                          description: 'Выезд снизу',
                        },
                        {
                          id: 'scale',
                          label: 'Scale',
                          description: 'Увеличение',
                        },
                        {
                          id: 'typewriter',
                          label: 'Typewriter',
                          description: 'Печатная машинка',
                        },
                        {
                          id: 'mask',
                          label: 'Mask Reveal',
                          description: 'Маска снизу',
                        },
                        {
                          id: 'simple_fade',
                          label: 'Simple Fade',
                          description: 'Мягкое',
                        },
                        {
                          id: 'spark',
                          label: 'Spark',
                          description: 'Свечение',
                        },
                        {
                          id: 'karaoke',
                          label: 'Karaoke',
                          description: 'Цвет',
                        },
                      ].map((option) => (
                        <button
                          key={option.id}
                          type="button"
                          disabled={loading}
                          onClick={() => setSubtitleAnimation(option.id)}
                          className={`btn-anim-hover p-3 border rounded-xl text-left transition group relative overflow-hidden ${
                            subtitleAnimation === option.id
                              ? 'border-purple-600 bg-purple-50 ring-1 ring-purple-600'
                              : 'border-gray-200 hover:border-purple-400 hover:bg-gray-50'
                          }`}
                        >
                          <div className="font-semibold text-gray-900 flex items-center">
                            {/* This span will animate on hover */}
                            <span
                              className={`anim-target anim-${option.id} inline-block`}
                            >
                              {option.label}
                            </span>
                          </div>
                          <div className="text-xs text-gray-500 mt-1">
                            {option.description}
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* TAB: TEXT (Font, Size, Bg) */}
                {activeTab === 'text' && (
                  <div className="space-y-6 animate-fadeIn">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Шрифт
                      </label>
                      <select
                        className="input-field"
                        value={subtitleFont}
                        disabled={loading}
                        onChange={(e) => setSubtitleFont(e.target.value)}
                        style={{
                          fontFamily: FONT_OPTIONS.find(
                            (f) => f.id === subtitleFont
                          )?.css,
                        }}
                      >
                        {FONT_OPTIONS.map((font) => (
                          <option
                            key={font.id}
                            value={font.id}
                            style={{ fontFamily: font.css }}
                          >
                            {font.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Размер текста
                      </label>
                      <div className="flex flex-wrap gap-2">
                        {FONT_SIZE_OPTIONS.map((size) => (
                          <button
                            key={size}
                            type="button"
                            disabled={loading}
                            onClick={() => setSubtitleFontSize(size)}
                            className={`px-4 py-2 rounded-lg border text-sm font-semibold transition ${
                              subtitleFontSize === size
                                ? 'border-purple-600 bg-purple-50 text-purple-700'
                                : 'border-gray-200 hover:border-purple-500 text-gray-700'
                            }`}
                          >
                            {size}px
                          </button>
                        ))}
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Эффекты читаемости
                      </label>
                      <div className="flex gap-3 flex-wrap">
                        <button
                          type="button"
                          disabled={loading}
                          onClick={() => setSubtitleGlow((prev) => !prev)}
                          className={`px-4 py-2 rounded-lg border text-sm font-semibold transition flex items-center justify-center gap-2 ${
                            subtitleGlow
                              ? 'border-purple-600 bg-purple-50 text-purple-700'
                              : 'border-gray-200 hover:border-purple-500 text-gray-700'
                          }`}
                        >
                          <span>
                            {subtitleGlow
                              ? '✨ Свечение'
                              : '⬜️ Свечение'}
                          </span>
                        </button>
                        <button
                          type="button"
                          disabled={loading}
                          onClick={() => setSubtitleBackground((prev) => !prev)}
                          className={`px-4 py-2 rounded-lg border text-sm font-semibold transition flex items-center justify-center gap-2 ${
                            subtitleBackground
                              ? 'border-purple-600 bg-purple-50 text-purple-700'
                              : 'border-gray-200 hover:border-purple-500 text-gray-700'
                          }`}
                        >
                          <span>
                            {subtitleBackground
                              ? '🔲 Бокс'
                              : '⬜️ Бокс'}
                          </span>
                        </button>
                        <button
                          type="button"
                          disabled={loading}
                          onClick={() => setSubtitleGradient((prev) => !prev)}
                          className={`px-4 py-2 rounded-lg border text-sm font-semibold transition flex items-center justify-center gap-2 ${
                            subtitleGradient
                              ? 'border-purple-600 bg-purple-50 text-purple-700'
                              : 'border-gray-200 hover:border-purple-500 text-gray-700'
                          }`}
                        >
                          <span>
                            {subtitleGradient
                              ? '🌑 Градиент'
                              : '⬜️ Градиент'}
                          </span>
                        </button>
                      </div>
                      <p className="text-xs text-gray-500 mt-2">
                        Свечение — тень вокруг текста. Бокс — прямоугольник. Градиент — затемнение снизу.
                      </p>
                    </div>

                  </div>
                )}

                {/* TAB: POSITION */}
                {activeTab === 'position' && (
                  <div className="animate-fadeIn">
                    <div className="grid grid-cols-1 gap-3">
                      {SUBTITLE_POSITIONS.map((preset) => (
                        <button
                          key={preset.id}
                          type="button"
                          disabled={loading}
                          onClick={() => setSubtitlePosition(preset.id)}
                          className={`p-4 border rounded-xl text-left transition flex items-center justify-between ${
                            subtitlePosition === preset.id
                              ? 'border-purple-600 bg-purple-50'
                              : 'border-gray-200 hover:border-purple-500'
                          }`}
                        >
                          <div>
                            <div className="font-semibold text-gray-900">
                              {preset.label}
                            </div>
                            <div className="text-sm text-gray-600">
                              {preset.description}
                            </div>
                          </div>
                          {subtitlePosition === preset.id && (
                            <span className="text-purple-600 text-xl">📍</span>
                          )}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Right Column: Sticky Preview */}
            <div className="lg:col-span-5">
              <SubtitlePreview
                text={previewText}
                positionId={subtitlePosition}
                fontFamily={
                  FONT_OPTIONS.find((opt) => opt.id === subtitleFont)?.css ||
                  '"Montserrat", sans-serif'
                }
                fontWeight={
                  FONT_OPTIONS.find((opt) => opt.id === subtitleFont)?.weight ||
                  400
                }
                fontSize={subtitleFontSize}
                animation={subtitleAnimation}
                showBackground={subtitleBackground}
                thumbnailUrl={videoThumbnail}
              />
            </div>
          </div>

          {/* Process Buttons */}
          <div className="mt-8 flex justify-end border-t pt-6">
            <div className="flex items-center gap-4">
              <p className="text-sm text-gray-600">
                Выбрано сегментов:{' '}
                <span className="font-bold text-gray-900">
                  {selectedSegments.length}
                </span>
              </p>
              
              {/* AI Dubbing Button */}
              {onDubbing && (
                <button
                  onClick={handleDubbing}
                  disabled={selectedSegments.length === 0 || loading}
                  className={`px-6 py-3 text-lg rounded-xl font-semibold transition shadow-lg ${
                    selectedSegments.length === 0 || loading
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      : 'bg-gradient-to-r from-amber-500 to-orange-500 text-white hover:from-amber-600 hover:to-orange-600'
                  }`}
                  title="ElevenLabs AI Dubbing - автоматический перевод с клонированием голоса"
                >
                  {loading ? (
                    <span className="flex items-center">
                      <svg className="animate-spin -ml-1 mr-2 h-5 w-5" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
                      </svg>
                      AI Дубляж...
                    </span>
                  ) : (
                    '🎬 AI Дубляж'
                  )}
                </button>
              )}
              
              {/* Standard Processing Button */}
              <button
                onClick={handleProcess}
                disabled={selectedSegments.length === 0 || loading}
                className={`btn-primary px-8 py-3 text-lg shadow-xl ${
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
                    Обработка...
                  </span>
                ) : (
                  `Создать клипы 🚀`
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
};

export default SegmentsList;

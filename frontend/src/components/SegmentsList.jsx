import React, { useMemo, useState } from 'react';

const CRITERIA_LABELS = {
  emotional_intensity: 'Эмоциональность',
  hook_potential: 'Сила начала',
  key_value: 'Ценность',
  story_moment: 'Сюжетность',
  humor: 'Юмор',
  dynamic_flow: 'Динамика',
  clip_worthiness: 'Годится для клипа',
};

const SUBTITLE_POSITIONS = [
  {
    id: 'mid_low',
    label: 'Чуть ниже центра',
    description: 'Сдвигаем текст чуть ниже центральной линии',
    previewStyle: {
      top: '48%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
    },
  },
  {
    id: 'lower_center',
    label: 'Нижняя треть',
    description: 'Классическая позиция ближе к нижней трети',
    previewStyle: {
      top: '58%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
    },
  },
  {
    id: 'lower_left',
    label: 'Левее центра',
    description: 'Субтитры смещены влево (герой справа)',
    previewStyle: { top: '63%', left: '30%' },
  },
  {
    id: 'lower_right',
    label: 'Правее центра',
    description: 'Смещаем блок вправо',
    previewStyle: { top: '63%', left: '70%' },
  },
  {
    id: 'bottom_center',
    label: 'Самый низ',
    description: 'Максимально низкое размещение',
    previewStyle: {
      top: '75%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
    },
  },
];

const FONT_OPTIONS = [
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
];

const FONT_SIZE_OPTIONS = [72, 82, 92, 102];

const PREVIEW_WIDTH_PX = 2556;
const PREVIEW_HEIGHT_PX = 1179;
const PREVIEW_RATIO_PERCENT = (PREVIEW_HEIGHT_PX / PREVIEW_WIDTH_PX) * 100;

const SubtitlePreview = ({
  text,
  positionId,
  fontFamily,
  fontSize,
  fontWeight,
  animation,
  thumbnailUrl,
}) => {
  const position =
    SUBTITLE_POSITIONS.find((preset) => preset.id === positionId)
      ?.previewStyle || SUBTITLE_POSITIONS[0].previewStyle;
  const previewFontSize = Math.round(fontSize * 0.6);
  const previewLines = useMemo(() => {
    const words = text.split(/\s+/).filter(Boolean);
    if (words.length === 0) return ['Текст субтитров'];
    const chunks = [];
    let current = [];
    const maxWordsPerLine = 4;
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
    ? 'relative mx-auto w-full rounded-xl overflow-hidden bg-black'
    : 'relative mx-auto w-full rounded-xl overflow-hidden bg-gradient-to-br from-gray-800 via-gray-700 to-gray-900';

  const containerStyle = {
    maxWidth: `${PREVIEW_WIDTH_PX}px`,
    paddingBottom: `${PREVIEW_RATIO_PERCENT}%`,
    backgroundImage: thumbnailUrl ? `url(${thumbnailUrl})` : undefined,
    backgroundSize: 'cover',
    backgroundPosition: 'center',
  };

  return (
    <div className="bg-gray-50 border rounded-2xl shadow-inner p-4">
      <div className="flex items-center justify-between mb-2">
        <p className="text-sm font-semibold text-gray-800">Превью макета</p>
        <span className="text-xs text-gray-500">9:16</span>
      </div>
      <div className={containerClass} style={containerStyle}>
        <div className="absolute inset-0 opacity-30 bg-[radial-gradient(circle_at_top,_rgba(255,255,255,0.35),_transparent_55%)]" />
        <div
          className={`absolute max-w-[80%] bg-white/10 backdrop-blur-md px-4 py-3 rounded-2xl text-center text-white font-semibold tracking-wide subtitle-preview-card preview-anim-${animation}`}
          style={{
            fontFamily: fontFamily,
            fontSize: `${previewFontSize}px`,
            lineHeight: 1.2,
            fontWeight,
            whiteSpace: 'nowrap',
            ...position,
          }}
        >
          {previewLines.map((line, idx) => (
            <span
              key={idx}
              className="block leading-tight"
              style={{
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
            >
              {line}
            </span>
          ))}
        </div>
      </div>
      <p className="text-xs text-gray-500 mt-3">
        Макет помогает оценить расположение текста и выбранный стиль до рендера.
      </p>
    </div>
  );
};

const SegmentsList = ({
  segments,
  videoTitle,
  onProcess,
  loading,
  videoThumbnail,
}) => {
  const [selectedSegments, setSelectedSegments] = useState([]);
  const [expandedSegments, setExpandedSegments] = useState([]);
  const [verticalMethod, setVerticalMethod] = useState('letterbox');
  const [subtitleAnimation, setSubtitleAnimation] = useState('bounce');
  const [subtitlePosition, setSubtitlePosition] = useState('mid_low');
  const [subtitleFont, setSubtitleFont] = useState('Montserrat Light');
  const [subtitleFontSize, setSubtitleFontSize] = useState(86);

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

  const handleProcess = () => {
    if (selectedSegments.length > 0) {
      onProcess(
        selectedSegments,
        verticalMethod,
        subtitleAnimation,
        subtitlePosition,
        subtitleFont,
        subtitleFontSize
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
    <div className="space-y-6 max-w-6xl mx-auto">
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">
              Найденные моменты
            </h2>
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
          {segments.map((segment, index) => {
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
                          Сегмент {index + 1}
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
                      {Object.entries(segment.criteria_scores)
                        .filter(([_, score]) => score > 0.6)
                        .slice(0, 5)
                        .map(([criterion, score]) => (
                          <span
                            key={criterion}
                            className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-gray-100 text-gray-700"
                          >
                            {CRITERIA_LABELS[criterion] ||
                              criterion.replace('_', ' ')}
                            : {(score * 100).toFixed(0)}%
                          </span>
                        ))}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-6 pt-6 border-t space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-5">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  📱 Формат для Reels/Shorts (9:16):
                </label>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {[
                    {
                      id: 'letterbox',
                      label: '⚫️ Чёрные поля',
                      description:
                        'Вписываем ролик без обрезки, добавляем поля сверху/снизу',
                    },
                    {
                      id: 'center_crop',
                      label: '✂️ Центр-кроп',
                      description: 'Обрезаем центр под 9:16',
                    },
                  ].map((method) => (
                    <button
                      key={method.id}
                      type="button"
                      disabled={loading}
                      onClick={() => setVerticalMethod(method.id)}
                      className={`p-4 border rounded-xl text-left transition ${
                        verticalMethod === method.id
                          ? 'border-purple-600 bg-purple-50'
                          : 'hover:border-purple-500'
                      } ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
                    >
                      <div className="font-semibold text-gray-900">
                        {method.label}
                      </div>
                      <div className="text-sm text-gray-600">
                        {method.description}
                      </div>
                    </button>
                  ))}
                </div>
                <p className="mt-1 text-xs text-gray-500">
                  Видео будет автоматически переведено в вертикальный формат
                  1080×1920
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  ✨ Анимация субтитров:
                </label>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {[
                    {
                      id: 'bounce',
                      label: '🔊 Bounce',
                      description: 'Пружинящее появление слов, как в CapCut',
                    },
                    {
                      id: 'slide',
                      label: '⬆️ Slide-up',
                      description: 'Плавный выезд снизу + мягкое проявление',
                    },
                    {
                      id: 'spark',
                      label: '✨ Spark',
                      description: 'Лёгкое свечение каждого слова',
                    },
                  ].map((option) => (
                    <button
                      key={option.id}
                      type="button"
                      disabled={loading}
                      onClick={() => setSubtitleAnimation(option.id)}
                      className={`p-4 border rounded-xl text-left transition ${
                        subtitleAnimation === option.id
                          ? 'border-purple-600 bg-purple-50'
                          : 'hover-border-purple-500'
                      } ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
                    >
                      <div className="font-semibold text-gray-900">
                        {option.label}
                      </div>
                      <div className="text-sm text-gray-600">
                        {option.description}
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  📍 Позиция субтитров:
                </label>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {SUBTITLE_POSITIONS.map((preset) => (
                    <button
                      key={preset.id}
                      type="button"
                      disabled={loading}
                      onClick={() => setSubtitlePosition(preset.id)}
                      className={`p-4 border rounded-xl text-left transition ${
                        subtitlePosition === preset.id
                          ? 'border-purple-600 bg-purple-50'
                          : 'hover-border-purple-500'
                      } ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
                    >
                      <div className="font-semibold text-gray-900">
                        {preset.label}
                      </div>
                      <div className="text-sm text-gray-600">
                        {preset.description}
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    🅰️ Шрифт:
                  </label>
                  <select
                    className="input-field"
                    value={subtitleFont}
                    disabled={loading}
                    onChange={(e) => setSubtitleFont(e.target.value)}
                  >
                    {FONT_OPTIONS.map((font) => (
                      <option key={font.id} value={font.id}>
                        {font.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    🔠 Размер:
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
                        } ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
                      >
                        {size}px
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

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
              thumbnailUrl={videoThumbnail}
            />
          </div>

          <div className="flex items-center justify-between">
            <p className="text-sm text-gray-600">
              Выбрано сегментов:{' '}
              <span className="font-semibold">{selectedSegments.length}</span>
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

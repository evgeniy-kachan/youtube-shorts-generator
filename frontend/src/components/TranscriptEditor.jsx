import React, { useState, useMemo, useCallback, useEffect } from 'react';

const SEGMENT_COLORS = [
  'rgba(241, 245, 249, 0.7)',   // slate-100
  'rgba(239, 246, 255, 0.7)',   // blue-50
  'rgba(240, 253, 244, 0.7)',   // green-50
  'rgba(254, 252, 232, 0.7)',   // yellow-50
  'rgba(255, 247, 237, 0.7)',   // orange-50
  'rgba(254, 242, 242, 0.7)',   // red-50
];

const getDurationColor = (seconds) => {
  if (seconds <= 60) return { bg: 'bg-green-100', text: 'text-green-700', emoji: '🟢' };
  if (seconds <= 90) return { bg: 'bg-yellow-100', text: 'text-yellow-700', emoji: '🟡' };
  if (seconds <= 120) return { bg: 'bg-orange-100', text: 'text-orange-700', emoji: '🟠' };
  return { bg: 'bg-red-100', text: 'text-red-700', emoji: '🔴' };
};

const formatTime = (seconds) => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

const TranscriptEditor = ({
  sentences = [],
  segments = [],
  onSegmentsChange,
  onClose,
}) => {
  // Local state for segment boundaries (sentence indices)
  // Each segment is defined by [startSentenceIdx, endSentenceIdx]
  const [segmentBoundaries, setSegmentBoundaries] = useState([]);
  const [selectedSegmentIdx, setSelectedSegmentIdx] = useState(null);
  
  // Initialize boundaries from segments
  useEffect(() => {
    if (segments.length > 0 && sentences.length > 0) {
      const boundaries = segments.map(seg => {
        // Find sentence indices that match segment times
        // startIdx: first sentence that starts at or after segment start
        let startIdx = sentences.findIndex(s => s.start >= seg.start_time - 0.5);
        if (startIdx === -1) startIdx = 0;
        
        // endIdx: last sentence that ends at or before segment end
        let endIdx = -1;
        for (let i = sentences.length - 1; i >= 0; i--) {
          if (sentences[i].end <= seg.end_time + 0.5) {
            endIdx = i;
            break;
          }
        }
        if (endIdx === -1) endIdx = sentences.length - 1;
        
        // Ensure startIdx <= endIdx
        if (startIdx > endIdx) {
          startIdx = endIdx;
        }
        
        return {
          id: seg.id,
          startIdx,
          endIdx,
          score: seg.highlight_score || 0,
          originalStart: seg.start_time,
          originalEnd: seg.end_time,
        };
      });
      setSegmentBoundaries(boundaries);
      console.log('Initialized boundaries:', boundaries);
    }
  }, [segments, sentences]);

  // Calculate segment info (duration, text) based on current boundaries
  const segmentInfos = useMemo(() => {
    return segmentBoundaries.map((boundary, idx) => {
      const segSentences = sentences.slice(boundary.startIdx, boundary.endIdx + 1);
      const startTime = segSentences[0]?.start || 0;
      const endTime = segSentences[segSentences.length - 1]?.end || 0;
      const duration = endTime - startTime;
      const text = segSentences.map(s => s.text).join(' ');
      
      return {
        ...boundary,
        startTime,
        endTime,
        duration,
        text,
        colorIdx: idx % SEGMENT_COLORS.length,
      };
    });
  }, [segmentBoundaries, sentences]);

  // Move segment start boundary
  const moveStart = useCallback((segIdx, direction, e) => {
    if (e) e.stopPropagation();
    console.log('moveStart called:', { segIdx, direction, sentencesLength: sentences.length });
    
    if (segIdx === null || segIdx === undefined) {
      console.log('Blocked: segIdx is null/undefined');
      return;
    }
    
    setSegmentBoundaries(prev => {
      const newBoundaries = [...prev];
      const current = newBoundaries[segIdx];
      if (!current) {
        console.log('No current boundary found for segIdx:', segIdx);
        return prev;
      }
      const newStartIdx = current.startIdx + direction;
      
      console.log('moveStart:', { current: current.startIdx, newStartIdx, endIdx: current.endIdx });
      
      // Constraints
      if (newStartIdx < 0) {
        console.log('Blocked: newStartIdx < 0');
        return prev;
      }
      if (newStartIdx > current.endIdx) {
        console.log('Blocked: newStartIdx > endIdx');
        return prev;
      }
      
      // Don't overlap with previous segment
      if (segIdx > 0 && newStartIdx <= newBoundaries[segIdx - 1].endIdx) {
        console.log('Blocked: would overlap with previous');
        return prev;
      }
      
      newBoundaries[segIdx] = { ...current, startIdx: newStartIdx };
      console.log('Success: new startIdx =', newStartIdx);
      return newBoundaries;
    });
  }, [sentences.length]);

  // Move segment end boundary
  const moveEnd = useCallback((segIdx, direction, e) => {
    if (e) e.stopPropagation();
    console.log('moveEnd called:', { segIdx, direction, sentencesLength: sentences.length });
    
    if (segIdx === null || segIdx === undefined) {
      console.log('Blocked: segIdx is null/undefined');
      return;
    }
    
    setSegmentBoundaries(prev => {
      const newBoundaries = [...prev];
      const current = newBoundaries[segIdx];
      if (!current) {
        console.log('No current boundary found for segIdx:', segIdx);
        return prev;
      }
      const newEndIdx = current.endIdx + direction;
      
      console.log('moveEnd:', { current: current.endIdx, newEndIdx, startIdx: current.startIdx });
      
      // Constraints
      if (newEndIdx >= sentences.length) {
        console.log('Blocked: newEndIdx >= sentences.length', sentences.length);
        return prev;
      }
      if (newEndIdx < current.startIdx) {
        console.log('Blocked: newEndIdx < startIdx');
        return prev;
      }
      
      // Don't overlap with next segment
      if (segIdx < newBoundaries.length - 1 && newEndIdx >= newBoundaries[segIdx + 1].startIdx) {
        console.log('Blocked: would overlap with next');
        return prev;
      }
      
      newBoundaries[segIdx] = { ...current, endIdx: newEndIdx };
      console.log('Success: new endIdx =', newEndIdx);
      return newBoundaries;
    });
  }, [sentences.length]);

  // Save changes
  const handleSave = useCallback(() => {
    const updatedSegments = segmentInfos.map(info => ({
      id: info.id,
      start_time: info.startTime,
      end_time: info.endTime,
      duration: info.duration,
    }));
    onSegmentsChange?.(updatedSegments);
    onClose?.();
  }, [segmentInfos, onSegmentsChange, onClose]);

  // Get segment index for a sentence
  const getSentenceSegment = useCallback((sentenceIdx) => {
    for (let i = 0; i < segmentBoundaries.length; i++) {
      const b = segmentBoundaries[i];
      if (sentenceIdx >= b.startIdx && sentenceIdx <= b.endIdx) {
        return i;
      }
    }
    return -1;
  }, [segmentBoundaries]);

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-5xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <div>
            <h2 className="text-xl font-bold text-gray-900">Редактор границ сегментов</h2>
            <p className="text-sm text-gray-500">Нажмите на сегмент для редактирования границ</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-600 hover:text-gray-900 transition"
            >
              Отмена
            </button>
            <button
              onClick={handleSave}
              className="px-6 py-2 bg-purple-600 text-white rounded-lg font-semibold hover:bg-purple-700 transition"
            >
              Сохранить
            </button>
          </div>
        </div>

        {/* Segment Legend */}
        <div className="flex items-center gap-4 px-4 py-2 bg-gray-50 border-b text-sm">
          <span className="font-medium text-gray-700">Длина:</span>
          <span className="flex items-center gap-1">🟢 30-60с</span>
          <span className="flex items-center gap-1">🟡 60-90с</span>
          <span className="flex items-center gap-1">🟠 90-120с</span>
          <span className="flex items-center gap-1">🔴 &gt;120с</span>
        </div>

        {/* Main Content */}
        <div className="flex flex-1 overflow-hidden">
          {/* Transcript with segments */}
          <div className="flex-1 overflow-y-auto p-4">
            <div className="space-y-0">
              {sentences.map((sentence, idx) => {
                const segIdx = getSentenceSegment(idx);
                const segInfo = segIdx >= 0 ? segmentInfos[segIdx] : null;
                const isSelected = segIdx === selectedSegmentIdx;
                const isSegmentStart = segInfo && idx === segInfo.startIdx;
                const isSegmentEnd = segInfo && idx === segInfo.endIdx;
                
                const bgColor = segInfo 
                  ? SEGMENT_COLORS[segInfo.colorIdx]
                  : 'transparent';
                
                return (
                  <div
                    key={idx}
                    className={`relative transition-all ${isSelected ? 'z-10' : ''}`}
                    onClick={() => segIdx >= 0 && setSelectedSegmentIdx(segIdx)}
                  >
                    {/* Segment start indicator */}
                    {isSegmentStart && (
                      <div className="flex items-center gap-2 py-1 px-2 -mx-2 rounded-t-lg" style={{ backgroundColor: bgColor }}>
                        <div className={`h-1 flex-1 rounded ${isSelected ? 'bg-purple-400' : 'bg-gray-300'}`} />
                        <span className={`text-xs font-medium ${isSelected ? 'text-purple-700' : 'text-gray-500'}`}>
                          Сегмент {segIdx + 1}
                        </span>
                        <div className={`h-1 flex-1 rounded ${isSelected ? 'bg-purple-400' : 'bg-gray-300'}`} />
                      </div>
                    )}
                    
                    {/* Sentence content */}
                    <div
                      className={`
                        px-3 py-1.5 cursor-pointer transition-all
                        ${isSelected ? 'ring-2 ring-purple-400 ring-inset' : ''}
                        ${isSegmentStart && isSelected ? 'border-l-4 border-purple-500' : ''}
                        ${isSegmentEnd && isSelected ? 'border-r-4 border-purple-500' : ''}
                      `}
                      style={{ backgroundColor: bgColor }}
                    >
                      <div className="flex items-start gap-2">
                        {/* Blinking cursor at start */}
                        {isSegmentStart && isSelected && (
                          <div className="w-0.5 h-6 bg-purple-500 animate-pulse flex-shrink-0" />
                        )}
                        
                        {/* Speaker label */}
                        {sentence.speaker && (
                          <span className="text-xs font-semibold text-purple-600 flex-shrink-0 mt-0.5">
                            [{sentence.speaker}]
                          </span>
                        )}
                        
                        {/* Text */}
                        <span className="text-sm text-gray-800 flex-1">
                          {sentence.text}
                        </span>
                        
                        {/* Time */}
                        <span className="text-xs text-gray-400 flex-shrink-0 mt-0.5">
                          {formatTime(sentence.start)}
                        </span>
                        
                        {/* Blinking cursor at end */}
                        {isSegmentEnd && isSelected && (
                          <div className="w-0.5 h-6 bg-purple-500 animate-pulse flex-shrink-0" />
                        )}
                      </div>
                    </div>
                    
                    {/* Segment end indicator with stats */}
                    {isSegmentEnd && segInfo && (
                      <div 
                        className="flex items-center justify-between py-1 px-2 -mx-2 rounded-b-lg mb-2"
                        style={{ backgroundColor: bgColor }}
                      >
                        {(() => {
                          const durationStyle = getDurationColor(segInfo.duration);
                          return (
                            <div className="flex items-center gap-3 w-full">
                              <div className={`h-0.5 flex-1 rounded ${isSelected ? 'bg-purple-400' : 'bg-gray-300'}`} />
                              <span className={`text-xs font-medium px-2 py-0.5 rounded ${durationStyle.bg} ${durationStyle.text}`}>
                                {durationStyle.emoji} {Math.round(segInfo.duration)}с
                              </span>
                              <span className="text-xs text-gray-500">
                                Score: {(segInfo.score * 100).toFixed(0)}%
                              </span>
                              <div className={`h-0.5 flex-1 rounded ${isSelected ? 'bg-purple-400' : 'bg-gray-300'}`} />
                            </div>
                          );
                        })()}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Right Panel - Selected Segment Controls */}
          {selectedSegmentIdx !== null && segmentInfos[selectedSegmentIdx] && (
            <div className="w-72 border-l bg-gray-50 p-4 overflow-y-auto">
              <div className="sticky top-0">
                <h3 className="font-bold text-gray-900 mb-4">
                  Сегмент {selectedSegmentIdx + 1}
                </h3>
                
                {(() => {
                  const info = segmentInfos[selectedSegmentIdx];
                  const durationStyle = getDurationColor(info.duration);
                  
                  return (
                    <div className="space-y-4">
                      {/* Duration */}
                      <div className={`p-3 rounded-lg ${durationStyle.bg}`}>
                        <div className="flex items-center justify-between">
                          <span className={`text-2xl font-bold ${durationStyle.text}`}>
                            {durationStyle.emoji} {Math.round(info.duration)}с
                          </span>
                          <span className="text-sm text-gray-600">
                            {formatTime(info.startTime)} - {formatTime(info.endTime)}
                          </span>
                        </div>
                      </div>
                      
                      {/* Score */}
                      <div className="p-3 bg-white rounded-lg border">
                        <span className="text-sm text-gray-500">Score</span>
                        <div className="text-xl font-bold text-gray-900">
                          {(info.score * 100).toFixed(0)}%
                        </div>
                      </div>
                      
                      {/* Start boundary controls */}
                      <div className="space-y-2">
                        <label className="text-xs font-bold text-gray-500 uppercase">
                          Начало сегмента
                        </label>
                        <div className="flex gap-2">
                          <button
                            onClick={(e) => moveStart(selectedSegmentIdx, -1, e)}
                            className="flex-1 py-2 px-3 bg-white border rounded-lg hover:bg-gray-50 transition font-medium"
                          >
                            ▲ Вверх
                          </button>
                          <button
                            onClick={(e) => moveStart(selectedSegmentIdx, 1, e)}
                            className="flex-1 py-2 px-3 bg-white border rounded-lg hover:bg-gray-50 transition font-medium"
                          >
                            ▼ Вниз
                          </button>
                        </div>
                        <p className="text-xs text-gray-500">
                          Предложение {info.startIdx + 1}
                        </p>
                      </div>
                      
                      {/* End boundary controls */}
                      <div className="space-y-2">
                        <label className="text-xs font-bold text-gray-500 uppercase">
                          Конец сегмента
                        </label>
                        <div className="flex gap-2">
                          <button
                            onClick={(e) => moveEnd(selectedSegmentIdx, -1, e)}
                            className="flex-1 py-2 px-3 bg-white border rounded-lg hover:bg-gray-50 transition font-medium"
                          >
                            ▲ Вверх
                          </button>
                          <button
                            onClick={(e) => moveEnd(selectedSegmentIdx, 1, e)}
                            className="flex-1 py-2 px-3 bg-white border rounded-lg hover:bg-gray-50 transition font-medium"
                          >
                            ▼ Вниз
                          </button>
                        </div>
                        <p className="text-xs text-gray-500">
                          Предложение {info.endIdx + 1}
                        </p>
                      </div>
                      
                      {/* Preview text */}
                      <div className="space-y-2">
                        <label className="text-xs font-bold text-gray-500 uppercase">
                          Текст сегмента
                        </label>
                        <div className="p-3 bg-white rounded-lg border text-sm text-gray-700 max-h-48 overflow-y-auto">
                          {info.text}
                        </div>
                      </div>
                    </div>
                  );
                })()}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TranscriptEditor;

import React, { useState, useMemo, useCallback, useEffect, useRef } from 'react';

const SEGMENT_COLORS = [
  'rgba(99, 102, 241, 0.15)',   // indigo - сегмент 1
  'rgba(16, 185, 129, 0.15)',   // emerald - сегмент 2
  'rgba(245, 158, 11, 0.15)',   // amber - сегмент 3
  'rgba(239, 68, 68, 0.15)',    // red - сегмент 4
  'rgba(139, 92, 246, 0.15)',   // violet - сегмент 5
  'rgba(6, 182, 212, 0.15)',    // cyan - сегмент 6
];

const SEGMENT_BORDER_COLORS = [
  'rgb(99, 102, 241)',   // indigo
  'rgb(16, 185, 129)',   // emerald
  'rgb(245, 158, 11)',   // amber
  'rgb(239, 68, 68)',    // red
  'rgb(139, 92, 246)',   // violet
  'rgb(6, 182, 212)',    // cyan
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
  const [segmentBoundaries, setSegmentBoundaries] = useState([]);
  const [selectedSegmentIdx, setSelectedSegmentIdx] = useState(0);
  const sentenceRefs = useRef({});
  
  // Initialize boundaries from segments
  useEffect(() => {
    if (segments.length > 0 && sentences.length > 0) {
      const boundaries = segments.map((seg, originalIdx) => {
        // Find sentence indices that OVERLAP with segment time range
        // Segment times come from DeepSeek which may have adjusted boundaries
        const segStart = seg.start_time;
        const segEnd = seg.end_time;
        
        let startIdx = -1;
        let endIdx = -1;
        
        // Find ALL sentences that overlap with segment time range
        // A sentence overlaps if its time range intersects with segment time range
        for (let i = 0; i < sentences.length; i++) {
          const s = sentences[i];
          // Sentence overlaps if: NOT (sentence ends before segment starts OR sentence starts after segment ends)
          // With 3 second tolerance to catch edge cases
          const overlaps = !(s.end < segStart - 3 || s.start > segEnd + 3);
          
          if (overlaps) {
            if (startIdx === -1) startIdx = i;
            endIdx = i;
          }
        }
        
        // Fallback if no overlap found
        if (startIdx === -1) {
          // Find closest sentence to segment start
          let minDist = Infinity;
          for (let i = 0; i < sentences.length; i++) {
            const dist = Math.abs(sentences[i].start - segStart);
            if (dist < minDist) {
              minDist = dist;
              startIdx = i;
            }
          }
          endIdx = startIdx;
        }
        
        return {
          id: seg.id,
          globalIndex: -1, // Will be set below
          tier: seg.tier || 'extended',
          startIdx,
          endIdx,
          score: seg.highlight_score || 0,
          originalStart: seg.start_time,
          originalEnd: seg.end_time,
        };
      });
      
      // Assign globalIndex matching SegmentsList numbering:
      // SegmentsList groups by tier (strict → extended → fallback)
      // and numbers them sequentially. We replicate this order.
      // segments array comes in original order from API.
      // SegmentsList splits into tiers keeping original array order within each tier.
      const tierOrder = { strict: 0, extended: 1, fallback: 2 };
      
      // Create indexed list to track original positions
      const indexed = segments.map((seg, i) => ({ 
        idx: i, 
        tier: seg.tier || 'extended' 
      }));
      
      // Sort by tier (strict first, then extended, then fallback)
      // Within same tier, keep original array order
      // IMPORTANT: use ?? not || because tierOrder['strict'] = 0 and (0 || 1) = 1 in JS!
      indexed.sort((a, b) => {
        const tierDiff = (tierOrder[a.tier] ?? 1) - (tierOrder[b.tier] ?? 1);
        if (tierDiff !== 0) return tierDiff;
        return a.idx - b.idx; // Preserve original order within tier
      });
      
      // Assign globalIndex: position in tier-sorted list + 1
      const globalIndexMap = {};
      indexed.forEach((item, pos) => {
        globalIndexMap[item.idx] = pos + 1;
      });
      
      // Apply to boundaries (which are in same order as segments at this point)
      boundaries.forEach((b, i) => {
        b.globalIndex = globalIndexMap[i] || (i + 1);
      });
      
      // Sort by start time (chronological order in transcript)
      // but keep original globalIndex for display
      boundaries.sort((a, b) => a.startIdx - b.startIdx);
      
      setSegmentBoundaries(boundaries);
      setSelectedSegmentIdx(0);
    }
  }, [segments, sentences]);

  // Track if we should scroll (only on segment selection, not on boundary changes)
  const shouldScrollRef = useRef(false);
  
  // Scroll to selected segment only when explicitly selecting a new segment
  useEffect(() => {
    if (shouldScrollRef.current && selectedSegmentIdx !== null && segmentBoundaries[selectedSegmentIdx]) {
      const startIdx = segmentBoundaries[selectedSegmentIdx].startIdx;
      const el = sentenceRefs.current[startIdx];
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
      shouldScrollRef.current = false;
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSegmentIdx]); // Intentionally exclude segmentBoundaries to prevent scroll on boundary changes
  
  // Function to select segment WITH scroll
  const selectSegmentWithScroll = useCallback((idx) => {
    shouldScrollRef.current = true;
    setSelectedSegmentIdx(idx);
  }, []);

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
    
    if (segIdx === null || segIdx === undefined) {
      return;
    }
    
    setSegmentBoundaries(prev => {
      const newBoundaries = [...prev];
      const current = newBoundaries[segIdx];
      if (!current) {
        return prev;
      }
      const newStartIdx = current.startIdx + direction;
      
      // Constraints
      if (newStartIdx < 0) {
        return prev;
      }
      if (newStartIdx > current.endIdx) {
        return prev;
      }
      
      // If expanding into previous segment (moving start earlier), push its end back
      if (direction < 0 && segIdx > 0) {
        const prevSeg = newBoundaries[segIdx - 1];
        if (newStartIdx <= prevSeg.endIdx) {
          // Push previous segment's end to before our new start
          const newPrevEndIdx = newStartIdx - 1;
          // But don't let previous segment become invalid (end < start)
          if (newPrevEndIdx < prevSeg.startIdx) {
            return prev; // Can't steal - previous segment would become empty
          }
          newBoundaries[segIdx - 1] = { ...prevSeg, endIdx: newPrevEndIdx };
        }
      }
      
      newBoundaries[segIdx] = { ...current, startIdx: newStartIdx };
      return newBoundaries;
    });
  }, []);

  // Move segment end boundary
  const moveEnd = useCallback((segIdx, direction, e) => {
    if (e) e.stopPropagation();
    
    if (segIdx === null || segIdx === undefined) {
      return;
    }
    
    setSegmentBoundaries(prev => {
      const newBoundaries = [...prev];
      const current = newBoundaries[segIdx];
      if (!current) {
        return prev;
      }
      const newEndIdx = current.endIdx + direction;
      
      // Constraints
      if (newEndIdx >= sentences.length) {
        return prev;
      }
      if (newEndIdx < current.startIdx) {
        return prev;
      }
      
      // If expanding into next segment, push its start forward
      if (direction > 0 && segIdx < newBoundaries.length - 1) {
        const nextSeg = newBoundaries[segIdx + 1];
        if (newEndIdx >= nextSeg.startIdx) {
          // Push next segment's start to after our new end
          const newNextStartIdx = newEndIdx + 1;
          // But don't let next segment become invalid (start > end)
          if (newNextStartIdx > nextSeg.endIdx) {
            return prev; // Can't steal - next segment would become empty
          }
          newBoundaries[segIdx + 1] = { ...nextSeg, startIdx: newNextStartIdx };
        }
      }
      
      newBoundaries[segIdx] = { ...current, endIdx: newEndIdx };
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

  // Build a lookup map for each sentence: which segments contain it, start here, end here
  const sentenceLookup = useMemo(() => {
    const lookup = {};
    for (let idx = 0; idx < sentences.length; idx++) {
      const containing = []; // segment indices that contain this sentence
      const starting = [];   // segment indices that START at this sentence
      const ending = [];     // segment indices that END at this sentence
      
      for (let i = 0; i < segmentBoundaries.length; i++) {
        const b = segmentBoundaries[i];
        if (idx >= b.startIdx && idx <= b.endIdx) {
          containing.push(i);
        }
        if (b.startIdx === idx) {
          starting.push(i);
        }
        if (b.endIdx === idx) {
          ending.push(i);
        }
      }
      
      // Primary segment = LATEST started one (last in containing array, since sorted by startIdx)
      const primary = containing.length > 0 ? containing[containing.length - 1] : -1;
      
      lookup[idx] = { containing, starting, ending, primary };
    }
    return lookup;
  }, [segmentBoundaries, sentences.length]);

  // Navigate between segments (with scroll)
  const goToPrevSegment = useCallback(() => {
    shouldScrollRef.current = true;
    setSelectedSegmentIdx(prev => Math.max(0, (prev || 0) - 1));
  }, []);

  const goToNextSegment = useCallback(() => {
    shouldScrollRef.current = true;
    setSelectedSegmentIdx(prev => Math.min(segmentBoundaries.length - 1, (prev || 0) + 1));
  }, [segmentBoundaries.length]);

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-6xl max-h-[95vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b bg-gradient-to-r from-purple-50 to-indigo-50">
          <div>
            <h2 className="text-xl font-bold text-gray-900">✂️ Редактор границ сегментов</h2>
            <p className="text-sm text-gray-500">Выберите сегмент слева → настройте границы справа</p>
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
              className="px-6 py-2 bg-purple-600 text-white rounded-lg font-semibold hover:bg-purple-700 transition shadow-lg"
            >
              💾 Сохранить
            </button>
          </div>
        </div>

        {/* Main Content - 3 columns */}
        <div className="flex flex-1 min-h-0">
          
          {/* LEFT: Segment List (quick navigation) */}
          <div className="w-56 border-r bg-gray-50 overflow-y-auto">
            <div className="p-3 border-b bg-white sticky top-0 z-10">
              <h3 className="font-semibold text-gray-700 text-sm">📋 Сегменты ({segmentBoundaries.length})</h3>
            </div>
            <div className="p-2 space-y-1">
              {segmentInfos.map((info, idx) => {
                const isSelected = idx === selectedSegmentIdx;
                const durationStyle = getDurationColor(info.duration);
                const borderColor = SEGMENT_BORDER_COLORS[info.colorIdx];
                
                return (
                  <button
                    key={info.id}
                    onClick={() => selectSegmentWithScroll(idx)}
                    className={`w-full text-left p-3 rounded-lg transition-all ${
                      isSelected 
                        ? 'bg-white shadow-md ring-2 ring-purple-400' 
                        : 'hover:bg-white hover:shadow-sm'
                    }`}
                    style={{ borderLeft: `4px solid ${borderColor}` }}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold text-gray-900">Сегмент {info.globalIndex}</span>
                      <span className={`text-xs px-1.5 py-0.5 rounded ${durationStyle.bg} ${durationStyle.text}`}>
                        {durationStyle.emoji} {Math.round(info.duration)}с
                      </span>
                    </div>
                    <div className="text-xs text-gray-500">
                      {formatTime(info.startTime)} - {formatTime(info.endTime)}
                    </div>
                    <div className="text-xs text-gray-400 mt-1 line-clamp-2">
                      {info.text.slice(0, 60)}...
                    </div>
                  </button>
                );
              })}
            </div>
        </div>

          {/* CENTER: Transcript with highlighted segments */}
          <div className="flex-1 overflow-y-auto p-4 pt-6 bg-white">
            <div className="max-w-3xl mx-auto pb-8">
              {sentences.map((sentence, idx) => {
                const info = sentenceLookup[idx] || { containing: [], starting: [], ending: [], primary: -1 };
                
                // Primary segment = latest started segment containing this sentence
                const primarySegInfo = info.primary >= 0 ? segmentInfos[info.primary] : null;
                const isInSelectedSegment = info.containing.includes(selectedSegmentIdx);
                
                // Show speaker only when it changes
                const prevSpeaker = idx > 0 ? sentences[idx - 1]?.speaker : null;
                const showSpeaker = sentence.speaker && sentence.speaker !== prevSpeaker;
                
                // Background from primary segment
                const bgColor = primarySegInfo ? SEGMENT_COLORS[primarySegInfo.colorIdx] : 'transparent';
                const leftBorderColor = primarySegInfo ? SEGMENT_BORDER_COLORS[primarySegInfo.colorIdx] : 'transparent';
                
                return (
                  <div
                    key={idx}
                    ref={el => sentenceRefs.current[idx] = el}
                    className="relative"
                    onClick={() => info.primary >= 0 && setSelectedSegmentIdx(info.primary)}
                  >
                    {/* END markers for segments ending at PREVIOUS sentence (render close before new open) */}
                    {/* This is handled below */}
                    
                    {/* START markers for ALL segments starting at this sentence */}
                    {info.starting.map(segIdx => {
                      const sInfo = segmentInfos[segIdx];
                      if (!sInfo) return null;
                      const sColor = SEGMENT_BORDER_COLORS[sInfo.colorIdx];
                      return (
                        <div
                          key={`start-${segIdx}`}
                          className="flex items-center gap-1.5 py-1 px-3 mt-2 cursor-pointer"
                          style={{ borderLeft: `3px solid ${sColor}` }}
                          onClick={(e) => { e.stopPropagation(); setSelectedSegmentIdx(segIdx); }}
                        >
                          <span 
                            className="text-xs font-bold px-1.5 py-0.5 rounded text-white"
                            style={{ backgroundColor: sColor }}
                          >
                            {sInfo.globalIndex}
                          </span>
                          <span className="text-xs text-gray-500">
                            {formatTime(sInfo.startTime)}-{formatTime(sInfo.endTime)}
                          </span>
                          {segIdx === selectedSegmentIdx && (
                            <span className="text-xs text-purple-600 ml-auto">✏️</span>
                          )}
                        </div>
                      );
                    })}
                    
                    {/* Speaker change indicator */}
                    {showSpeaker && (
                      <div 
                        className="px-3 py-0.5"
                        style={{ 
                          backgroundColor: bgColor,
                          borderLeft: primarySegInfo ? `3px solid ${leftBorderColor}` : '3px solid transparent',
                        }}
                      >
                        <span className="text-xs font-semibold text-indigo-500">
                          🎤 {sentence.speaker}
                        </span>
                      </div>
                    )}
                    
                    {/* Sentence content */}
                    <div
                      className="px-3 py-1 cursor-pointer"
                      style={{ 
                        backgroundColor: isInSelectedSegment ? 'rgba(139, 92, 246, 0.08)' : bgColor,
                        borderLeft: primarySegInfo ? `3px solid ${leftBorderColor}` : '3px solid transparent',
                      }}
                    >
                      <div className="flex items-start gap-2">
                        <span className="text-xs text-gray-300 font-mono w-5 flex-shrink-0 mt-0.5 text-right">
                          {idx + 1}
                        </span>
                        
                        {info.starting.includes(selectedSegmentIdx) && (
                          <span className="text-green-500 text-xs flex-shrink-0 mt-0.5">▶</span>
                        )}
                        
                        <span className={`text-sm flex-1 leading-relaxed ${isInSelectedSegment ? 'text-gray-900' : primarySegInfo ? 'text-gray-700' : 'text-gray-500'}`}>
                          {sentence.text}
                        </span>
                        
                        {info.ending.includes(selectedSegmentIdx) && (
                          <span className="text-red-500 text-xs flex-shrink-0 mt-0.5">◀</span>
                        )}
                        
                        <span className="text-xs text-gray-300 flex-shrink-0 mt-0.5 font-mono">
                          {formatTime(sentence.start)}
                        </span>
                      </div>
                    </div>
                    
                    {/* END markers for ALL segments ending at this sentence */}
                    {info.ending.map(segIdx => {
                      const eInfo = segmentInfos[segIdx];
                      if (!eInfo) return null;
                      const eColor = SEGMENT_BORDER_COLORS[eInfo.colorIdx];
                      return (
                        <div
                          key={`end-${segIdx}`}
                          className="flex items-center gap-2 py-0.5 px-3 mb-1 cursor-pointer"
                          style={{ borderLeft: `3px solid ${eColor}` }}
                          onClick={(e) => { e.stopPropagation(); setSelectedSegmentIdx(segIdx); }}
                        >
                          <span className="text-xs text-gray-400">
                            ── конец {eInfo.globalIndex} ──
                          </span>
                          <span className="text-xs text-gray-400">
                            {Math.round(eInfo.duration)}с
                          </span>
                        </div>
                      );
                    })}
                  </div>
                );
              })}
            </div>
          </div>

          {/* RIGHT: Controls Panel */}
          <div className="w-80 border-l bg-gray-50 overflow-y-auto">
            {selectedSegmentIdx !== null && segmentInfos[selectedSegmentIdx] && (() => {
                  const info = segmentInfos[selectedSegmentIdx];
                  const durationStyle = getDurationColor(info.duration);
              const borderColor = SEGMENT_BORDER_COLORS[info.colorIdx];
              
              return (
                <div className="p-4 space-y-4">
                  {/* Segment Navigation */}
                  <div className="flex items-center justify-between">
                    <button
                      onClick={goToPrevSegment}
                      disabled={selectedSegmentIdx === 0}
                      className="p-2 rounded-lg bg-white border hover:bg-gray-50 disabled:opacity-30 disabled:cursor-not-allowed"
                    >
                      ← Пред.
                    </button>
                    <span 
                      className="text-lg font-bold px-4 py-2 rounded-lg text-white"
                      style={{ backgroundColor: borderColor }}
                    >
                      Сегмент {info.globalIndex}
                    </span>
                    <button
                      onClick={goToNextSegment}
                      disabled={selectedSegmentIdx === segmentBoundaries.length - 1}
                      className="p-2 rounded-lg bg-white border hover:bg-gray-50 disabled:opacity-30 disabled:cursor-not-allowed"
                    >
                      След. →
                    </button>
                  </div>
                  
                  {/* Duration & Time */}
                  <div className={`p-4 rounded-xl ${durationStyle.bg}`}>
                        <div className="flex items-center justify-between">
                      <span className={`text-3xl font-bold ${durationStyle.text}`}>
                            {durationStyle.emoji} {Math.round(info.duration)}с
                          </span>
                      <div className="text-right">
                        <div className="text-sm font-medium text-gray-700">
                            {formatTime(info.startTime)} - {formatTime(info.endTime)}
                        </div>
                        <div className="text-xs text-gray-500">
                          Score: {(info.score * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Duration Legend */}
                  <div className="flex items-center justify-around text-xs bg-white rounded-lg p-2 border">
                    <span>🟢 30-60с</span>
                    <span>🟡 60-90с</span>
                    <span>🟠 90-120с</span>
                    <span>🔴 &gt;120с</span>
                      </div>
                      
                  {/* START Boundary Controls */}
                  <div className="bg-white rounded-xl p-4 border-2 border-green-200">
                    <div className="flex items-center gap-2 mb-3">
                      <span className="text-green-500 font-bold text-lg">▶</span>
                      <label className="text-sm font-bold text-green-700 uppercase">
                          Начало сегмента
                        </label>
                    </div>
                    <div className="flex gap-2 mb-2">
                          <button
                        onClick={(e) => moveStart(selectedSegmentIdx, -1, e)}
                        className="flex-1 py-3 px-4 bg-green-50 border-2 border-green-300 rounded-lg hover:bg-green-100 transition font-bold text-green-700"
                          >
                        ⬆️ Раньше
                          </button>
                          <button
                        onClick={(e) => moveStart(selectedSegmentIdx, 1, e)}
                        className="flex-1 py-3 px-4 bg-green-50 border-2 border-green-300 rounded-lg hover:bg-green-100 transition font-bold text-green-700"
                          >
                        ⬇️ Позже
                          </button>
                        </div>
                    <p className="text-xs text-gray-500 text-center">
                      Предложение #{info.startIdx + 1} • {formatTime(sentences[info.startIdx]?.start || 0)}
                        </p>
                      </div>
                      
                  {/* END Boundary Controls */}
                  <div className="bg-white rounded-xl p-4 border-2 border-red-200">
                    <div className="flex items-center gap-2 mb-3">
                      <span className="text-red-500 font-bold text-lg">◀</span>
                      <label className="text-sm font-bold text-red-700 uppercase">
                          Конец сегмента
                        </label>
                    </div>
                    <div className="flex gap-2 mb-2">
                          <button
                        onClick={(e) => moveEnd(selectedSegmentIdx, -1, e)}
                        className="flex-1 py-3 px-4 bg-red-50 border-2 border-red-300 rounded-lg hover:bg-red-100 transition font-bold text-red-700"
                          >
                        ⬆️ Раньше
                          </button>
                          <button
                        onClick={(e) => moveEnd(selectedSegmentIdx, 1, e)}
                        className="flex-1 py-3 px-4 bg-red-50 border-2 border-red-300 rounded-lg hover:bg-red-100 transition font-bold text-red-700"
                          >
                        ⬇️ Позже
                          </button>
                        </div>
                    <p className="text-xs text-gray-500 text-center">
                      Предложение #{info.endIdx + 1} • {formatTime(sentences[info.endIdx]?.end || 0)}
                        </p>
                      </div>
                      
                      {/* Preview text */}
                  <div className="bg-white rounded-xl p-4 border">
                    <label className="text-xs font-bold text-gray-500 uppercase block mb-2">
                      📝 Текст сегмента ({info.endIdx - info.startIdx + 1} предл.)
                        </label>
                    <div className="text-sm text-gray-700 max-h-40 overflow-y-auto leading-relaxed">
                          {info.text}
                        </div>
                      </div>
                    </div>
                  );
                })()}
              </div>
        </div>
      </div>
    </div>
  );
};

export default TranscriptEditor;

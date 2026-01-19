"""Analyze transcription to find interesting segments using LLM."""
import logging
import math
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

from backend.config import (
    DEEPSEEK_MODEL,
    HIGHLIGHT_CONCURRENT_REQUESTS,
    HIGHLIGHT_SEGMENTS_PER_CHUNK,
)
from backend.services.deepseek_client import DeepSeekClient

logger = logging.getLogger(__name__)


HIGHLIGHT_SCORE_THRESHOLD = 0.30   # Minimum score to be considered "good"
MIN_HIGHLIGHTS = 3                  # Always try to return at least this many
MAX_HIGHLIGHTS = 45                 # Never return more than this
FALLBACK_MIN_SCORE = 0.20           # Fallback segments must be at least this good
PREV_TOPIC_MAX_WORDS = 50
NEXT_TOPIC_MAX_WORDS = 30


class HighlightAnalyzer:
    """Analyze video segments to find interesting moments."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize analyzer with LLM.
        
        Args:
            model_name: DeepSeek model name (e.g., "deepseek-reasoner")
        """
        self.model_name = model_name or DEEPSEEK_MODEL
        self.client = DeepSeekClient(model=self.model_name)
        logger.info("Initialized HighlightAnalyzer with DeepSeek model: %s", self.model_name)
        
    def analyze_segments(
        self,
        segments: List[Dict],
        min_duration: int = 20,
        max_duration: int = 180,
        max_parallel: int = None,
    ) -> List[Dict]:
        """
        Analyze transcription segments to find highlights (with parallel processing).
        
        Args:
            segments: List of transcription segments with timestamps
            min_duration: Minimum segment duration in seconds
            max_duration: Maximum segment duration in seconds
            max_parallel: Maximum number of parallel LLM requests
            
        Returns:
            List of highlighted segments with scores
        """
        try:
            video_duration = segments[-1]['end'] if segments else 0.0
            logger.info(
                "Video duration %.1fs (~%.1f min), quality threshold=%.2f, fallback threshold=%.2f",
                video_duration,
                video_duration / 60 if video_duration else 0.0,
                HIGHLIGHT_SCORE_THRESHOLD,
                FALLBACK_MIN_SCORE,
            )
            # Merge consecutive Whisper segments into larger candidates to limit LLM calls
            potential_segments = self._merge_segments(
                segments,
                min_duration=min_duration,
                max_duration=max_duration,
                segments_per_chunk=HIGHLIGHT_SEGMENTS_PER_CHUNK,
            )
            self._attach_context_metadata(potential_segments, video_duration=video_duration)
            logger.info(
                "Prepared %d merged candidates from %d raw Whisper segments (chunk=%s)",
                len(potential_segments),
                len(segments),
                HIGHLIGHT_SEGMENTS_PER_CHUNK,
            )
            if not potential_segments:
                logger.warning("No candidate segments available for highlight analysis")
                return []
            
            max_parallel = max_parallel or HIGHLIGHT_CONCURRENT_REQUESTS
            logger.info(f"Starting parallel LLM analysis with {max_parallel} concurrent requests...")
            
            # Analyze segments in parallel
            analyzed_segments = self._analyze_segments_parallel(potential_segments, max_parallel)
            
            # Post-process: merge incomplete segments and re-score
            analyzed_segments = self._merge_incomplete_segments(
                analyzed_segments,
                max_duration=max_duration,
                max_parallel=max_parallel,
                original_segments=segments,  # Pass original for dynamic expansion
            )
            
            # Filter segments with good scores (>= HIGHLIGHT_SCORE_THRESHOLD)
            highlights = []
            scored_segments: list[tuple[int, dict, dict, float]] = []
            for i, (segment, scores) in enumerate(analyzed_segments):
                highlight_score = self._calculate_highlight_score(scores)
                scored_segments.append((i, segment, scores, highlight_score))

                if highlight_score >= HIGHLIGHT_SCORE_THRESHOLD:
                    highlights.append(self._build_highlight_payload(i, segment, scores, highlight_score))

            good_count = len(highlights)
            logger.info(
                "Found %d segments with score >= %.2f (threshold)",
                good_count, HIGHLIGHT_SCORE_THRESHOLD
            )

            # If we have fewer than MIN_HIGHLIGHTS, add fallbacks (but only if score >= FALLBACK_MIN_SCORE)
            if len(highlights) < MIN_HIGHLIGHTS and scored_segments:
                existing_ids = {item['id'] for item in highlights}
                fallback_candidates = sorted(
                    scored_segments,
                    key=lambda tpl: tpl[3],
                    reverse=True,
                )
                fallback_added = 0
                for idx, segment, scores, highlight_score in fallback_candidates:
                    # Stop if we've reached MIN_HIGHLIGHTS
                    if len(highlights) >= MIN_HIGHLIGHTS:
                        break
                    # Skip if already included
                    if f"segment_{idx}" in existing_ids:
                        continue
                    # Only add if score is at least FALLBACK_MIN_SCORE
                    if highlight_score < FALLBACK_MIN_SCORE:
                        logger.info(
                            "Stopping fallback: segment %d score %.2f < %.2f threshold",
                            idx, highlight_score, FALLBACK_MIN_SCORE
                        )
                        break
                    highlights.append(
                        self._build_highlight_payload(
                            idx,
                            segment,
                            scores,
                            highlight_score,
                            is_fallback=True,
                        )
                    )
                    existing_ids.add(f"segment_{idx}")
                    fallback_added += 1
                
                if fallback_added > 0:
                    logger.info(
                        "Added %d fallback segments (score >= %.2f) to reach minimum",
                        fallback_added, FALLBACK_MIN_SCORE
                    )
            
            # Sort by highlight score
            highlights.sort(key=lambda x: x['highlight_score'], reverse=True)

            # Cap at MAX_HIGHLIGHTS
            if MAX_HIGHLIGHTS and len(highlights) > MAX_HIGHLIGHTS:
                logger.info(
                    "Trimming highlight list from %d to top %d entries per MAX_HIGHLIGHTS.",
                    len(highlights),
                    MAX_HIGHLIGHTS,
                )
                highlights = highlights[:MAX_HIGHLIGHTS]
            
            logger.info(
                "Final result: %d highlights (%d good + %d fallback)",
                len(highlights), good_count, len(highlights) - good_count
            )
            return highlights
            
        except Exception as e:
            logger.error(f"Error analyzing segments: {e}", exc_info=True)
            raise
    
    def _analyze_segments_parallel(self, segments: List[Dict], max_parallel: int) -> List[tuple]:
        """Analyze segments in parallel using ThreadPoolExecutor."""
        
        def analyze_single(i, segment):
            logger.info(f"Analyzing segment {i+1}/{len(segments)}")
            scores = self._analyze_segment_with_llm(segment)
            return (segment, scores)
        
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = [
                executor.submit(analyze_single, i, segment)
                for i, segment in enumerate(segments)
            ]
            results = [future.result() for future in futures]
        
        return results

    def _merge_incomplete_segments(
        self,
        analyzed_segments: List[tuple],
        max_duration: int = 180,
        max_parallel: int = 4,
        original_segments: List[Dict] = None,
    ) -> List[tuple]:
        """
        Merge adjacent incomplete segments and re-analyze them.
        
        Logic:
        - If segment[i] has needs_next_context=True AND segment[i+1] has needs_previous_context=True
          → merge them into one segment and re-analyze
        - Also merge if segment[i].needs_next_context=True and scores suggest continuation
        - If needs_next but no adjacent chunk exists, try to expand from original_segments
        
        Args:
            analyzed_segments: List of (segment, scores) tuples
            max_duration: Maximum allowed duration for merged segments
            max_parallel: Number of parallel re-analysis threads
            original_segments: Original Whisper segments for dynamic expansion
        """
        if len(analyzed_segments) < 2:
            return analyzed_segments
        
        # Find merge candidates
        merge_pairs = []
        i = 0
        while i < len(analyzed_segments) - 1:
            seg_a, scores_a = analyzed_segments[i]
            seg_b, scores_b = analyzed_segments[i + 1]
            
            needs_next = scores_a.get('needs_next_context', False)
            needs_prev = scores_b.get('needs_previous_context', False)
            
            # Check if merged duration would be acceptable
            merged_duration = seg_b['end'] - seg_a['start']
            
            # Merge conditions:
            # 1. Direct link: A needs next AND B needs previous
            # 2. A needs next and B is low-scored (likely continuation)
            # 3. Both are low-scored and together might form a story
            should_merge = False
            merge_reason = ""
            
            if needs_next and needs_prev and merged_duration <= max_duration:
                should_merge = True
                merge_reason = "direct_link"
            elif needs_next and merged_duration <= max_duration:
                # A clearly needs continuation
                score_a = self._calculate_highlight_score(scores_a)
                score_b = self._calculate_highlight_score(scores_b)
                if score_a < 0.35 and score_b < 0.35:
                    should_merge = True
                    merge_reason = "both_low_score"
            elif needs_prev and merged_duration <= max_duration:
                # B clearly needs context from A
                score_a = self._calculate_highlight_score(scores_a)
                if score_a < 0.40:
                    should_merge = True
                    merge_reason = "b_needs_context"
            
            if should_merge:
                merge_pairs.append((i, i + 1, merge_reason))
                logger.info(
                    "MERGE CANDIDATE: segments %d+%d (%.1f-%.1fs, reason=%s, dur=%.1fs)",
                    i, i + 1, seg_a['start'], seg_b['end'], merge_reason, merged_duration
                )
                i += 2  # Skip the next segment as it's being merged
            else:
                i += 1
        
        # Handle edge case: last segment needs_next but has no neighbor
        # Try to expand it from original Whisper segments
        expand_candidates = []
        if original_segments:
            last_idx = len(analyzed_segments) - 1
            last_seg, last_scores = analyzed_segments[last_idx]
            
            if (last_scores.get('needs_next_context', False) and 
                last_idx not in [p[1] for p in merge_pairs] and  # Not already being merged
                not last_seg.get('is_video_end', False)):  # Not the actual end of video
                
                # Find original segments that come AFTER this chunk
                chunk_end = last_seg['end']
                extra_segments = [
                    s for s in original_segments 
                    if s['start'] >= chunk_end - 0.5  # Small overlap tolerance
                ]
                
                if extra_segments:
                    # Take enough segments to add ~30 seconds but not exceed max_duration
                    current_duration = last_seg['duration']
                    expansion_target = min(30, max_duration - current_duration)
                    
                    expanded_text_parts = [last_seg['text']]
                    expanded_end = chunk_end
                    expanded_words = list(last_seg.get('words', []))
                    expanded_dialogue = list(last_seg.get('dialogue', []))
                    
                    for extra_seg in extra_segments:
                        if extra_seg['end'] - last_seg['start'] > max_duration:
                            break
                        expanded_text_parts.append(extra_seg.get('text', ''))
                        expanded_end = extra_seg['end']
                        expanded_words.extend(extra_seg.get('words', []))
                        # Add to dialogue if different speaker or append to last turn
                        if expanded_dialogue and extra_seg.get('speaker') == expanded_dialogue[-1].get('speaker'):
                            expanded_dialogue[-1]['text'] += ' ' + extra_seg.get('text', '')
                            expanded_dialogue[-1]['end'] = extra_seg['end']
                        else:
                            expanded_dialogue.append({
                                'speaker': extra_seg.get('speaker'),
                                'text': extra_seg.get('text', ''),
                                'start': extra_seg['start'],
                                'end': extra_seg['end'],
                            })
                        
                        if expanded_end - last_seg['start'] >= current_duration + expansion_target:
                            break
                    
                    if expanded_end > chunk_end + 5:  # At least 5 seconds added
                        expanded_segment = {
                            **last_seg,
                            'end': expanded_end,
                            'duration': expanded_end - last_seg['start'],
                            'text': ' '.join(expanded_text_parts).strip(),
                            'words': expanded_words,
                            'dialogue': expanded_dialogue,
                            '_expanded_from': chunk_end,
                        }
                        expand_candidates.append((last_idx, expanded_segment))
                        logger.info(
                            "EXPAND CANDIDATE: segment %d expanded %.1f→%.1fs (+%.1fs) to get more context",
                            last_idx, chunk_end, expanded_end, expanded_end - chunk_end
                        )
        
        if not merge_pairs and not expand_candidates:
            logger.info("No incomplete segments to merge or expand")
            return analyzed_segments
        
        logger.info(f"Found {len(merge_pairs)} segment pairs to merge, {len(expand_candidates)} to expand")
        
        # Create merged segments
        segments_to_reanalyze = []
        merged_indices = set()
        
        for idx_a, idx_b, reason in merge_pairs:
            seg_a, _ = analyzed_segments[idx_a]
            seg_b, _ = analyzed_segments[idx_b]
            
            # Merge segment data
            merged_segment = self._merge_two_segments(seg_a, seg_b)
            segments_to_reanalyze.append(('merge', merged_segment, idx_a, idx_b))
            merged_indices.add(idx_a)
            merged_indices.add(idx_b)
        
        # Add expanded segments
        expanded_indices = set()
        for idx, expanded_seg in expand_candidates:
            segments_to_reanalyze.append(('expand', expanded_seg, idx, None))
            expanded_indices.add(idx)
        
        # Re-analyze all modified segments in parallel
        total_to_reanalyze = len(segments_to_reanalyze)
        logger.info(f"Re-analyzing {total_to_reanalyze} modified segments...")
        
        def reanalyze(item):
            op_type, seg, idx_a, idx_b = item
            new_scores = self._analyze_segment_with_llm(seg)
            new_highlight = self._calculate_highlight_score(new_scores)
            if op_type == 'merge':
                logger.info(
                    "MERGED %d+%d: new_score=%.2f, needs_prev=%s, needs_next=%s",
                    idx_a, idx_b, new_highlight,
                    new_scores.get('needs_previous_context'),
                    new_scores.get('needs_next_context'),
                )
            else:
                logger.info(
                    "EXPANDED %d: new_score=%.2f (was incomplete, added %.1fs)",
                    idx_a, new_highlight, seg['end'] - seg.get('_expanded_from', seg['start'])
                )
            return (op_type, seg, new_scores, idx_a, idx_b)
        
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            reanalyzed = list(executor.map(reanalyze, segments_to_reanalyze))
        
        # Build maps for quick lookup
        merge_map = {}  # (idx_a, idx_b) -> (seg, scores)
        expand_map = {}  # idx -> (seg, scores)
        
        for op_type, seg, scores, idx_a, idx_b in reanalyzed:
            if op_type == 'merge':
                merge_map[(idx_a, idx_b)] = (seg, scores)
            else:
                expand_map[idx_a] = (seg, scores)
        
        # Build final list
        result = []
        i = 0
        
        while i < len(analyzed_segments):
            if i in merged_indices:
                # Check if this is the start of a merged pair
                for idx_a, idx_b, _ in merge_pairs:
                    if idx_a == i:
                        merged_seg, merged_scores = merge_map[(idx_a, idx_b)]
                        result.append((merged_seg, merged_scores))
                        i = idx_b + 1  # Skip past the merged pair
                        break
                else:
                    # This index was part of a merge but not the start
                    i += 1
            elif i in expanded_indices:
                # Replace with expanded version
                expanded_seg, expanded_scores = expand_map[i]
                result.append((expanded_seg, expanded_scores))
                i += 1
            else:
                result.append(analyzed_segments[i])
                i += 1
        
        logger.info(
            "After processing: %d segments (was %d, merged=%d, expanded=%d)",
            len(result), len(analyzed_segments), len(merge_pairs), len(expand_candidates)
        )
        return result

    def _merge_two_segments(self, seg_a: Dict, seg_b: Dict) -> Dict:
        """Merge two segment dictionaries into one."""
        # Merge speakers
        speakers_a = seg_a.get('speakers', [])
        speakers_b = seg_b.get('speakers', [])
        merged_speakers = list(dict.fromkeys(speakers_a + speakers_b))  # Preserve order, remove dups
        
        # Merge dialogue
        dialogue_a = list(seg_a.get('dialogue', []))
        dialogue_b = list(seg_b.get('dialogue', []))
        
        # If last speaker of A == first speaker of B, merge those turns
        if dialogue_a and dialogue_b:
            if dialogue_a[-1].get('speaker') == dialogue_b[0].get('speaker'):
                last_turn = dialogue_a.pop()
                first_turn = dialogue_b.pop(0)
                merged_turn = {
                    "speaker": last_turn.get('speaker'),
                    "text": (last_turn.get('text', '') + ' ' + first_turn.get('text', '')).strip(),
                    "start": last_turn.get('start'),
                    "end": first_turn.get('end'),
                    "words": (last_turn.get('words') or []) + (first_turn.get('words') or []),
                }
                dialogue_a.append(merged_turn)
        
        merged_dialogue = dialogue_a + dialogue_b
        
        # Merge words
        words_a = seg_a.get('words', [])
        words_b = seg_b.get('words', [])
        merged_words = words_a + words_b
        
        # Determine primary speaker
        primary_speaker = seg_a.get('primary_speaker') or seg_b.get('primary_speaker', 'Unknown')
        
        return {
            'start': seg_a['start'],
            'end': seg_b['end'],
            'duration': seg_b['end'] - seg_a['start'],
            'text': (seg_a.get('text', '') + ' ' + seg_b.get('text', '')).strip(),
            'speakers': merged_speakers,
            'dialogue': merged_dialogue,
            'words': merged_words,
            'primary_speaker': primary_speaker,
            'prev_topic': seg_a.get('prev_topic', 'Unknown'),
            'next_topic': seg_b.get('next_topic', 'Unknown'),
            '_merged_from': [seg_a.get('start'), seg_b.get('start')],  # Debug info
        }
    
    def _create_time_windows(self, segments: List[Dict], min_duration: int, max_duration: int) -> List[Dict]:
        """
        Create overlapping time windows of several fixed sizes.
        This is a simpler, more robust method to generate candidate segments.
        """
        windows = []
        if not segments:
            return windows
            
        video_duration = segments[-1]['end']
        
        # Define a set of window sizes to try
        possible_sizes = [30, 45, 60, 90, 120, 180]
        window_sizes = [s for s in possible_sizes if min_duration <= s <= max_duration]
        
        # If the user-defined range is weird, add a size in the middle
        if not window_sizes:
            avg_size = (min_duration + max_duration) // 2
            if avg_size > 0:
                window_sizes.append(avg_size)

        for size in window_sizes:
            step = max(size, 1)  # non-overlapping windows
            max_start = max(int(video_duration - size), 0)
            start_positions = list(range(0, max_start + 1, step)) or [0]

            # Ensure we always cover the tail of the video
            if start_positions[-1] + size < video_duration:
                start_positions.append(max(video_duration - size, 0))
            
            for start_time_int in start_positions:
                start_time = float(start_time_int)
                end_time = min(start_time + size, video_duration)
                
                # Collect text from segments that fall within this window
                window_text_parts = []
                for seg in segments:
                    # Check for overlap between segment and window
                    if seg['start'] < end_time and seg['end'] > start_time:
                        window_text_parts.append(seg['text'])
                
                if window_text_parts:
                    windows.append({
                        'start': start_time,
                        'end': end_time,
                        'duration': size,
                        'text': " ".join(window_text_parts)
                    })
        
        # Remove duplicate windows (which can happen with this method)
        unique_windows = []
        seen_texts = set()
        for window in windows:
            if window['text'] not in seen_texts:
                unique_windows.append(window)
                seen_texts.add(window['text'])

        logger.info(
            "Created %d unique time windows with sizes %s (step equals window size)",
            len(unique_windows),
            window_sizes,
        )
        return unique_windows

    def _merge_segments(
        self,
        segments: List[Dict],
        min_duration: int,
        max_duration: int,
        segments_per_chunk: int,
    ) -> List[Dict]:
        """
        Merge consecutive Whisper segments so that DeepSeek receives fewer, denser requests.
        Also preserves dialogue structure (speaker turns) for multi-speaker TTS.
        """
        if not segments:
            return []

        merged: List[Dict] = []
        current: List[Dict] = []
        max_duration = max_duration or 180
        segments_per_chunk = max(1, segments_per_chunk)

        def build_chunk(chunk_segments: List[Dict]) -> Dict:
            start = chunk_segments[0]["start"]
            end = chunk_segments[-1]["end"]
            
            # Build full text for LLM
            text = " ".join(seg.get("text", "").strip() for seg in chunk_segments if seg.get("text"))
            duration = max(end - start, 0.0)
            
            # Extract unique speakers for metadata
            speakers = []
            seen_speakers = set()
            for seg in chunk_segments:
                spk = seg.get("speaker")
                if spk and spk not in seen_speakers:
                    speakers.append(spk)
                    seen_speakers.add(spk)
            
            # Build dialogue turns (group consecutive segments by same speaker)
            # Now also preserving word-level timestamps for phrase-level sync!
            dialogue = []
            
            # DIARIZATION DIAGNOSTIC: Log speaker changes in this chunk
            if chunk_segments:
                speaker_changes = []
                prev_speaker = None
                for seg_idx, seg in enumerate(chunk_segments):
                    seg_speaker = seg.get("speaker")
                    if seg_speaker != prev_speaker:
                        speaker_changes.append((seg_idx, seg_speaker, seg.get("text", "")[:30]))
                        prev_speaker = seg_speaker
                
                if len(speaker_changes) > 1:
                    logger.debug(
                        "DIARIZATION: Chunk has %d speaker changes: %s",
                        len(speaker_changes) - 1,
                        [(idx, spk, txt) for idx, spk, txt in speaker_changes]
                    )
                elif len(speaker_changes) == 1:
                    logger.debug(
                        "DIARIZATION: Chunk has single speaker: %s",
                        speaker_changes[0][1]
                    )
            
            if chunk_segments:
                first_seg = chunk_segments[0]
                current_turn = {
                    "speaker": first_seg.get("speaker"),
                    "text": first_seg.get("text", "").strip(),
                    "start": first_seg["start"],
                    "end": first_seg["end"],
                    "words": list(first_seg.get("words") or []),  # Word timestamps!
                }
                
                for seg in chunk_segments[1:]:
                    speaker = seg.get("speaker")
                    text_part = seg.get("text", "").strip()
                    seg_words = seg.get("words") or []
                    
                    if speaker == current_turn["speaker"]:
                        # Same speaker, append text and words
                        current_turn["text"] += " " + text_part
                        current_turn["end"] = seg["end"]
                        current_turn["words"].extend(seg_words)
                    else:
                        # New speaker, save current turn and start new
                        if current_turn["text"]:
                            dialogue.append(current_turn)
                        current_turn = {
                            "speaker": speaker,
                            "text": text_part,
                            "start": seg["start"],
                            "end": seg["end"],
                            "words": list(seg_words),  # Word timestamps for new turn
                        }
                
                if current_turn["text"]:
                    dialogue.append(current_turn)
            
            # Refine turn boundaries using word-level timestamps (more accurate than Pyannote)
            for turn in dialogue:
                words = turn.get("words") or []
                if words:
                    # Find actual first and last word timestamps
                    first_word_start = None
                    last_word_end = None
                    for w in words:
                        w_start = w.get("start")
                        w_end = w.get("end")
                        if w_start is not None and (first_word_start is None or w_start < first_word_start):
                            first_word_start = w_start
                        if w_end is not None and (last_word_end is None or w_end > last_word_end):
                            last_word_end = w_end
                    
                    # Update turn boundaries if word timestamps are available
                    old_start, old_end = turn["start"], turn["end"]
                    if first_word_start is not None:
                        turn["start"] = first_word_start
                    if last_word_end is not None:
                        turn["end"] = last_word_end
                    
                    # Log significant corrections
                    new_duration = turn["end"] - turn["start"]
                    old_duration = old_end - old_start
                    if abs(new_duration - old_duration) > 0.5:
                        logger.debug(
                            "Refined turn [%s]: %.1fs->%.1fs (was %.1fs, now %.1fs)",
                            turn.get("speaker", "?"),
                            old_start, turn["start"],
                            old_duration, new_duration,
                        )

            return {
                "start": start,
                "end": end,
                "duration": duration,
                "text": text.strip(),
                "speakers": speakers,
                "dialogue": dialogue,
                "words": [w for seg in chunk_segments for w in (seg.get("words") or [])]
            }

        def merge_with_previous(prev: Dict, extra: Dict) -> Dict:
            if not prev:
                return extra
            merged_speakers = list(set((prev.get("speakers") or []) + (extra.get("speakers") or [])))
            
            # Merge dialogues
            # If the last turn of prev and first turn of extra are same speaker, merge them
            new_dialogue = list(prev.get("dialogue", []))
            extra_dialogue = list(extra.get("dialogue", []))
            
            if new_dialogue and extra_dialogue and new_dialogue[-1]["speaker"] == extra_dialogue[0]["speaker"]:
                last = new_dialogue.pop()
                first = extra_dialogue.pop(0)
                merged_turn = {
                    "speaker": last["speaker"],
                    "text": (last["text"] + " " + first["text"]).strip(),
                    "start": last["start"],
                    "end": first["end"],
                    "words": (last.get("words") or []) + (first.get("words") or []),  # Merge words too!
                }
                new_dialogue.append(merged_turn)
            
            new_dialogue.extend(extra_dialogue)

            return {
                "start": prev["start"],
                "end": extra["end"],
                "duration": max(extra["end"] - prev["start"], 0.0),
                "text": f"{prev['text']} {extra['text']}".strip(),
                "speakers": merged_speakers,
                "dialogue": new_dialogue,
                "words": (prev.get("words") or []) + (extra.get("words") or [])
            }

        for seg in segments:
            current.append(seg)
            duration = current[-1]["end"] - current[0]["start"]
            reached_count = len(current) >= segments_per_chunk
            reached_duration = duration >= max_duration
            long_enough = duration >= min_duration

            if (reached_count or reached_duration) and long_enough:
                chunk = build_chunk(current)
                if chunk["text"]:
                    merged.append(chunk)
                current = []

        if current:
            chunk = build_chunk(current)
            if chunk["text"]:
                if merged and chunk["duration"] < min_duration:
                    merged[-1] = merge_with_previous(merged[-1], chunk)
                else:
                    merged.append(chunk)

        # Fallback: if merging collapsed everything into a single tiny chunk, reuse original window logic
        if not merged and segments:
            return self._create_time_windows(segments, min_duration, max_duration)

        return merged

    @staticmethod
    @staticmethod
    def _sanitize_topic_text(text: str | None, max_words: int) -> str:
        cleaned_words = (text or "").split()
        if not cleaned_words:
            return "Unknown"
        snippet = " ".join(cleaned_words[:max_words])
        if len(cleaned_words) > max_words:
            snippet += "..."
        return snippet

    def _attach_context_metadata(self, segments: list[dict], video_duration: float = 0) -> None:
        """
        Attach context metadata to each segment.
        
        Args:
            segments: List of merged segments
            video_duration: Total video duration to detect true end of video
        """
        for idx, segment in enumerate(segments):
            prev_text = segments[idx - 1]['text'] if idx > 0 else None
            next_text = segments[idx + 1]['text'] if idx + 1 < len(segments) else None
            
            segment['prev_topic'] = self._sanitize_topic_text(prev_text, PREV_TOPIC_MAX_WORDS)
            segment['next_topic'] = self._sanitize_topic_text(next_text, NEXT_TOPIC_MAX_WORDS)
            
            # Mark if this is truly the first/last segment of the video
            # (vs just the edge of our merged chunks list)
            is_first_segment = idx == 0 and segment['start'] < 5.0  # Within first 5 seconds
            is_last_segment = (
                idx + 1 >= len(segments) and 
                video_duration > 0 and 
                (video_duration - segment['end']) < 5.0  # Within last 5 seconds
            )
            
            segment['is_video_start'] = is_first_segment
            segment['is_video_end'] = is_last_segment

            primary_speaker = None
            if segment.get('speakers'):
                primary_speaker = segment['speakers'][0]
            if not primary_speaker and segment.get('dialogue'):
                primary_speaker = (segment['dialogue'][0] or {}).get('speaker')
            segment['primary_speaker'] = primary_speaker or "Unknown"

    @staticmethod
    def _build_highlight_payload(idx: int, segment: dict, scores: dict, highlight_score: float, is_fallback: bool = False) -> dict:
        payload = {
            'id': f"segment_{idx}",
            'start_time': segment['start'],
            'end_time': segment['end'],
            'duration': segment['duration'],
            'text': segment['text'],
            'speakers': segment.get('speakers', []),
            'dialogue': segment.get('dialogue', []),
            'words': segment.get('words', []),
            'highlight_score': highlight_score,
            'criteria_scores': scores,
            'needs_previous_context': scores.get('needs_previous_context', False),
            'needs_next_context': scores.get('needs_next_context', False),
        }
        if is_fallback:
            payload['is_fallback'] = True
        # Add debug info if segment was merged
        if segment.get('_merged_from'):
            payload['merged_from_starts'] = segment['_merged_from']
        return payload

    def _analyze_segment_with_llm(self, segment: Dict) -> Dict[str, float]:
        """Analyze a single segment using LLM."""
        
        prompt = f"""You are an editor who curates punchy clips from long-form podcasts and interviews (Diary of a CEO, Joe Rogan, Ali Abdaal, Huberman Lab, Lex Fridman). Guests include entrepreneurs, scientists, psychologists, doctors, athletes, authors, and thought leaders.

Judge this fragment as a potential 20–90 second short for viewers who crave:

BUSINESS & ENTREPRENEURSHIP:
- actionable business insights, growth tactics, metrics, frameworks, contrarian strategies
- founder stories with specific numbers, pivots, failures, lessons learned

SCIENCE & RESEARCH:
- fascinating scientific findings explained simply
- counterintuitive research results that challenge assumptions
- "here's what the data actually shows" moments

PSYCHOLOGY & SELF-IMPROVEMENT:
- insights about human behavior, motivation, relationships
- mental models, cognitive biases, decision-making frameworks
- therapy/coaching breakthroughs with practical applications

HEALTH & WELLNESS:
- evidence-based health advice (sleep, nutrition, exercise, longevity)
- specific protocols and routines with scientific backing
- personal health transformations with measurable results

PRODUCTIVITY & LEARNING:
- systems, tools, and methods for getting things done
- learning techniques, study methods, skill acquisition
- time management and energy optimization

PHILOSOPHY & LIFE:
- profound life lessons from real experience
- wisdom about relationships, purpose, meaning
- perspective shifts that change how you see the world

UNIVERSAL CRITERIA for high scores:
- Strong POV: bold opinions backed by experience/evidence
- Specificity: numbers, timeframes, concrete examples
- Complete arc: setup → insight → takeaway
- Emotional resonance: surprise, inspiration, validation, curiosity

Assume natural speech (~140 words per minute). Focus on fragments roughly 45–210 words long (≈20–90 s).

CONTEXT:
Previous topic: {segment.get('prev_topic', 'Unknown')}
Next topic: {segment.get('next_topic', 'Unknown')}
Speaker: {segment.get('primary_speaker', 'Unknown')}
Is video start: {segment.get('is_video_start', False)}
Is video end: {segment.get('is_video_end', False)}

IMPORTANT:
- Use ONLY the provided text. Do not invent context.
- Reward specificity: numbers, clear takeaways, step-by-step advice.
- Penalize fluff, clichés, or passages that require too much surrounding context.
- Prefer self-contained arcs (question → insight/answer).

HANDLING "UNKNOWN" CONTEXT:
• If "Previous topic" = "Unknown" BUT "Is video start" = False:
  → There IS previous content, we just didn't include it. Be careful with needs_previous_context.
• If "Next topic" = "Unknown" BUT "Is video end" = False:
  → There IS more content after this! If segment ends on a cliffhanger, continuation likely EXISTS.
  → Do NOT assume the story is unfinished — check "Is video end" first.
• If "Is video end" = True AND segment ends incomplete:
  → The video truly ends here. Final score will be capped at 0.20.

SEGMENT BOUNDARY DETECTION (CRITICAL):

Set "needs_previous_context": true if the text:
• STARTS with connectors: "And", "So", "That's why", "But", "Because", "Then", "He said", "She replied"
  (Russian: "И", "А", "Так что", "Поэтому", "Но", "Потому что", "Тогда", "Он сказал", "Она ответила")
• OR references something unexplained ("he did it", "the politician left", "that's when")
• OR feels like a continuation/punchline without setup
• OR starts mid-sentence or mid-thought
• OR (SEMANTIC CHECK) the "Previous topic" describes a story/event that THIS segment is clearly 
  a conclusion or lesson from. Example: prev="How I failed my first business" + text="Now I always 
  require prepayment" → the lesson loses impact without the failure story.

Set "needs_next_context": true if the text:
• ENDS mid-story (setup without punchline, buildup without payoff)
• OR ends with cliffhangers: "and then—", "that's when—", "what happened next..."
• OR poses a question that isn't answered in this segment
• OR ends with a teaser/promise without delivery
• OR the "Next topic" clearly contains the missing resolution
• OR (SEMANTIC CHECK) the segment is clearly a setup/problem statement and "Next topic" 
  contains the solution/answer. Example: text="I was losing $50K/month" + next="How I turned 
  it around" → the problem without the solution is incomplete.

When EITHER flag is true:
• Final score will be capped at 0.25 (incomplete segments are not standalone clips)
• The segment may still have high other scores if the content itself is good

CALIBRATION EXAMPLES (with new scoring system):

=== HIGH SCORE EXAMPLE (final ~0.82) ===

"I was $400K in debt at 28. Here's what saved me: I called every creditor 
and negotiated 60% settlements. Then I built a side business doing exactly 
what got me fired — but for myself. In 18 months I was debt-free."

Scores:
- surprise_novelty: 0.7 (unexpected: 60% debt reduction is possible)
- specificity_score: 0.9 ($400K, 60%, 18 months, age 28)
- personal_connection: 0.8 (personal failure/redemption story)
- actionability_score: 0.9 (clear steps: call creditors, negotiate, build business)
- clarity_simplicity: 0.8 (simple language, clear structure)
- completeness_arc: 0.9 (problem → actions → result)
- hook_quality: 0.85 ("$400K in debt at 28" — instantly grabs attention)
- needs_previous_context: false
- needs_next_context: false

=== MEDIUM SCORE EXAMPLE (final ~0.45) ===

"AI is going to change everything. We're already seeing it in our industry. 
Companies that don't adapt will be left behind. The question is not if, but when."

Scores:
- surprise_novelty: 0.2 (everyone says this)
- specificity_score: 0.1 (no numbers, no examples, no timeline)
- personal_connection: 0.1 (no personal story)
- actionability_score: 0.1 (no steps, just "adapt")
- clarity_simplicity: 0.7 (easy to understand)
- completeness_arc: 0.5 (has a point, but no story)
- hook_quality: 0.4 (generic opening)
- needs_previous_context: false
- needs_next_context: false

=== LOW SCORE EXAMPLE — Incomplete (final capped at 0.25) ===

"And said: 'Ready.' The politician left satisfied. Do you do the same 
with politicians? I noticed when I get involved in politics, it ends badly."

Scores:
- surprise_novelty: 0.3 (potentially interesting)
- specificity_score: 0.1 (no details)
- personal_connection: 0.2 (mentions personal experience vaguely)
- actionability_score: 0.0 (no advice)
- clarity_simplicity: 0.3 (confusing without context)
- completeness_arc: 0.2 (missing beginning)
- hook_quality: 0.1 ("And said" — terrible hook)
- needs_previous_context: TRUE (who said? about what?)
- needs_next_context: false
→ Final score CAPPED at 0.25 due to incomplete context

=== PLATITUDES EXAMPLE (final ~0.20) ===

"The key to success is persistence. Never give up on your dreams. 
Believe in yourself. You can do anything you set your mind to."

Scores:
- surprise_novelty: 0.0 (the opposite of surprising)
- specificity_score: 0.0 (zero specifics)
- personal_connection: 0.0 (no personal story)
- actionability_score: 0.1 (just "persist" and "believe")
- clarity_simplicity: 0.8 (easy to understand)
- completeness_arc: 0.3 (has a message, but no story)
- hook_quality: 0.2 (cliché opening)
- needs_previous_context: false
- needs_next_context: false

TEXT:
"{segment['text']}"

SCORING DIMENSIONS (each measures ONE specific thing):

1. surprise_novelty (0.0–1.0)
   What: Counterintuitive insight, challenges assumptions, reveals something unexpected
   NOT: Importance or usefulness of information
   High: "Sleeping MORE than 9 hours is as bad as sleeping less than 6"
   Low: "Sleep is important for health"

2. specificity_score (0.0–1.0)
   What: Concrete numbers, dates, names, measurable results, specific examples
   NOT: Quality of the idea or emotional delivery
   High: "$400K debt → 60% discount → 18 months → debt-free"
   Low: "I had a lot of debt and eventually paid it off"

3. personal_connection (0.0–1.0)
   What: Personal story, vulnerability, emotional narrative, relatable experience
   NOT: Practical usefulness or novelty
   High: "At my father's funeral, I realized no one mentioned his money..."
   Low: "Studies show people value relationships over wealth"

4. actionability_score (0.0–1.0)
   What: Clear instructions, steps, protocols, "do X, then Y, avoid Z"
   NOT: Importance of the problem or emotional weight
   High: "Call each creditor, ask for 60% reduction, document everything"
   Low: "You should try to negotiate with creditors"

5. clarity_simplicity (0.0–1.0)
   What: Accessible language, no jargon, well-structured, easy to follow
   NOT: Depth or originality of the thought
   High: Complex topic explained with everyday examples
   Low: Dense academic language, assumes prior knowledge

6. completeness_arc (0.0–1.0)
   What: Has beginning, middle, end. Problem → action → result/lesson
   NOT: Quality of each individual element
   High: "I was broke → did X → now I'm successful"
   Low: "Here's what I did..." (no setup or result)

7. hook_quality (0.0–1.0)
   What: ONLY the first 5-10 words. Does it grab attention immediately?
   NOT: Quality of the rest of the clip
   High: "Everything you know about diets is wrong"
   Low: "So, yeah, I was thinking about this..."

8. needs_previous_context (true/false)
   Does this segment require the previous one to make sense?

9. needs_next_context (true/false)
   Does this segment require the next one for a complete thought?

SCORING PROCESS:
1. First check needs_previous_context and needs_next_context
2. Then evaluate completeness_arc (most important for standalone clips)
3. Score remaining 6 criteria independently
4. Be strict: 0.8+ requires exceptional quality in that dimension

OUTPUT FORMAT:
Respond ONLY with valid JSON (no explanations, no markdown):
{{
  "surprise_novelty": 0.0,
  "specificity_score": 0.0,
  "personal_connection": 0.0,
  "actionability_score": 0.0,
  "clarity_simplicity": 0.0,
  "completeness_arc": 0.0,
  "hook_quality": 0.0,
  "needs_previous_context": false,
  "needs_next_context": false
}}"""

        try:
            response_json = self.client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an editor who scores transcript fragments for viral short-form video potential. "
                        "Deliver concise JSON scores between 0 and 1.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.25,
            )
            response_text = DeepSeekClient.extract_text(response_json)
            scores = DeepSeekClient.extract_json(response_text)
            
            # Define expected fields
            numeric_fields = [
                'surprise_novelty', 'specificity_score', 'personal_connection',
                'actionability_score', 'clarity_simplicity', 'completeness_arc', 'hook_quality'
            ]
            boolean_fields = ['needs_previous_context', 'needs_next_context']
            
            # Validate scores
            for key in scores:
                if key in boolean_fields:
                    if not isinstance(scores[key], bool):
                        scores[key] = str(scores[key]).lower() in ('true', '1', 'yes')
                elif key in numeric_fields:
                    if not isinstance(scores[key], (int, float)):
                        scores[key] = 0.0
                    scores[key] = max(0.0, min(1.0, float(scores[key])))
            
            # Ensure all required fields exist with defaults
            for field in numeric_fields:
                scores.setdefault(field, 0.5)
            for field in boolean_fields:
                scores.setdefault(field, False)
            
            return scores
            
        except Exception as e:
            logger.warning(f"Error parsing LLM response, using default scores: {e}")
            # Return neutral scores if parsing fails
            return {
                'surprise_novelty': 0.5,
                'specificity_score': 0.5,
                'personal_connection': 0.5,
                'actionability_score': 0.5,
                'clarity_simplicity': 0.5,
                'completeness_arc': 0.5,
                'hook_quality': 0.5,
                'needs_previous_context': False,
                'needs_next_context': False,
            }
    
    def _calculate_highlight_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted highlight score using the new orthogonal criteria system.
        
        Formula prioritizes:
        - completeness_arc (50%): A clip MUST tell a complete micro-story
        - Viral factors (40%): surprise, hook, personal connection
        - Value factors (10%): specificity, actionability, clarity
        
        Incomplete segments (needs_*_context) are capped at 0.25
        """
        # Base score calculation
        base_score = (
            # Most important: complete story arc
            0.30 * scores.get('completeness_arc', 0) +
            
            # Viral factors
            0.15 * scores.get('surprise_novelty', 0) +
            0.15 * scores.get('hook_quality', 0) +
            0.10 * scores.get('personal_connection', 0) +
            
            # Value factors
            0.10 * scores.get('specificity_score', 0) +
            0.10 * scores.get('actionability_score', 0) +
            0.10 * scores.get('clarity_simplicity', 0)
        )
        
        # Cap incomplete segments at 0.25
        if scores.get('needs_previous_context') or scores.get('needs_next_context'):
            return min(0.25, base_score)
        
        return min(1.0, base_score)

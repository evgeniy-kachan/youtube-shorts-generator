# -*- coding: utf-8 -*-
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


STRICT_SCORE_THRESHOLD = 0.35      # "Строгая выборка" - only high-quality segments
HIGHLIGHT_SCORE_THRESHOLD = 0.25   # "Расширенная выборка" - good segments
MIN_HIGHLIGHTS_BASE = 3             # Base minimum highlights
MIN_HIGHLIGHTS_PER_20MIN = 1        # Add +1 to minimum for every 20 minutes of video
MAX_HIGHLIGHTS = 60                 # Never return more than this
FALLBACK_MIN_SCORE = 0.15           # Fallback segments must be at least this good
PREV_TOPIC_MAX_WORDS = 50
NEXT_TOPIC_MAX_WORDS = 30

# Logical boundary detection settings
BOUNDARY_CHUNK_DURATION = 600  # 10 minutes per chunk for boundary detection
CHUNK_OVERLAP_DURATION = 60    # Overlap between chunks to avoid cutting mid-thought (1 minute)
MIN_SEGMENT_DURATION = 25      # Minimum segment duration after logical split
MAX_SEGMENT_DURATION = 60      # Maximum segment duration (IDEAL for Shorts/Reels)
MAX_MERGED_DURATION = 120      # Maximum duration when merging incomplete segments


def get_min_highlights(video_duration: float) -> int:
    """
    Calculate minimum highlights based on video duration.
    
    Formula: BASE + (video_minutes / 20)
    
    Examples:
    - 10 min → 3 + 0 = 3
    - 60 min → 3 + 3 = 6
    - 120 min → 3 + 6 = 9
    - 180 min → 3 + 9 = 12
    """
    if video_duration <= 0:
        return MIN_HIGHLIGHTS_BASE
    
    minutes = video_duration / 60.0
    dynamic_min = MIN_HIGHLIGHTS_BASE + int(minutes / 20)
    return dynamic_min


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
            min_highlights = get_min_highlights(video_duration)
            logger.info(
                "Video duration %.1fs (~%.1f min) -> min_highlights=%d, quality>=%.2f, fallback>=%.2f",
                video_duration,
                video_duration / 60 if video_duration else 0.0,
                min_highlights,
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
            
            # Adjust segment boundaries based on trim_first/last_sentences
            analyzed_segments = self._adjust_segment_boundaries(
                analyzed_segments,
                max_parallel=max_parallel,
            )
            
            # Post-process: merge incomplete segments and re-score
            analyzed_segments = self._merge_incomplete_segments(
                analyzed_segments,
                max_duration=max_duration,
                max_parallel=max_parallel,
                original_segments=segments,  # Pass original for dynamic expansion
            )
            
            # Filter segments into tiers:
            # - "strict": score >= 0.35 (high quality)
            # - "extended": score >= 0.25 (good quality)
            # - "fallback": score >= 0.15 (acceptable, only to fill min_highlights)
            highlights = []
            scored_segments: list[tuple[int, dict, dict, float]] = []
            strict_count = 0
            extended_count = 0
            
            for i, (segment, scores) in enumerate(analyzed_segments):
                highlight_score = self._calculate_highlight_score(scores)
                scored_segments.append((i, segment, scores, highlight_score))

                if highlight_score >= STRICT_SCORE_THRESHOLD:
                    # High quality - "strict" tier
                    highlights.append(self._build_highlight_payload(
                        i, segment, scores, highlight_score, tier="strict"
                    ))
                    strict_count += 1
                elif highlight_score >= HIGHLIGHT_SCORE_THRESHOLD:
                    # Good quality - "extended" tier
                    highlights.append(self._build_highlight_payload(
                        i, segment, scores, highlight_score, tier="extended"
                    ))
                    extended_count += 1

            logger.info(
                "Found %d segments: %d strict (>=%.2f), %d extended (>=%.2f)",
                len(highlights), strict_count, STRICT_SCORE_THRESHOLD,
                extended_count, HIGHLIGHT_SCORE_THRESHOLD
            )

            # If we have fewer than min_highlights, add fallbacks (but only if score >= FALLBACK_MIN_SCORE)
            if len(highlights) < min_highlights and scored_segments:
                existing_ids = {item['id'] for item in highlights}
                fallback_candidates = sorted(
                    scored_segments,
                    key=lambda tpl: tpl[3],
                    reverse=True,
                )
                fallback_added = 0
                for idx, segment, scores, highlight_score in fallback_candidates:
                    # Stop if we've reached min_highlights
                    if len(highlights) >= min_highlights:
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
                            tier="fallback",
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
            
            fallback_count = len([h for h in highlights if h.get('is_fallback')])
            logger.info(
                "Final result: %d highlights (%d strict + %d extended + %d fallback)",
                len(highlights), strict_count, extended_count, fallback_count
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

    def _adjust_segment_boundaries(
        self,
        analyzed_segments: List[tuple],
        max_parallel: int = 4,
    ) -> List[tuple]:
        """
        Adjust segment boundaries based on trim_first_sentences / trim_last_sentences.
        
        When DeepSeek detects that sentences belong to adjacent segments, this function
        moves them and re-analyzes the affected segments.
        """
        import re
        
        if len(analyzed_segments) < 2:
            return analyzed_segments
        
        # Collect adjustment requests
        adjustments = []  # List of (from_idx, to_idx, num_sentences, direction)
        
        for i, (segment, scores) in enumerate(analyzed_segments):
            trim_last = scores.get('trim_last_sentences', 0)
            trim_first = scores.get('trim_first_sentences', 0)
            
            # trim_last_sentences: move N sentences from end of segment[i] to start of segment[i+1]
            if trim_last > 0 and i + 1 < len(analyzed_segments):
                adjustments.append((i, i + 1, trim_last, 'last_to_next'))
                logger.info(
                    "BOUNDARY ADJUST: segment %d requests trim_last_sentences=%d (move to segment %d)",
                    i, trim_last, i + 1
                )
            
            # trim_first_sentences: move N sentences from start of segment[i] to end of segment[i-1]
            if trim_first > 0 and i > 0:
                adjustments.append((i, i - 1, trim_first, 'first_to_prev'))
                logger.info(
                    "BOUNDARY ADJUST: segment %d requests trim_first_sentences=%d (move to segment %d)",
                    i, trim_first, i - 1
                )
        
        if not adjustments:
            logger.info("No boundary adjustments requested by LLM")
            return analyzed_segments
        
        logger.info("Processing %d boundary adjustment requests", len(adjustments))
        
        # Helper: split text into sentences
        def split_sentences(text: str) -> List[str]:
            """Split text into sentences, handling common abbreviations."""
            pattern = r'(?<=[.!?])\s+(?=[A-ZА-ЯЁ])'
            sentences = re.split(pattern, text.strip())
            return [s.strip() for s in sentences if s.strip()]
        
        # Helper: find word boundaries for sentences
        def find_sentence_word_boundaries(words: List[Dict], sentences: List[str], from_start: bool, count: int) -> int:
            """Find the word index where sentence boundary occurs."""
            if not words or not sentences or count <= 0:
                return 0 if from_start else len(words)
            
            if from_start:
                # Find where first N sentences end
                target_sentences = sentences[:count]
                target_text = ' '.join(target_sentences)
                target_words = target_text.split()
                return min(len(target_words), len(words))
            else:
                # Find where last N sentences start
                target_sentences = sentences[-count:]
                target_text = ' '.join(target_sentences)
                target_words = target_text.split()
                return max(0, len(words) - len(target_words))
        
        # Apply adjustments - create modified segments
        modified_indices = set()
        segment_modifications = {}
        
        for from_idx, to_idx, num_sentences, direction in adjustments:
            from_seg, from_scores = analyzed_segments[from_idx]
            to_seg, to_scores = analyzed_segments[to_idx]
            
            from_text = from_seg.get('text', '')
            from_words = from_seg.get('words', [])
            from_sentences = split_sentences(from_text)
            
            if direction == 'last_to_next':
                # Move last N sentences from from_seg to start of to_seg
                if len(from_sentences) <= num_sentences:
                    logger.warning(
                        "BOUNDARY ADJUST: segment %d has only %d sentences, cannot trim %d",
                        from_idx, len(from_sentences), num_sentences
                    )
                    continue
                
                sentences_to_move = from_sentences[-num_sentences:]
                text_to_move = ' '.join(sentences_to_move)
                
                # Find word boundary
                word_boundary = find_sentence_word_boundaries(from_words, from_sentences, from_start=False, count=num_sentences)
                words_to_move = from_words[word_boundary:] if word_boundary < len(from_words) else []
                
                # Initialize modification dicts if needed
                if from_idx not in segment_modifications:
                    segment_modifications[from_idx] = {'trim_end': 0, 'trim_start': 0, 'prepend_text': '', 'append_text': '', 'prepend_words': [], 'append_words': []}
                if to_idx not in segment_modifications:
                    segment_modifications[to_idx] = {'trim_end': 0, 'trim_start': 0, 'prepend_text': '', 'append_text': '', 'prepend_words': [], 'append_words': []}
                
                segment_modifications[from_idx]['trim_end'] = num_sentences
                segment_modifications[to_idx]['prepend_text'] = text_to_move + ' ' + segment_modifications[to_idx].get('prepend_text', '')
                segment_modifications[to_idx]['prepend_words'] = words_to_move + segment_modifications[to_idx].get('prepend_words', [])
                
                modified_indices.add(from_idx)
                modified_indices.add(to_idx)
                
                logger.info(
                    "BOUNDARY ADJUST: moving %d sentences (%d words) from segment %d end to segment %d start: '%s...'",
                    num_sentences, len(words_to_move), from_idx, to_idx, text_to_move[:50]
                )
                
            elif direction == 'first_to_prev':
                # Move first N sentences from from_seg to end of to_seg (previous)
                if len(from_sentences) <= num_sentences:
                    logger.warning(
                        "BOUNDARY ADJUST: segment %d has only %d sentences, cannot trim %d",
                        from_idx, len(from_sentences), num_sentences
                    )
                    continue
                
                sentences_to_move = from_sentences[:num_sentences]
                text_to_move = ' '.join(sentences_to_move)
                
                # Find word boundary
                word_boundary = find_sentence_word_boundaries(from_words, from_sentences, from_start=True, count=num_sentences)
                words_to_move = from_words[:word_boundary] if word_boundary > 0 else []
                
                # Initialize modification dicts if needed
                if from_idx not in segment_modifications:
                    segment_modifications[from_idx] = {'trim_end': 0, 'trim_start': 0, 'prepend_text': '', 'append_text': '', 'prepend_words': [], 'append_words': []}
                if to_idx not in segment_modifications:
                    segment_modifications[to_idx] = {'trim_end': 0, 'trim_start': 0, 'prepend_text': '', 'append_text': '', 'prepend_words': [], 'append_words': []}
                
                segment_modifications[from_idx]['trim_start'] = num_sentences
                segment_modifications[to_idx]['append_text'] = segment_modifications[to_idx].get('append_text', '') + ' ' + text_to_move
                segment_modifications[to_idx]['append_words'] = segment_modifications[to_idx].get('append_words', []) + words_to_move
                
                modified_indices.add(from_idx)
                modified_indices.add(to_idx)
                
                logger.info(
                    "BOUNDARY ADJUST: moving %d sentences (%d words) from segment %d start to segment %d end: '%s...'",
                    num_sentences, len(words_to_move), from_idx, to_idx, text_to_move[:50]
                )
        
        if not modified_indices:
            return analyzed_segments
        
        # Apply modifications and rebuild segments
        modified_segments = []
        
        for i, (segment, scores) in enumerate(analyzed_segments):
            if i not in modified_indices:
                modified_segments.append((segment, scores, False))
                continue
            
            mods = segment_modifications.get(i, {})
            new_segment = dict(segment)
            
            # Get current text and words
            current_text = segment.get('text', '')
            current_words = list(segment.get('words', []))
            current_sentences = split_sentences(current_text)
            
            # Apply trim_start
            trim_start = mods.get('trim_start', 0)
            if trim_start > 0 and len(current_sentences) > trim_start:
                current_sentences = current_sentences[trim_start:]
                word_boundary = find_sentence_word_boundaries(current_words, split_sentences(current_text), from_start=True, count=trim_start)
                current_words = current_words[word_boundary:]
            
            # Apply trim_end
            trim_end = mods.get('trim_end', 0)
            if trim_end > 0 and len(current_sentences) > trim_end:
                current_sentences = current_sentences[:-trim_end]
                word_boundary = find_sentence_word_boundaries(current_words, current_sentences + [''] * trim_end, from_start=False, count=trim_end)
                current_words = current_words[:word_boundary] if word_boundary > 0 else current_words
            
            # Apply prepend
            prepend_text = mods.get('prepend_text', '').strip()
            prepend_words = mods.get('prepend_words', [])
            if prepend_text:
                current_sentences = split_sentences(prepend_text) + current_sentences
                current_words = prepend_words + current_words
            
            # Apply append
            append_text = mods.get('append_text', '').strip()
            append_words = mods.get('append_words', [])
            if append_text:
                current_sentences = current_sentences + split_sentences(append_text)
                current_words = current_words + append_words
            
            # Rebuild segment
            new_segment['text'] = ' '.join(current_sentences)
            new_segment['words'] = current_words
            
            # Update timing based on words if available
            if current_words:
                first_word_start = current_words[0].get('start')
                last_word_end = current_words[-1].get('end')
                if first_word_start is not None:
                    new_segment['start'] = first_word_start
                if last_word_end is not None:
                    new_segment['end'] = last_word_end
                new_segment['duration'] = new_segment['end'] - new_segment['start']
            
            new_segment['_boundary_adjusted'] = True
            modified_segments.append((new_segment, scores, True))
        
        # Skip re-analysis - use original scores (saves ~50% analysis time)
        # Boundary adjustments only move 1-2 sentences, score change is minimal
        if segments_to_reanalyze := [(i, seg) for i, (seg, _, needs) in enumerate(modified_segments) if needs]:
            logger.info("Skipping re-analysis of %d boundary-adjusted segments (using original scores)", len(segments_to_reanalyze))
            for i, (seg, old_scores, _) in enumerate(modified_segments):
                if any(idx == i for idx, _ in segments_to_reanalyze):
                    old_highlight = self._calculate_highlight_score(old_scores)
                    logger.info("BOUNDARY ADJUSTED segment %d: score %.2f (kept)", i, old_highlight)
        
        return [(seg, scores) for seg, scores, _ in modified_segments]

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
            
            # Merge conditions using DeepSeek's merge evaluation
            # Use MAX_MERGED_DURATION (120s) as hard limit for Shorts/Reels
            merge_limit = min(max_duration, MAX_MERGED_DURATION)
            
            should_merge = False
            merge_reason = ""
            
            # Get merge benefit hints from DeepSeek
            merge_benefit_a = scores_a.get('merge_benefit', 'none')
            merge_benefit_b = scores_b.get('merge_benefit', 'none')
            estimated_duration_a = scores_a.get('estimated_merged_duration', 0)
            merged_completeness_a = scores_a.get('merged_completeness_score', 0)
            
            # Condition 1: Direct link with DeepSeek approval
            if needs_next and needs_prev and merged_duration <= merge_limit:
                # DeepSeek says merge is beneficial
                if merge_benefit_a in ['high', 'medium'] or merge_benefit_b in ['high', 'medium']:
                    should_merge = True
                    merge_reason = "direct_link_approved"
                # Even without explicit approval, if both need context and duration is short
                elif merged_duration <= 90:
                    should_merge = True
                    merge_reason = "direct_link_short"
            
            # Condition 2: A needs next, DeepSeek recommends merge
            elif needs_next and merged_duration <= merge_limit:
                if merge_benefit_a == 'high':
                    should_merge = True
                    merge_reason = "a_needs_next_high"
                elif merge_benefit_a == 'medium' and merged_duration <= 90:
                    should_merge = True
                    merge_reason = "a_needs_next_medium"
                # Fallback: both low-scored (original logic)
                else:
                    score_a = self._calculate_highlight_score(scores_a)
                    score_b = self._calculate_highlight_score(scores_b)
                    if score_a < 0.35 and score_b < 0.35 and merged_duration <= 90:
                        should_merge = True
                        merge_reason = "both_low_score"
            
            # Condition 3: B needs previous context
            elif needs_prev and merged_duration <= merge_limit:
                if merge_benefit_b == 'high':
                    should_merge = True
                    merge_reason = "b_needs_prev_high"
                elif merge_benefit_b == 'medium' and merged_duration <= 90:
                    should_merge = True
                    merge_reason = "b_needs_prev_medium"
                # Fallback: A is low-scored
                else:
                    score_a = self._calculate_highlight_score(scores_a)
                    if score_a < 0.40 and merged_duration <= 90:
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
        
        # Combine scores without re-analyzing (faster, uses weighted average by duration)
        total_to_combine = len(segments_to_reanalyze)
        logger.info(f"Combining scores for {total_to_combine} modified segments (no re-analysis)...")
        
        reanalyzed = []
        for item in segments_to_reanalyze:
            op_type, seg, idx_a, idx_b = item
            
            if op_type == 'merge':
                # Combine scores from both segments using weighted average by duration
                _, scores_a = analyzed_segments[idx_a]
                _, scores_b = analyzed_segments[idx_b]
                seg_a, _ = analyzed_segments[idx_a]
                seg_b, _ = analyzed_segments[idx_b]
                
                dur_a = seg_a.get('duration', 60)
                dur_b = seg_b.get('duration', 60)
                total_dur = dur_a + dur_b
                weight_a = dur_a / total_dur
                weight_b = dur_b / total_dur
                
                # Weighted average for numeric scores
                new_scores = {}
                numeric_fields = [
                    'surprise_novelty', 'specificity_score', 'personal_connection',
                    'actionability_score', 'clarity_simplicity', 'completeness_arc', 'hook_quality'
                ]
                for field in numeric_fields:
                    val_a = scores_a.get(field, 0.25)
                    val_b = scores_b.get(field, 0.25)
                    new_scores[field] = val_a * weight_a + val_b * weight_b
                
                # For context flags: merged segment needs previous only if A needed it,
                # needs next only if B needed it
                new_scores['needs_previous_context'] = scores_a.get('needs_previous_context', False)
                new_scores['needs_next_context'] = scores_b.get('needs_next_context', False)
                new_scores['trim_first_sentences'] = 0
                new_scores['trim_last_sentences'] = 0
                
                new_highlight = self._calculate_highlight_score(new_scores)
                logger.info(
                    "MERGED %d+%d: combined_score=%.2f (was %.2f + %.2f)",
                    idx_a, idx_b, new_highlight,
                    self._calculate_highlight_score(scores_a),
                    self._calculate_highlight_score(scores_b),
                )
            else:
                # For expanded segments, keep original scores but mark as no longer needing context
                _, original_scores = analyzed_segments[idx_a]
                new_scores = dict(original_scores)
                new_scores['needs_next_context'] = False  # We expanded, so context is now included
                new_highlight = self._calculate_highlight_score(new_scores)
                logger.info(
                    "EXPANDED %d: score=%.2f (added %.1fs of context)",
                    idx_a, new_highlight, seg['end'] - seg.get('_expanded_from', seg['start'])
                )
        
            reanalyzed.append((op_type, seg, new_scores, idx_a, idx_b))
        
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

    def _detect_logical_boundaries(
        self,
        text: str,
        chunk_start_time: float,
        words: List[Dict],
    ) -> List[float]:
        """
        Use DeepSeek to detect logical boundaries in a text chunk.
        
        Args:
            text: Full text of the chunk (~10 min)
            chunk_start_time: Start time of this chunk in the video
            words: Word-level timestamps for precise boundary mapping
            
        Returns:
            List of timestamps where logical boundaries should be
        """
        if not text or not words:
            return []
        
        # Split text into sentences for DeepSeek to reference
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 3:
            return []  # Too short to split
        
        # Build numbered sentence list for the prompt
        numbered_sentences = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(sentences)
        )
        
        prompt = f"""Ты — редактор, который создаёт короткие клипы для YouTube Shorts / Instagram Reels из русскоязычных подкастов.

ТВОЯ ЗАДАЧА:
Найти логические границы в этом 10-минутном фрагменте транскрипции. Каждый получившийся сегмент станет отдельным коротким видео.

============================================================================
ЧАСТЬ 1: ПРАВИЛА ДЛИНЫ СЕГМЕНТОВ (КРИТИЧНО ДЛЯ SHORTS/REELS)
============================================================================

ИЕРАРХИЯ ДЛИНЫ (от лучшего к худшему):

1. ИДЕАЛ: 30-60 секунд (70-140 слов)
   → Лучшее вовлечение, максимальный охват, идеально для формата shorts
   → Стремись к этому по возможности

2. ДОПУСТИМО: 60-90 секунд (140-210 слов)
   → Всё ещё работает, но меньше вирусный потенциал
   → Только если мысль действительно требует такой длины

3. ТЕРПИМО: 90-120 секунд (210-280 слов)
   → Будет штрафоваться при финальной оценке
   → Используй ТОЛЬКО для исключительного контента, который нельзя разделить

4. НЕДОПУСТИМО: >120 секунд
   → НЕ создавай сегменты длиннее 120 секунд ни при каких обстоятельствах

ИСКЛЮЧЕНИЯ ДЛЯ ПОЛНЫХ АРГУМЕНТОВ:
Полные аргументы (тезис + доказательства + вывод) могут быть до 120 секунд, НО:
- 60-90 секунд: идеально для большинства аргументов
- 90-120 секунд: допустимо ТОЛЬКО если доказательства детальны и необходимы
- Аргумент ДОЛЖЕН быть действительно полным (вопрос → ответ → доказательство → вывод)

ПРАВИЛО ПРЕДПОЧТЕНИЯ:
Если можешь разделить 90-секундный сегмент на 45+45 без нарушения логики — ВСЕГДА ДЕЛАЙ ЭТО.
Короткие законченные мысли лучше длинных.

============================================================================
ЧАСТЬ 2: ГДЕ ДЕЛИТЬ
============================================================================

Дели когда:
- Тема явно меняется («Теперь поговорим о...», «Переходим к...»)
- Начинается новый пример или история
- Завершается цикл вопрос → ответ
- Меняется спикер (если несколько собеседников)

============================================================================
ЧАСТЬ 3: ГДЕ НЕ ДЕЛИТЬ (КРИТИЧНО)
============================================================================

НИКОГДА не дели в этих случаях:

1. ВВЕДЕНИЕ ТЕМЫ + ОБЪЯСНЕНИЕ (ВИСЯЩАЯ ЗАВЯЗКА)
   Если предложение N вводит человека/концепцию, а предложение N+1 объясняет — они ДОЛЖНЫ быть вместе.

   Примеры:
   «Его специализация — высокотемпературная сверхпроводимость. В этой области вышло 50 тысяч статей...»
   → Тема введена в предложении 1, объяснена в предложении 2 → ОДИН СЕГМЕНТ.

2. СТРУКТУРА АРГУМЕНТА (КРИТИЧНО)
   Когда видишь такой паттерн:
   - Вопрос или смелое утверждение
   - Ответ
   - Доказательства/примеры в поддержку
   
   Это ОДНА логическая единица. НЕ дели между этими частями.

   Пример ХОРОШО (держать вместе):
   «Веришь в симуляцию? — Высокий процент. — Посмотри на видеоигры за 50 лет...»
   → Вопрос → ответ → доказательство → ОДИН СЕГМЕНТ

   Типичная ошибка (ПЛОХО):
   Деление между ответом и его доказательством
   → Первый сегмент: только вопрос + ответ (неполный)
   → Второй сегмент: только доказательство (без контекста)

3. СЕРЕДИНА ИСТОРИИ
   Никогда не режь посреди примера, анекдота или личной истории
   Дождись развязки или вывода

4. ЗАВЯЗКА + РАЗВЯЗКА
   Если предложение задаёт вопрос, а следующее отвечает — держи вместе

5. УСЛОВИЕ + РЕЗУЛЬТАТ
   «Если сделаешь X, то произойдёт Y» — держи вместе

6. ВОПРОС + ОТВЕТ (КРИТИЧНО — НИКОГДА НЕ ДЕЛИ)
   Если видишь ВОПРОС, за которым следует ОТВЕТ — они ДОЛЖНЫ быть в ОДНОМ сегменте.
   
   Паттерн для обнаружения:
   - Предложение заканчивается на «?»
   - Следующие 1-3 предложения — это ответ
   - Ответ может быть очень коротким («Да», «Нет», «Высокий процент», «Довольно высокий»)
   
   НИКОГДА не ставь границу между вопросом и его ответом!
   
   Примеры:
   ПЛОХО: «Какой процент вы бы дали?» | «Довольно высокий.»
   ХОРОШО: «Какой процент вы бы дали? Довольно высокий.» (один сегмент)
   
   ПЛОХО: «Вы верите в симуляцию?» | «Да, довольно высокий процент.»
   ХОРОШО: «Вы верите в симуляцию? Да, довольно высокий процент.» (один сегмент)

============================================================================
ЧАСТЬ 4: ОБРАБОТКА СПИСКОВ
============================================================================

Когда спикер перечисляет несколько идей/примеров:

- Если каждая идея ОБЪЯСНЯЕТСЯ (30+ секунд объяснения) → отдельные сегменты
- Если идеи просто перечислены кратко без объяснения → держи как один сегмент

Примеры:
НЕПРАВИЛЬНО:
«Вот идеи: А. Б. В. Г. Д.» — каждая просто упомянута
→ Это ОДИН сегмент (просто список)

ПРАВИЛЬНО:
«Первая идея: [объяснение 30 сек]. Вторая идея: [объяснение 30 сек]»
→ Это ДВА сегмента, ставь границу между ними

============================================================================
ЧАСТЬ 5: РЕТРОСПЕКТИВНЫЕ ССЫЛКИ — СИГНАЛЫ НОВОЙ ТЕМЫ (КРИТИЧНО)
============================================================================

Когда спикер ссылается НАЗАД на предыдущее обсуждение, чтобы НАЧАТЬ новую тему,
фраза-ссылка ПРИНАДЛЕЖИТ НОВОМУ СЕГМЕНТУ.

МАРКЕРЫ:
- «Вы ранее говорили о...»
- «Ты упоминал...»
- «Как вы сказали ранее...»
- «Мы раньше обсуждали...»
- «Возвращаясь к разговору о...»
- «Помните, вы говорили про...»
- «Вы упоминали о...»

КРИТИЧЕСКОЕ ПРАВИЛО:
Эти фразы ВВОДЯТ смену темы.
Граница должна быть поставлена ПЕРЕД такой фразой.

ПРИМЕР:
ХОРОШЕЕ деление:
Сегмент 4: «...долг станет менее серьёзной проблемой. Да, скорее всего.»
Сегмент 5: «Вы ранее говорили о симуляции. Я люблю «Матрицу»...»

ПЛОХОЕ деление:
Сегмент 4: «...долг станет менее серьёзной проблемой. Да, скорее всего. Вы ранее говорили о симуляции.»
Сегмент 5: «Я люблю «Матрицу»...»

ПОЧЕМУ ХОРОШЕЕ правильно:
- «Вы ранее говорили» вводит НОВУЮ тему (симуляция)
- Зритель сразу понимает, о чём речь
- Ссылка служит контекстом для новой темы

ПОЧЕМУ ПЛОХОЕ неправильно:
- Ссылка застряла в неправильном сегменте
- Сегмент 5 начинается без контекста («Я люблю «Матрицу»» — какую Матрицу?)
- Слушатель в замешательстве

============================================================================
ЧАСТЬ 6: ТЕКСТ ДЛЯ АНАЛИЗА
============================================================================

ПРИМЕЧАНИЕ: Предложения в начале этого фрагмента могут быть продолжением предыдущего фрагмента.
Если видишь фразы типа «Вы ранее говорили» или «Как я сказал» В НАЧАЛЕ,
проверь, объясняется ли ссылка ВНУТРИ этого фрагмента.
Если нет, считай ПЕРВЫЕ 1-2 предложения контекстными и держи их с этим фрагментом.

Вот предложения с номерами. Каждое предложение примерно 2-3 секунды.
Обращай внимание на логические связи, а не только на границы предложений.

ПРЕДЛОЖЕНИЯ:
{numbered_sentences}

============================================================================
ЧАСТЬ 7: ТВОЙ ВЫВОД — СТРОГИЙ ФОРМАТ
============================================================================

Верни ТОЛЬКО номера предложений через запятую, где должны НАЧИНАТЬСЯ новые сегменты.

ПРАВИЛА:
- Числа должны быть целыми от 2 до {len(sentences)}
- НЕ включай предложение 1 (первый сегмент всегда начинается там)
- Числа должны быть в порядке возрастания
- Через запятую (пробелы опционально)

Примеры:
✅ 4, 8, 15, 23
✅ 4,8,15,23

Если границы не нужны: верни точно «NONE»

ГРАНИЦЫ:"""

        try:
            response = self.client.generate(prompt, max_tokens=100, temperature=0.3)
            response_text = response.strip()
            
            if response_text.upper() == "NONE" or not response_text:
                return []
            
            # Parse sentence numbers
            boundary_sentences = []
            for part in response_text.replace(",", " ").split():
                try:
                    num = int(part.strip())
                    if 1 <= num <= len(sentences):
                        boundary_sentences.append(num)
                except ValueError:
                    continue
            
            if not boundary_sentences:
                return []
            
            # Map sentence numbers to timestamps using word positions
            boundary_times = []
            current_word_idx = 0
            current_sentence_idx = 0
            
            for sentence_num in sorted(set(boundary_sentences)):
                # Find the start of this sentence in words
                target_sentence = sentences[sentence_num - 1] if sentence_num <= len(sentences) else None
                if not target_sentence:
                    continue
                
                # Find first word of this sentence
                first_words = target_sentence.split()[:3]  # First 3 words
                first_words_lower = [w.lower().strip('.,!?') for w in first_words]
                
                # Search in words array
                for i in range(current_word_idx, len(words)):
                    word_text = (words[i].get('word') or words[i].get('text', '')).lower().strip('.,!?')
                    if word_text in first_words_lower:
                        # Found potential match, verify next words
                        match = True
                        for j, fw in enumerate(first_words_lower[1:], 1):
                            if i + j < len(words):
                                next_word = (words[i+j].get('word') or words[i+j].get('text', '')).lower().strip('.,!?')
                                if next_word != fw:
                                    match = False
                                    break
                        
                        if match:
                            word_start = words[i].get('start')
                            if word_start is not None:
                                boundary_times.append(word_start)
                                current_word_idx = i
                            break

            logger.info(
                "Detected %d logical boundaries in chunk starting at %.1fs: %s",
                len(boundary_times),
                chunk_start_time,
                [f"{t:.1f}s" for t in boundary_times]
            )
            return boundary_times
            
        except Exception as e:
            logger.warning("Failed to detect logical boundaries: %s", e)
            return []

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
                    # text_en: English original (needed for Stage1 isochronic translation at render)
                    "text_en": first_seg.get("text_en", "").strip(),
                    "start": first_seg["start"],
                    "end": first_seg["end"],
                    "words": list(first_seg.get("words") or []),  # Word timestamps!
                }
                
                for seg in chunk_segments[1:]:
                    speaker = seg.get("speaker")
                    text_part = seg.get("text", "").strip()
                    text_en_part = seg.get("text_en", "").strip()
                    seg_words = seg.get("words") or []
                    
                    if speaker == current_turn["speaker"]:
                        # Same speaker, append text and words
                        current_turn["text"] += " " + text_part
                        if text_en_part:
                            current_turn["text_en"] = (current_turn.get("text_en", "") + " " + text_en_part).strip()
                        current_turn["end"] = seg["end"]
                        current_turn["words"].extend(seg_words)
                    else:
                        # New speaker, save current turn and start new
                        if current_turn["text"]:
                            dialogue.append(current_turn)
                        current_turn = {
                            "speaker": speaker,
                            "text": text_part,
                            "text_en": text_en_part,
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

        # Step 1: Build large chunks (~10 min) with 1 min overlap for boundary detection
        # Overlap ensures we don't cut mid-thought at chunk boundaries
        large_chunks = []
        current_idx = 0
        
        while current_idx < len(segments):
            # Find end of this chunk (chunk_size seconds from start)
            chunk_start_time = segments[current_idx]["start"]
            end_time = chunk_start_time + BOUNDARY_CHUNK_DURATION
            
            # Find the last segment that fits in this chunk
            end_idx = current_idx
            while end_idx < len(segments) and segments[end_idx]["end"] <= end_time:
                end_idx += 1
            
            # Make sure we include at least one segment
            if end_idx == current_idx:
                end_idx = current_idx + 1
            
            # Build chunk from current_idx to end_idx
            chunk_segments = segments[current_idx:end_idx]
            if chunk_segments:
                chunk = build_chunk(chunk_segments)
            if chunk["text"]:
                    large_chunks.append(chunk)
            
            # Move to next chunk: start at (chunk_size - overlap) from current start
            next_start_time = chunk_start_time + (BOUNDARY_CHUNK_DURATION - CHUNK_OVERLAP_DURATION)
            
            # Find first segment that starts at or after next_start_time
            while current_idx < len(segments) and segments[current_idx]["start"] < next_start_time:
                current_idx += 1
            
            # If we didn't move forward, force move to avoid infinite loop
            if current_idx < len(segments) and segments[current_idx]["start"] < next_start_time:
                current_idx = end_idx
        
        logger.info(
            "Built %d large chunks (~%d sec each, %d sec overlap) for boundary detection",
            len(large_chunks),
            BOUNDARY_CHUNK_DURATION,
            CHUNK_OVERLAP_DURATION
        )
        
        # Step 2: Detect logical boundaries in each large chunk and split
        for large_chunk in large_chunks:
            chunk_text = large_chunk["text"]
            chunk_words = large_chunk.get("words", [])
            chunk_start = large_chunk["start"]
            chunk_end = large_chunk["end"]
            
            # Get logical boundary timestamps from DeepSeek
            boundary_times = self._detect_logical_boundaries(
                chunk_text,
                chunk_start,
                chunk_words
            )
            
            if not boundary_times:
                # No boundaries found - use mechanical split as fallback
                logger.debug("No logical boundaries in chunk %.1f-%.1fs, using mechanical split", chunk_start, chunk_end)
                # Split by MAX_SEGMENT_DURATION (not max_duration which is 180s)
                sub_current = []
                for seg in segments:
                    if seg["start"] < chunk_start or seg["end"] > chunk_end:
                        continue
                    sub_current.append(seg)
                    sub_duration = sub_current[-1]["end"] - sub_current[0]["start"]
                    if sub_duration >= MAX_SEGMENT_DURATION:
                        sub_chunk = build_chunk(sub_current)
                        if sub_chunk["text"]:
                            merged.append(sub_chunk)
                        sub_current = []
                if sub_current:
                    sub_chunk = build_chunk(sub_current)
                    if sub_chunk["text"]:
                        merged.append(sub_chunk)
                continue
            
            # Add chunk boundaries
            all_boundaries = [chunk_start] + sorted(boundary_times) + [chunk_end]
            
            # Create segments between boundaries
            for i in range(len(all_boundaries) - 1):
                seg_start = all_boundaries[i]
                seg_end = all_boundaries[i + 1]
                seg_duration = seg_end - seg_start
                
                # Skip if too short
                if seg_duration < MIN_SEGMENT_DURATION:
                    continue
                
                # Find whisper segments in this range
                seg_whisper = [
                    s for s in segments
                    if s["start"] >= seg_start - 0.5 and s["end"] <= seg_end + 0.5
                ]
                
                if seg_whisper:
                    sub_chunk = build_chunk(seg_whisper)
                    if sub_chunk["text"]:
                        # If segment is too long, split mechanically by MAX_SEGMENT_DURATION
                        if sub_chunk["duration"] > MAX_SEGMENT_DURATION:
                            # Split into smaller pieces (~MAX_SEGMENT_DURATION each)
                            sub_current = []
                            for ws in seg_whisper:
                                sub_current.append(ws)
                                if sub_current[-1]["end"] - sub_current[0]["start"] >= MAX_SEGMENT_DURATION:
                                    sc = build_chunk(sub_current)
                                    if sc["text"]:
                                        merged.append(sc)
                                    sub_current = []
                            if sub_current:
                                sc = build_chunk(sub_current)
                                if sc["text"]:
                                    merged.append(sc)
                else:
                            merged.append(sub_chunk)

        # Fallback: if merging collapsed everything into a single tiny chunk, reuse original window logic
        if not merged and segments:
            return self._create_time_windows(segments, min_duration, max_duration)

        # Deduplicate segments from overlapping chunks
        # Sort by start time and remove segments that overlap significantly
        merged.sort(key=lambda x: x["start"])
        deduped = []
        for chunk in merged:
            if not deduped:
                deduped.append(chunk)
            else:
                last = deduped[-1]
                # If this segment starts before the last one ends (overlap), keep the better one
                if chunk["start"] < last["end"] - 1:  # 1 sec tolerance
                    # Keep the longer/more complete segment
                    if chunk["duration"] > last["duration"]:
                        deduped[-1] = chunk
                    # else keep the existing one
                else:
                    deduped.append(chunk)
        
        # Merge any segments that are too short with neighbors
        final_merged = []
        for chunk in deduped:
            if final_merged and chunk["duration"] < min_duration:
                final_merged[-1] = merge_with_previous(final_merged[-1], chunk)
            else:
                final_merged.append(chunk)

        logger.info(
            "Logical boundary split: %d segments from %d large chunks",
            len(final_merged),
            len(large_chunks)
        )
        
        return final_merged

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
    def _build_highlight_payload(
        idx: int,
        segment: dict,
        scores: dict,
        highlight_score: float,
        tier: str = "extended",
        is_fallback: bool = False
    ) -> dict:
        """
        Build highlight payload with tier classification.
        
        Tiers:
        - "strict": score >= STRICT_SCORE_THRESHOLD (0.35) - high quality
        - "extended": score >= HIGHLIGHT_SCORE_THRESHOLD (0.25) - good quality
        - "fallback": score >= FALLBACK_MIN_SCORE (0.15) - acceptable
        """
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
            'tier': tier,  # "strict", "extended", or "fallback"
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
        
        prompt = f"""Ты — редактор, который отбирает яркие клипы из длинных подкастов и интервью (вДудь, Diary of a CEO, Joe Rogan, Huberman Lab, Lex Fridman). Гости — предприниматели, учёные, психологи, врачи, спортсмены, авторы и лидеры мнений.

Оцени этот фрагмент как потенциальный YouTube Short / Instagram Reel.

ЦЕЛЕВАЯ ДЛИНА: 30-90 секунд (ИДЕАЛ: 30-60 секунд для лучшей виральности)
КОЛИЧЕСТВО СЛОВ: ~70-210 слов (140 слов ≈ 60 секунд)

Примечание: Сегменты длиннее 90 секунд плохо подходят для формата Shorts/Reels.

Оценивай для зрителей, которые ищут:

БИЗНЕС И ПРЕДПРИНИМАТЕЛЬСТВО:
- практичные бизнес-инсайты, тактики роста, метрики, фреймворки, контринтуитивные стратегии
- истории основателей с конкретными цифрами, пивотами, провалами, извлечёнными уроками

НАУКА И ИССЛЕДОВАНИЯ:
- увлекательные научные открытия, объяснённые просто
- контринтуитивные результаты исследований, бросающие вызов предположениям
- моменты «вот что на самом деле показывают данные»

ПСИХОЛОГИЯ И САМОРАЗВИТИЕ:
- инсайты о человеческом поведении, мотивации, отношениях
- ментальные модели, когнитивные искажения, фреймворки принятия решений
- прорывы в терапии/коучинге с практическим применением

ЗДОРОВЬЕ И ВЕЛНЕС:
- доказательные советы по здоровью (сон, питание, упражнения, долголетие)
- конкретные протоколы и рутины с научным обоснованием
- личные трансформации здоровья с измеримыми результатами

ПРОДУКТИВНОСТЬ И ОБУЧЕНИЕ:
- системы, инструменты и методы для достижения результатов
- техники обучения, методы изучения, освоение навыков
- управление временем и оптимизация энергии

ФИЛОСОФИЯ И ЖИЗНЬ:
- глубокие жизненные уроки из реального опыта
- мудрость об отношениях, цели, смысле
- сдвиги перспективы, меняющие взгляд на мир

УНИВЕРСАЛЬНЫЕ КРИТЕРИИ для высоких оценок:
- Сильная позиция: смелые мнения, подкреплённые опытом/доказательствами
- Конкретика: цифры, временные рамки, конкретные примеры
- Законченная арка: завязка → инсайт → вывод
- Эмоциональный резонанс: удивление, вдохновение, подтверждение, любопытство

Предполагай естественную речь (~140 слов в минуту). Фокусируйся на фрагментах примерно 70–210 слов (≈30–90 секунд, идеально 30-60с).

КОНТЕКСТ:
Предыдущая тема: {segment.get('prev_topic', 'Неизвестно')}
Следующая тема: {segment.get('next_topic', 'Неизвестно')}
Спикер: {segment.get('primary_speaker', 'Неизвестно')}
Начало видео: {segment.get('is_video_start', False)}
Конец видео: {segment.get('is_video_end', False)}

ВАЖНО:
- Используй ТОЛЬКО предоставленный текст. Не выдумывай контекст.
- Поощряй конкретику: цифры, чёткие выводы, пошаговые советы.
- Штрафуй воду, клише или отрывки, требующие слишком много окружающего контекста.
- Предпочитай самодостаточные арки (вопрос → инсайт/ответ).

ОБРАБОТКА «НЕИЗВЕСТНОГО» КОНТЕКСТА:
• Если «Предыдущая тема» = «Неизвестно», НО «Начало видео» = False:
  → Предыдущий контент ЕСТЬ, мы просто его не включили. Будь осторожен с needs_previous_context.
• Если «Следующая тема» = «Неизвестно», НО «Конец видео» = False:
  → После этого ЕСТЬ ещё контент! Если сегмент заканчивается на клиффхэнгере, продолжение скорее всего СУЩЕСТВУЕТ.
  → НЕ предполагай, что история не закончена — сначала проверь «Конец видео».
• Если «Конец видео» = True И сегмент заканчивается незавершённым:
  → Видео действительно заканчивается здесь. Итоговая оценка будет ограничена 0.20.

ОПРЕДЕЛЕНИЕ ГРАНИЦ СЕГМЕНТА (КРИТИЧНО):

Установи "needs_previous_context": true если текст:
• НАЧИНАЕТСЯ со связок: «И», «А», «Так что», «Поэтому», «Но», «Потому что», «Тогда», «Он сказал», «Она ответила»
• ИЛИ ссылается на что-то необъяснённое («он это сделал», «политик ушёл», «тогда-то»)
• ИЛИ ощущается как продолжение/развязка без завязки
• ИЛИ начинается с середины предложения или мысли
• ИЛИ (СЕМАНТИЧЕСКАЯ ПРОВЕРКА) «Предыдущая тема» описывает историю/событие, а ЭТОТ сегмент явно является выводом или уроком из неё. Пример: пред=«Как я провалил первый бизнес» + текст=«Теперь я всегда требую предоплату» → урок теряет смысл без истории провала.
• ИЛИ (ОСИРОТЕВШИЙ ОТВЕТ) сегмент НАЧИНАЕТСЯ с короткого ответа (1-5 слов), который отвечает на вопрос:
  - Короткие подтверждения: «Да», «Нет», «Точно», «Конечно», «Да, скорее всего»
  - Короткие ответы: «Довольно высокий», «Примерно 50%», «Около года», «Три раза»
  - Если «Предыдущая тема» вероятно содержит вопрос → needs_previous_context: true, trim_first_sentences: 1

Установи "needs_next_context": true если текст:
• ЗАКАНЧИВАЕТСЯ посреди истории (завязка без развязки, нарастание без кульминации)
• ИЛИ заканчивается клиффхэнгером: «и тогда—», «в тот момент—», «что произошло дальше...»
• ИЛИ задаёт вопрос, на который нет ответа в этом сегменте
• ИЛИ заканчивается тизером/обещанием без исполнения
• ИЛИ «Следующая тема» явно содержит недостающее разрешение
• ИЛИ (СЕМАНТИЧЕСКАЯ ПРОВЕРКА) сегмент явно является завязкой/постановкой проблемы, а «Следующая тема» содержит решение/ответ. Пример: текст=«Я терял $50К/месяц» + след=«Как я это исправил» → проблема без решения неполна.

ОПРЕДЕЛЕНИЕ ВИСЯЩЕЙ ЗАВЯЗКИ (для needs_next_context):
• Если сегмент ЗАКАНЧИВАЕТСЯ конкретным фактом/деталью, которая НЕ используется в этом сегменте:
  - «Его специализация была X» → но X больше не упоминается в ЭТОМ сегменте
  - «Компания называлась Y» → но история Y продолжается в следующем сегменте
  - «Это произошло в 1987» → но что произошло объясняется позже
• Эти «висящие завязки» указывают, что сегмент захватил одно предложение лишнее
• Установи needs_next_context: true если ПОСЛЕДНИЕ 1-2 предложения вводят новую информацию, которая не разрешается и не используется в том же сегменте

Пример ВИСЯЩЕЙ ЗАВЯЗКИ:
ПЛОХО: «...занимаются псевдонаукой или мелкими улучшениями без ценности. Его специализация — высокотемпературная сверхпроводимость.»
→ «сверхпроводимость» введена, но не используется в ЭТОМ сегменте → needs_next_context: true, trim_last_sentences: 1

ХОРОШО: «...занимаются псевдонаукой или мелкими улучшениями без ценности. Вот тогда всё изменилось.»
→ Законченная мысль, нет висящей завязки → needs_next_context: false

КОРРЕКТИРОВКА ГРАНИЦ (trim_first_sentences / trim_last_sentences):

Когда обнаруживаешь проблему с границей, можешь предложить переместить предложения между сегментами.

Установи "trim_last_sentences": N (целое 0-3) если:
• ПОСЛЕДНИЕ N предложений этого сегмента на самом деле являются ЗАВЯЗКОЙ для следующей темы
• Эти предложения вводят новый субъект/человека/концепцию, которые НЕ развиваются ЗДЕСЬ
• Пример: «...все они были мошенниками. Его специализация — сверхпроводимость.»
  → «Его специализация — сверхпроводимость» — завязка для СЛЕДУЮЩЕГО сегмента → trim_last_sentences: 1

Установи "trim_first_sentences": N (целое 0-3) если:
• ПЕРВЫЕ N предложений на самом деле являются ВЫВОДОМ из предыдущей темы
• Эти предложения ссылаются на что-то из предыдущего сегмента без объяснения
• Пример: «Поэтому я теперь всегда требую предоплату. Переходим к маркетингу...»
  → «Поэтому я теперь всегда требую предоплату» — вывод из ПРЕДЫДУЩЕГО → trim_first_sentences: 1

ВАЖНО для полей trim:
• По умолчанию 0 (корректировка не нужна)
• Предлагай только 1-3 предложения, не больше
• Если needs_next_context=true из-за висящей завязки, установи trim_last_sentences соответственно
• Если needs_previous_context=true из-за осиротевшего вывода, установи trim_first_sentences соответственно

Когда ЛЮБОЙ флаг true:
• Итоговая оценка будет ограничена 0.25 (неполные сегменты не являются самостоятельными клипами)
• Сегмент всё ещё может иметь высокие другие оценки, если сам контент хорош

ЛОГИКА ОБЪЕДИНЕНИЯ (для сегментов, которым нужен оба контекста):
Если сегменту нужен И предыдущий И следующий контекст (needs_previous_context И needs_next_context),
и вместе с соседями образует законченную мысль (тезис + доказательства + вывод),
оцени, создаст ли объединение лучший клип.

ОЦЕНКА ОБЪЕДИНЕНИЯ (заполни эти поля когда needs_previous_context ИЛИ needs_next_context = true):
- "merge_benefit": Насколько объединение улучшит клип?
  • "high" = Объединение НЕОБХОДИМО, сегмент неполон без соседей
  • "medium" = Объединение заметно улучшит качество
  • "low" = Объединение немного поможет, но сегмент смотрибелен сам по себе
  • "none" = Сегмент полон, объединение не нужно
- "estimated_merged_duration": Твоя оценка в секундах если объединить с соседями
  • Если needs_previous_context: добавь ~30-60 секунд на предыдущий контекст
  • Если needs_next_context: добавь ~30-60 секунд на следующий контекст
  • Если оба: оцени общую объединённую длительность
- "merged_completeness_score": Ожидаемая оценка completeness_arc (0.0-1.0) после объединения

ОГРАНИЧЕНИЯ ОБЪЕДИНЕНИЯ:
- ИДЕАЛЬНАЯ длительность объединённого: 60-90 секунд (лучше всего для Shorts/Reels)
- ДОПУСТИМО: 90-120 секунд (только для исключительного контента)
- МАКСИМУМ: 120 секунд (длиннее НЕ объединять, используй trim вместо этого)
- Если объединённый превысит 120 секунд, установи merge_benefit в "none"

ОПРЕДЕЛЕНИЕ СПИСКА vs АРГУМЕНТА:
- Если сегмент — просто СПИСОК идей без развития (типа «вот 10 проблем: А, Б, В, Г...»)
  → Значительно снизь ВСЕ оценки (это низкоценный контент для Shorts)
  → Добавь комментарий, что это нужно разделить или пропустить
- Если сегмент — полный АРГУМЕНТ (тезис → доказательства → вывод)
  → Сохрани высокие оценки даже если 90-120 секунд
  → Это ценный контент, стоящий своей длины

КАЛИБРОВОЧНЫЕ ПРИМЕРЫ (с новой системой оценки):

=== ПРИМЕР ВЫСОКОЙ ОЦЕНКИ (итог ~0.82) ===

«Мне было 28, и я был должен 30 миллионов рублей. Вот что меня спасло: я позвонил каждому кредитору
и договорился о скидке 60%. Потом построил бизнес на том, за что меня уволили — но уже для себя.
Через 18 месяцев я был свободен от долгов.»

Оценки:
- surprise_novelty: 0.7 (неожиданно: скидка 60% по долгам возможна)
- specificity_score: 0.9 (30 млн, 60%, 18 месяцев, возраст 28)
- personal_connection: 0.8 (личная история провала/искупления)
- actionability_score: 0.9 (чёткие шаги: звонить кредиторам, договариваться, строить бизнес)
- clarity_simplicity: 0.8 (простой язык, чёткая структура)
- completeness_arc: 0.9 (проблема → действия → результат)
- hook_quality: 0.85 («30 миллионов долга в 28» — мгновенно цепляет)
- needs_previous_context: false
- needs_next_context: false

=== ПРИМЕР СРЕДНЕЙ ОЦЕНКИ (итог ~0.45) ===

«ИИ изменит всё. Мы уже видим это в нашей индустрии.
Компании, которые не адаптируются, останутся позади. Вопрос не в том, если, а когда.»

Оценки:
- surprise_novelty: 0.2 (все это говорят)
- specificity_score: 0.1 (нет цифр, примеров, сроков)
- personal_connection: 0.1 (нет личной истории)
- actionability_score: 0.1 (нет шагов, просто «адаптируйтесь»)
- clarity_simplicity: 0.7 (легко понять)
- completeness_arc: 0.5 (есть мысль, но нет истории)
- hook_quality: 0.4 (общее начало)
- needs_previous_context: false
- needs_next_context: false

=== ПРИМЕР НИЗКОЙ ОЦЕНКИ — Неполный (итог ограничен 0.25) ===

«И сказал: "Готово." Политик ушёл довольный. Ты так же поступаешь
с политиками? Я заметил, когда я ввязываюсь в политику, это плохо заканчивается.»

Оценки:
- surprise_novelty: 0.3 (потенциально интересно)
- specificity_score: 0.1 (нет деталей)
- personal_connection: 0.2 (упоминает личный опыт размыто)
- actionability_score: 0.0 (нет советов)
- clarity_simplicity: 0.3 (непонятно без контекста)
- completeness_arc: 0.2 (нет начала)
- hook_quality: 0.1 («И сказал» — ужасный хук)
- needs_previous_context: TRUE (кто сказал? о чём?)
- needs_next_context: false
→ Итоговая оценка ОГРАНИЧЕНА 0.25 из-за неполного контекста

=== ПРИМЕР БАНАЛЬНОСТЕЙ (итог ~0.20) ===

«Ключ к успеху — настойчивость. Никогда не сдавайся.
Верь в себя. Ты можешь достичь всего, к чему стремишься.»

Оценки:
- surprise_novelty: 0.0 (противоположность удивительному)
- specificity_score: 0.0 (ноль конкретики)
- personal_connection: 0.0 (нет личной истории)
- actionability_score: 0.1 (просто «будь настойчив» и «верь»)
- clarity_simplicity: 0.8 (легко понять)
- completeness_arc: 0.3 (есть посыл, но нет истории)
- hook_quality: 0.2 (клишированное начало)
- needs_previous_context: false
- needs_next_context: false

=== ПРИМЕР ВИСЯЩЕЙ ЗАВЯЗКИ (trim_last_sentences) ===

Текст сегмента: «В науке много табу: ставить под сомнение дарвинизм, исследования 
стволовых клеток, изменение климата — это опасно. Но он выбрал тему ещё опаснее. 
Он считал, что большинство так называемых учёных попросту воруют государственные 
деньги, занимаются псевдонаукой или мелкими улучшениями без ценности. 
Его специализация — высокотемпературная сверхпроводимость.»

Следующий сегмент начинается: «Он сказал мне: в этой области вышло 50 тысяч статей...»

Анализ: Последнее предложение «Его специализация — высокотемпературная сверхпроводимость»
вводит конкретную область, которая НЕ обсуждается в ЭТОМ сегменте, но ЯВЛЯЕТСЯ главной
темой СЛЕДУЮЩЕГО сегмента («в этой области вышло 50 тысяч статей...»). Это ВИСЯЩАЯ 
ЗАВЯЗКА — сегмент заканчивается новой информацией, которая имеет смысл только со следующей частью.

Оценки:
- completeness_arc: 0.5 (хорошая арка испорчена висящим последним предложением)
- needs_next_context: TRUE (деталь о специализации требует следующий сегмент)
- trim_last_sentences: 1 (переместить «Его специализация...» в следующий сегмент)
→ Итоговая оценка ОГРАНИЧЕНА 0.25 из-за висящей завязки

ТЕКСТ:
"{segment['text']}"

КРИТЕРИИ ОЦЕНКИ (каждый измеряет ОДНУ конкретную вещь):

1. surprise_novelty (0.0–1.0)
   Что: Контринтуитивный инсайт, бросает вызов предположениям, раскрывает неожиданное
   НЕ: Важность или полезность информации
   Высоко: «Спать БОЛЬШЕ 9 часов так же вредно, как меньше 6»
   Низко: «Сон важен для здоровья»

2. specificity_score (0.0–1.0)
   Что: Конкретные цифры, даты, имена, измеримые результаты, конкретные примеры
   НЕ: Качество идеи или эмоциональная подача
   Высоко: «30 млн долга → скидка 60% → 18 месяцев → свободен от долгов»
   Низко: «У меня было много долгов и в итоге я их выплатил»

3. personal_connection (0.0–1.0)
   Что: Личная история, уязвимость, эмоциональный нарратив, близкий опыт
   НЕ: Практическая полезность или новизна
   Высоко: «На похоронах отца я понял, что никто не упомянул его деньги...»
   Низко: «Исследования показывают, что люди ценят отношения выше богатства»

4. actionability_score (0.0–1.0)
   Что: Чёткие инструкции, шаги, протоколы, «сделай X, потом Y, избегай Z»
   НЕ: Важность проблемы или эмоциональный вес
   Высоко: «Позвони каждому кредитору, попроси скидку 60%, документируй всё»
   Низко: «Тебе стоит попробовать договориться с кредиторами»

5. clarity_simplicity (0.0–1.0)
   Что: Доступный язык, без жаргона, хорошая структура, легко следить
   НЕ: Глубина или оригинальность мысли
   Высоко: Сложная тема объяснена на бытовых примерах
   Низко: Плотный академический язык, требует предварительных знаний

6. completeness_arc (0.0–1.0)
   Что: Есть начало, середина, конец. Проблема → действие → результат/урок
   НЕ: Качество каждого отдельного элемента
   Высоко: «Я был банкротом → сделал X → теперь успешен»
   Низко: «Вот что я сделал...» (нет завязки или результата)

7. hook_quality (0.0–1.0)
   Что: ТОЛЬКО первые 5-10 слов. Цепляет ли внимание сразу?
   НЕ: Качество остальной части клипа
   Высоко: «Всё, что ты знаешь о диетах — неправда»
   Низко: «Ну, да, я тут думал об этом...»

8. needs_previous_context (true/false)
   Требует ли этот сегмент предыдущий, чтобы иметь смысл?

9. needs_next_context (true/false)
   Требует ли этот сегмент следующий для законченной мысли?
   ВКЛЮЧАЕТ висящие завязки: последнее предложение вводит что-то, не использованное в ЭТОМ сегменте.

10. trim_first_sentences (0-3)
    Сколько предложений с НАЧАЛА должны перейти в ПРЕДЫДУЩИЙ сегмент?
    Используй когда первые предложения — выводы из предыдущей темы.

11. trim_last_sentences (0-3)
    Сколько предложений с КОНЦА должны перейти в СЛЕДУЮЩИЙ сегмент?
    Используй когда последние предложения — завязка для следующей темы (висящая завязка).

ПРОЦЕСС ОЦЕНКИ:
1. Сначала проверь needs_previous_context и needs_next_context (включая висящие завязки!)
2. Если needs_next_context из-за висящей завязки, установи trim_last_sentences соответственно
3. Если needs_previous_context из-за осиротевшего вывода, установи trim_first_sentences соответственно
4. Затем оцени completeness_arc (самое важное для самостоятельных клипов)
5. Оцени оставшиеся 6 критериев независимо
6. Будь строг: 0.8+ требует исключительного качества по этому измерению

ФОРМАТ ВЫВОДА:
Ответь ТОЛЬКО валидным JSON (без объяснений, без markdown):
{{
  "surprise_novelty": 0.0,
  "specificity_score": 0.0,
  "personal_connection": 0.0,
  "actionability_score": 0.0,
  "clarity_simplicity": 0.0,
  "completeness_arc": 0.0,
  "hook_quality": 0.0,
  "needs_previous_context": false,
  "needs_next_context": false,
  "trim_first_sentences": 0,
  "trim_last_sentences": 0,
  "merge_benefit": "none",
  "estimated_merged_duration": 0,
  "merged_completeness_score": 0.0
}}"""

        try:
            response_json = self.client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "Ты — редактор, который оценивает фрагменты транскрипции на потенциал вирусного короткого видео. "
                        "Выдавай лаконичные JSON-оценки от 0 до 1.",
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
                'actionability_score', 'clarity_simplicity', 'completeness_arc', 'hook_quality',
                'merged_completeness_score'
            ]
            boolean_fields = ['needs_previous_context', 'needs_next_context']
            trim_fields = ['trim_first_sentences', 'trim_last_sentences']
            integer_fields = ['estimated_merged_duration']
            merge_benefit_values = ['high', 'medium', 'low', 'none']
            
            # Validate scores
            for key in scores:
                if key in boolean_fields:
                    if not isinstance(scores[key], bool):
                        scores[key] = str(scores[key]).lower() in ('true', '1', 'yes')
                elif key in trim_fields:
                    # Integer 0-3
                    if not isinstance(scores[key], int):
                        try:
                            scores[key] = int(scores[key])
                        except (ValueError, TypeError):
                            scores[key] = 0
                    scores[key] = max(0, min(3, scores[key]))
                elif key in integer_fields:
                    # Integer (duration in seconds)
                    if not isinstance(scores[key], int):
                        try:
                            scores[key] = int(scores[key])
                        except (ValueError, TypeError):
                            scores[key] = 0
                    scores[key] = max(0, scores[key])
                elif key == 'merge_benefit':
                    # String: high/medium/low/none
                    if scores[key] not in merge_benefit_values:
                        scores[key] = 'none'
                elif key in numeric_fields:
                    if not isinstance(scores[key], (int, float)):
                        scores[key] = 0.0
                    scores[key] = max(0.0, min(1.0, float(scores[key])))
            
            # Ensure all required fields exist with defaults
            for field in numeric_fields:
                scores.setdefault(field, 0.5 if field != 'merged_completeness_score' else 0.0)
            for field in boolean_fields:
                scores.setdefault(field, False)
            for field in trim_fields:
                scores.setdefault(field, 0)
            for field in integer_fields:
                scores.setdefault(field, 0)
            scores.setdefault('merge_benefit', 'none')
            
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
                'trim_first_sentences': 0,
                'trim_last_sentences': 0,
                'merge_benefit': 'none',
                'estimated_merged_duration': 0,
                'merged_completeness_score': 0.0,
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

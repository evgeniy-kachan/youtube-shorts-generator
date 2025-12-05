"""Analyze transcription to find interesting segments using LLM."""
import logging
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

from backend.config import (
    DEEPSEEK_MODEL,
    HIGHLIGHT_CONCURRENT_REQUESTS,
    HIGHLIGHT_SEGMENTS_PER_CHUNK,
)
from backend.services.deepseek_client import DeepSeekClient

logger = logging.getLogger(__name__)


HIGHLIGHT_SCORE_THRESHOLD = 0.30
MIN_HIGHLIGHTS = 3
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
            # Merge consecutive Whisper segments into larger candidates to limit LLM calls
            potential_segments = self._merge_segments(
                segments,
                min_duration=min_duration,
                max_duration=max_duration,
                segments_per_chunk=HIGHLIGHT_SEGMENTS_PER_CHUNK,
            )
            self._attach_context_metadata(potential_segments)
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
            
            # Filter segments with good scores
            highlights = []
            scored_segments: list[tuple[int, dict, dict, float]] = []
            for i, (segment, scores) in enumerate(analyzed_segments):
                highlight_score = self._calculate_highlight_score(scores)
                scored_segments.append((i, segment, scores, highlight_score))

                if highlight_score >= HIGHLIGHT_SCORE_THRESHOLD:
                    highlights.append(self._build_highlight_payload(i, segment, scores, highlight_score))

            if len(highlights) < MIN_HIGHLIGHTS and scored_segments:
                existing_ids = {item['id'] for item in highlights}
                fallback_candidates = sorted(
                    scored_segments,
                    key=lambda tpl: tpl[3],
                    reverse=True,
                )
                for idx, segment, scores, highlight_score in fallback_candidates:
                    if len(highlights) >= MIN_HIGHLIGHTS:
                        break
                    if f"segment_{idx}" in existing_ids:
                        continue
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
            
            # Sort by highlight score
            highlights.sort(key=lambda x: x['highlight_score'], reverse=True)
            
            logger.info(f"Found {len(highlights)} interesting segments")
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
            dialogue = []
            if chunk_segments:
                current_turn = {
                    "speaker": chunk_segments[0].get("speaker"),
                    "text": chunk_segments[0].get("text", "").strip(),
                    "start": chunk_segments[0]["start"],
                    "end": chunk_segments[0]["end"]
                }
                
                for seg in chunk_segments[1:]:
                    speaker = seg.get("speaker")
                    text_part = seg.get("text", "").strip()
                    
                    if speaker == current_turn["speaker"]:
                        # Same speaker, append text
                        current_turn["text"] += " " + text_part
                        current_turn["end"] = seg["end"]
                    else:
                        # New speaker, save current turn and start new
                        if current_turn["text"]:
                            dialogue.append(current_turn)
                        current_turn = {
                            "speaker": speaker,
                            "text": text_part,
                            "start": seg["start"],
                            "end": seg["end"]
                        }
                
                if current_turn["text"]:
                    dialogue.append(current_turn)

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
                    "end": first["end"]
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

    def _attach_context_metadata(self, segments: list[dict]) -> None:
        for idx, segment in enumerate(segments):
            prev_text = segments[idx - 1]['text'] if idx > 0 else None
            next_text = segments[idx + 1]['text'] if idx + 1 < len(segments) else None
            segment['prev_topic'] = self._sanitize_topic_text(prev_text, PREV_TOPIC_MAX_WORDS)
            segment['next_topic'] = self._sanitize_topic_text(next_text, NEXT_TOPIC_MAX_WORDS)

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
        }
        if is_fallback:
            payload['is_fallback'] = True
        return payload

    def _analyze_segment_with_llm(self, segment: Dict) -> Dict[str, float]:
        """Analyze a single segment using LLM."""
        
        prompt = f"""You are an editor who curates punchy clips from long-form business podcasts and interviews (Joe Rogan, founders, investors, motivational speakers). Judge this fragment as a potential 20–90 second short for viewers who crave:
- actionable business insights, growth tactics, metrics, frameworks, contrarian strategies;
- strong POV from the guest: bold opinions, mindset, philosophy, worldview;
- honest failure/recovery stories with clear lessons;
- tips about lifestyle, discipline, routines, productivity systems;
- inspirational moments backed by concrete experience, not vague hype.

Assume natural speech (~140 words per minute). Focus on fragments roughly 45–210 words long (≈20–90 s).

CONTEXT:
Previous topic: {segment.get('prev_topic', 'Unknown')}
Next topic: {segment.get('next_topic', 'Unknown')}
Speaker: {segment.get('primary_speaker', 'Unknown')}

IMPORTANT:
- Use ONLY the provided text. Do not invent context.
- Reward specificity: numbers, clear takeaways, step-by-step advice.
- Penalize fluff, clichés, or passages that require too much surrounding context.
- Prefer self-contained arcs (question → insight/answer). If the fragment stops right after a question, teaser, or promise and the response is missing, treat it as incomplete: cap clip_worthiness at 0.20 and push other scores down.
- Use "Next topic" to decide whether the immediate continuation likely contains the missing answer. Until that continuation is included, assume the current fragment is incomplete and score it conservatively.

TEXT:
"{segment['text']}"

Score each dimension from 0.0–1.0 (clip is acceptable when highlight_score ≥ 0.30):

1. emotional_intensity — strength of emotions, surprise, tension, inspiration.
2. hook_potential — how strongly the beginning grabs attention.
3. business_value — density of practical business insight (tactics, metrics, lessons).
4. actionable_value — clarity/usefulness of advice, steps, or frameworks.
5. clip_worthiness — suitability as a standalone viral clip viewers will watch fully.

SCORING RULES:
- 0.0 = completely absent
- 1.0 = extremely strong
- Use decimal values (e.g., 0.35, 0.72).
- Be strict and realistic; high scores only when clearly deserved.
- If the story feels unfinished (question without answer, buildup without payoff), keep highlight_score < 0.30.

OUTPUT FORMAT:
Respond ONLY with valid JSON (no explanations, no markdown). Correct format example:
{{
  "emotional_intensity": 0.0,
  "hook_potential": 0.0,
  "business_value": 0.0,
  "actionable_value": 0.0,
  "clip_worthiness": 0.0
}}
If the format is violated, the response is invalid."""

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
            
            # Validate scores
            for key in scores:
                if not isinstance(scores[key], (int, float)):
                    scores[key] = 0.0
                scores[key] = max(0.0, min(1.0, float(scores[key])))
            
            return scores
            
        except Exception as e:
            logger.warning(f"Error parsing LLM response, using default scores: {e}")
            # Return neutral scores if parsing fails
            return {
                'emotional_intensity': 0.5,
                'hook_potential': 0.5,
                'business_value': 0.5,
                'actionable_value': 0.5,
                'clip_worthiness': 0.5,
            }
    
    def _calculate_highlight_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted highlight score emphasizing actionable and emotional impact.
        """
        highlight_score = (
            0.30 * scores.get('emotional_intensity', 0) +
            0.25 * scores.get('hook_potential', 0) +
            0.25 * scores.get('business_value', 0) +
            0.15 * scores.get('actionable_value', 0) +
            0.05 * scores.get('clip_worthiness', 0)
        )
        
        return min(1.0, highlight_score)

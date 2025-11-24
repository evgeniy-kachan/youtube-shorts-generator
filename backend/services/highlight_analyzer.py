"""Analyze transcription to find interesting segments using LLM."""
import logging
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

from backend.config import (
    DEEPSEEK_MODEL,
    HIGHLIGHT_CONCURRENT_REQUESTS,
)
from backend.services.deepseek_client import DeepSeekClient

logger = logging.getLogger(__name__)


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
            # Create overlapping windows of fixed sizes
            potential_segments = self._create_time_windows(segments, min_duration, max_duration)
            
            max_parallel = max_parallel or HIGHLIGHT_CONCURRENT_REQUESTS
            logger.info(f"Starting parallel LLM analysis with {max_parallel} concurrent requests...")
            
            # Analyze segments in parallel
            analyzed_segments = self._analyze_segments_parallel(potential_segments, max_parallel)
            
            # Filter segments with good scores
            highlights = []
            for i, (segment, scores) in enumerate(analyzed_segments):
                highlight_score = self._calculate_highlight_score(scores)
                
                if highlight_score > 0.4:  # Threshold for interesting content
                    highlights.append({
                        'id': f"segment_{i}",
                        'start_time': segment['start'],
                        'end_time': segment['end'],
                        'duration': segment['duration'],
                        'text': segment['text'],
                        'highlight_score': highlight_score,
                        'criteria_scores': scores
                    })
            
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
            step = size // 2  # 50% overlap
            
            for start_time_int in range(0, int(video_duration - size), step):
                start_time = float(start_time_int)
                end_time = start_time + size
                
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

        logger.info(f"Created {len(unique_windows)} unique time windows with sizes {window_sizes}")
        return unique_windows

    def _analyze_segment_with_llm(self, segment: Dict) -> Dict[str, float]:
        """Analyze a single segment using LLM."""
        
        prompt = f"""Analyze this video transcript segment for its potential as a viral short video (Reels/Shorts).
Rate each of the 5 key criteria from 0.0 to 1.0.

TEXT: "{segment['text']}"

Evaluate these 5 criteria:

1. emotional_intensity: Strength of emotion, surprise, or sentiment.
2. hook_potential: How well the start grabs attention.
3. key_value: Presence of actionable advice, insights, or takeaways.
4. story_moment: Does it tell a compelling story or personal example?
5. humor: Is there humor, jokes, or clear entertainment value?

Respond ONLY with valid JSON (no markdown or explanations):
{{
  "emotional_intensity": 0.0-1.0,
  "hook_potential": 0.0-1.0,
  "key_value": 0.0-1.0,
  "story_moment": 0.0-1.0,
  "humor": 0.0-1.0
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
                'key_value': 0.5,
                'story_moment': 0.5,
                'humor': 0.5,
            }
    
    def _calculate_highlight_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted highlight score based on 5 key criteria.
        """
        highlight_score = (
            0.30 * scores.get('emotional_intensity', 0) +
            0.30 * scores.get('hook_potential', 0) +
            0.20 * scores.get('key_value', 0) +
            0.15 * scores.get('story_moment', 0) +
            0.05 * scores.get('humor', 0)
        )
        
        return min(1.0, highlight_score)

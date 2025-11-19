"""Analyze transcription to find interesting segments using LLM."""
import json
import logging
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import ollama
from backend.config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_PORT

logger = logging.getLogger(__name__)


class HighlightAnalyzer:
    """Analyze video segments to find interesting moments."""
    
    def __init__(self, model_name: str = None, host: str = None, port: int = None):
        """
        Initialize analyzer with LLM.
        
        Args:
            model_name: Ollama model name (e.g., "llama3.1:8b", "qwen2.5:7b", "mistral:7b")
            host: Ollama host name
            port: Ollama port number
        """
        self.model_name = model_name or OLLAMA_MODEL
        host = host or OLLAMA_HOST
        port = port or OLLAMA_PORT
        
        # Configure ollama client
        self.client = ollama.Client(host=f"http://{host}:{port}")
            
        logger.info(f"Initialized HighlightAnalyzer with model: {self.model_name} on host http://{host}:{port}")
        
    def analyze_segments(self, segments: List[Dict], min_duration: int = 20, max_duration: int = 180, max_parallel: int = 5) -> List[Dict]:
        """
        Analyze transcription segments to find highlights (with parallel processing).
        
        Args:
            segments: List of transcription segments with timestamps
            min_duration: Minimum segment duration in seconds
            max_duration: Maximum segment duration in seconds
            max_parallel: Maximum number of parallel LLM requests (default: 5)
            
        Returns:
            List of highlighted segments with scores
        """
        try:
            # Group segments into potential highlights (20s - 3min)
            potential_segments = self._create_time_windows(segments, min_duration, max_duration)
            
            logger.info(f"Created {len(potential_segments)} potential segments")
            logger.info(f"Starting parallel LLM analysis with {max_parallel} concurrent requests...")
            
            # Analyze segments in parallel using ThreadPoolExecutor
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
            logger.error(f"Error analyzing segments: {e}")
            raise
    
    def _analyze_segments_parallel(self, segments: List[Dict], max_parallel: int) -> List[tuple]:
        """Analyze segments in parallel using ThreadPoolExecutor."""
        
        def analyze_single(i, segment):
            logger.info(f"Analyzing segment {i+1}/{len(segments)}")
            scores = self._analyze_segment_with_llm(segment)
            return (segment, scores)
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = [
                executor.submit(analyze_single, i, segment)
                for i, segment in enumerate(segments)
            ]
            
            # Collect results as they complete
            results = [future.result() for future in futures]
        
        return results
    
    def _create_time_windows(self, segments: List[Dict], min_duration: int, max_duration: int) -> List[Dict]:
        """Create time windows of appropriate duration, splitting long ones."""
        windows = []
        
        i = 0
        while i < len(segments):
            window_start = segments[i]['start']
            window_text = segments[i]['text']
            window_segments = [i]
            j = i + 1
            
            # Extend window until we reach max duration or run out of segments
            while j < len(segments):
                current_duration = segments[j]['end'] - window_start
                
                if current_duration > max_duration:
                    break
                    
                window_text += " " + segments[j]['text']
                window_segments.append(j)
                j += 1
                
                # Check if we have a good window
                if current_duration >= min_duration:
                    current_window = {
                        'start': window_start,
                        'end': segments[j-1]['end'],
                        'duration': current_duration,
                        'text': window_text.strip()
                    }
                    
                    # If window is too long, split it into smaller chunks
                    if current_duration > max_duration:
                        windows.extend(self._split_long_window(segments, window_segments, min_duration, max_duration))
                    else:
                        windows.append(current_window)
            
            # Move to next potential starting point (with 50% overlap)
            i += max(1, (j - i) // 2)
        
        logger.info(f"Created {len(windows)} time windows (min: {min_duration}s, max: {max_duration}s)")
        
        return windows
    
    def _split_long_window(self, segments: List[Dict], segment_indices: List[int], min_duration: int, max_duration: int) -> List[Dict]:
        """Split a long window into multiple smaller windows."""
        split_windows = []
        target_duration = (min_duration + max_duration) // 2  # Target ~100s for 20-180 range
        
        current_start_idx = segment_indices[0]
        current_text = ""
        current_start_time = segments[current_start_idx]['start']
        
        for idx in segment_indices:
            seg = segments[idx]
            current_text += " " + seg['text'] if current_text else seg['text']
            current_duration = seg['end'] - current_start_time
            
            # If we've reached target duration, create a window
            if current_duration >= target_duration:
                split_windows.append({
                    'start': current_start_time,
                    'end': seg['end'],
                    'duration': current_duration,
                    'text': current_text.strip()
                })
                
                # Start new window
                current_text = ""
                if idx + 1 < len(segments):
                    current_start_idx = idx + 1
                    current_start_time = segments[current_start_idx]['start']
        
        # Add remaining text as final window if it meets min_duration
        if current_text:
            final_duration = segments[segment_indices[-1]]['end'] - current_start_time
            if final_duration >= min_duration:
                split_windows.append({
                    'start': current_start_time,
                    'end': segments[segment_indices[-1]]['end'],
                    'duration': final_duration,
                    'text': current_text.strip()
                })
        
        logger.info(f"Split long window into {len(split_windows)} smaller windows")
        return split_windows
    
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
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'top_p': 0.9,
                }
            )
            
            # Parse JSON response
            response_text = response['response'].strip()
            
            # Try to extract JSON if wrapped in markdown
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            scores = json.loads(response_text)
            
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
        # New weighted formula for 5 criteria
        highlight_score = (
            0.30 * scores.get('emotional_intensity', 0) +
            0.30 * scores.get('hook_potential', 0) +
            0.20 * scores.get('key_value', 0) +
            0.15 * scores.get('story_moment', 0) +
            0.05 * scores.get('humor', 0)
        )
        
        return min(1.0, highlight_score)


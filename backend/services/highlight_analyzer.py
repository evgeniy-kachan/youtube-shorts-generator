"""Analyze transcription to find interesting segments using LLM."""
import json
import logging
from typing import List, Dict
import ollama
from backend.config import OLLAMA_HOST, OLLAMA_MODEL

logger = logging.getLogger(__name__)


class HighlightAnalyzer:
    """Analyze video segments to find interesting moments."""
    
    def __init__(self, model_name: str = None, host: str = None):
        """
        Initialize analyzer with LLM.
        
        Args:
            model_name: Ollama model name (e.g., "llama3.1:8b", "qwen2.5:7b", "mistral:7b")
            host: Ollama host URL
        """
        self.model_name = model_name or OLLAMA_MODEL
        self.host = host or OLLAMA_HOST
        
        # Configure ollama client
        if self.host != "http://localhost:11434":
            ollama.Client(host=self.host)
            
        logger.info(f"Initialized HighlightAnalyzer with model: {self.model_name}")
        
    def analyze_segments(self, segments: List[Dict], min_duration: int = 20, max_duration: int = 180) -> List[Dict]:
        """
        Analyze transcription segments to find highlights.
        
        Args:
            segments: List of transcription segments with timestamps
            min_duration: Minimum segment duration in seconds
            max_duration: Maximum segment duration in seconds
            
        Returns:
            List of highlighted segments with scores
        """
        try:
            # Group segments into potential highlights (20s - 3min)
            potential_segments = self._create_time_windows(segments, min_duration, max_duration)
            
            logger.info(f"Created {len(potential_segments)} potential segments")
            
            # Analyze each segment with LLM
            highlights = []
            for i, segment in enumerate(potential_segments):
                logger.info(f"Analyzing segment {i+1}/{len(potential_segments)}")
                
                scores = self._analyze_segment_with_llm(segment)
                
                # Calculate total highlight score
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
    
    def _create_time_windows(self, segments: List[Dict], min_duration: int, max_duration: int) -> List[Dict]:
        """Create time windows of appropriate duration."""
        windows = []
        
        i = 0
        while i < len(segments):
            window_start = segments[i]['start']
            window_text = segments[i]['text']
            j = i + 1
            
            # Extend window until we reach max duration or run out of segments
            while j < len(segments):
                current_duration = segments[j]['end'] - window_start
                
                if current_duration > max_duration:
                    break
                    
                window_text += " " + segments[j]['text']
                j += 1
                
                # Check if we have a good window
                if current_duration >= min_duration:
                    windows.append({
                        'start': window_start,
                        'end': segments[j-1]['end'],
                        'duration': current_duration,
                        'text': window_text.strip()
                    })
            
            # Move to next potential starting point (with 50% overlap)
            i += max(1, (j - i) // 2)
        
        return windows
    
    def _analyze_segment_with_llm(self, segment: Dict) -> Dict[str, float]:
        """Analyze a single segment using LLM."""
        
        prompt = f"""Analyze this video transcript segment for viral/engaging potential for Reels/Shorts.
Rate each criterion from 0.0 to 1.0:

TEXT: "{segment['text']}"

Evaluate these 12 criteria:

1. information_density: Does it contain new ideas, insights, conclusions, or important facts?
2. emotional_intensity: Is there emotional expression, surprise, humor, or sentiment shifts?
3. topic_transition: Does it introduce a new topic or perspective?
4. key_value: Does it provide actionable advice, tips, rules, or takeaways?
5. hook_potential: Does it start with an engaging hook phrase?
6. tension: Is there conflict, disagreement, or debate?
7. story_moment: Does it tell a story or personal example?
8. humor: Is there humor, jokes, or laughter?
9. cadence_shift: Does the speaking pace or intensity change noticeably?
10. keyword_density: Are there important domain-specific keywords?
11. multimodal_score: Combined text quality and structural completeness
12. audience_appeal: Would this interest a wide audience?

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "information_density": 0.0-1.0,
  "emotional_intensity": 0.0-1.0,
  "topic_transition": 0.0-1.0,
  "key_value": 0.0-1.0,
  "hook_potential": 0.0-1.0,
  "tension": 0.0-1.0,
  "story_moment": 0.0-1.0,
  "humor": 0.0-1.0,
  "cadence_shift": 0.0-1.0,
  "keyword_density": 0.0-1.0,
  "multimodal_score": 0.0-1.0,
  "audience_appeal": 0.0-1.0
}}"""

        try:
            response = ollama.generate(
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
                'information_density': 0.5,
                'emotional_intensity': 0.5,
                'topic_transition': 0.5,
                'key_value': 0.5,
                'hook_potential': 0.5,
                'tension': 0.5,
                'story_moment': 0.5,
                'humor': 0.5,
                'cadence_shift': 0.5,
                'keyword_density': 0.5,
                'multimodal_score': 0.5,
                'audience_appeal': 0.5,
            }
    
    def _calculate_highlight_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted highlight score.
        
        Based on your formula:
        highlight_score = 0.4 * semantic_value + 0.25 * emotional_intensity +
                         0.15 * hook_probability + 0.1 * keyword_density +
                         0.1 * story_probability
        """
        # Semantic value: average of information density, key value, and topic transition
        semantic_value = (
            scores.get('information_density', 0) +
            scores.get('key_value', 0) +
            scores.get('topic_transition', 0)
        ) / 3
        
        emotional_intensity = scores.get('emotional_intensity', 0)
        hook_probability = scores.get('hook_potential', 0)
        keyword_density = scores.get('keyword_density', 0)
        story_probability = scores.get('story_moment', 0)
        
        highlight_score = (
            0.4 * semantic_value +
            0.25 * emotional_intensity +
            0.15 * hook_probability +
            0.1 * keyword_density +
            0.1 * story_probability
        )
        
        # Bonus for high audience appeal
        if scores.get('audience_appeal', 0) > 0.7:
            highlight_score *= 1.1
        
        # Bonus for humor
        if scores.get('humor', 0) > 0.7:
            highlight_score *= 1.05
        
        # Bonus for tension/conflict
        if scores.get('tension', 0) > 0.7:
            highlight_score *= 1.05
        
        return min(1.0, highlight_score)


import axios from 'axios';

export const API_BASE_URL = process.env.REACT_APP_API_URL || '';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const analyzeLocalVideo = async (filename, analysisMode = 'fast', diarizer = 'pyannote') => {
  const response = await api.post(
    `/api/video/analyze-local?filename=${filename}&analysis_mode=${analysisMode}&diarizer=${diarizer}`
  );
  return response.data;
};

export const getTaskStatus = async (taskId) => {
  const response = await api.get(`/api/video/task/${taskId}`);
  return response.data;
};

export const processSegments = async (
  videoId,
  segmentIds,
  ttsProvider = 'local',
  voiceMix = 'male_duo',
  verticalMethod = 'letterbox',
  subtitleAnimation = 'fade',
  subtitlePosition = 'mid_low',
  subtitleFont = 'Montserrat Light',
  subtitleFontSize = 86,
  subtitleBackground = false,
  subtitleGlow = true,
  subtitleGradient = false,
  preserveBackgroundAudio = true,
  cropFocus = 'center',
  speakerColorMode = 'colored',
  numSpeakers = 0,
  speakerChangeTime = '',
  speakerChangePhrase = '',
  rediarizeSegments = false
) => {
  const response = await api.post('/api/video/process', {
    video_id: videoId,
    segment_ids: segmentIds,
    tts_provider: ttsProvider,
    voice_mix: voiceMix,
    vertical_method: verticalMethod,
    subtitle_animation: subtitleAnimation,
    subtitle_position: subtitlePosition,
    subtitle_font: subtitleFont,
    subtitle_font_size: subtitleFontSize,
    subtitle_background: subtitleBackground,
    subtitle_glow: subtitleGlow,
    subtitle_gradient: subtitleGradient,
    preserve_background_audio: preserveBackgroundAudio,
    crop_focus: cropFocus,
    speaker_color_mode: speakerColorMode,
    num_speakers: numSpeakers,
    speaker_change_times: speakerChangeTime,
    speaker_change_phrases: speakerChangePhrase,
    rediarize_segments: rediarizeSegments,
  });
  return response.data;
};

export const getDownloadUrl = (videoId, segmentId) => {
  return `${API_BASE_URL}/api/video/download/${videoId}/${segmentId}`;
};

export const getTranscriptionDownloadUrl = (videoId) => {
  return `${API_BASE_URL}/api/video/download-transcription/${videoId}`;
};

export const dubSegment = async (
  videoId,
  segmentId,
  sourceLang = 'en',
  targetLang = 'ru',
  verticalMethod = 'center_crop',
  cropFocus = 'face_auto',
  subtitleAnimation = 'fade',
  subtitlePosition = 'mid_low',
  subtitleFont = 'Montserrat Light',
  subtitleFontSize = 86,
  subtitleBackground = false,
  subtitleGlow = true,
  subtitleGradient = false
) => {
  const response = await api.post('/api/video/dubbing', {
    video_id: videoId,
    segment_id: segmentId,
    source_lang: sourceLang,
    target_lang: targetLang,
    vertical_method: verticalMethod,
    crop_focus: cropFocus,
    subtitle_animation: subtitleAnimation,
    subtitle_position: subtitlePosition,
    subtitle_font: subtitleFont,
    subtitle_font_size: subtitleFontSize,
    subtitle_background: subtitleBackground,
    subtitle_glow: subtitleGlow,
    subtitle_gradient: subtitleGradient,
  });
  return response.data;
};

export const uploadVideoFile = async (file, onUploadProgress) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/api/video/upload-video', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (onUploadProgress && progressEvent.total) {
        const percentCompleted = progressEvent.loaded / progressEvent.total;
        const loadedMB = (progressEvent.loaded / (1024 * 1024)).toFixed(2);
        const totalMB = (progressEvent.total / (1024 * 1024)).toFixed(2);
        onUploadProgress({
          percent: percentCompleted,
          loadedMB: parseFloat(loadedMB),
          totalMB: parseFloat(totalMB),
        });
      }
    },
  });

  return response.data;
};

export const cleanupVideo = async (videoId) => {
  const response = await api.delete(`/api/video/cleanup/${videoId}`);
  return response.data;
};

export const restoreSession = async () => {
  const response = await api.get('/api/video/session/restore');
  return response.data;
};

// Development helper: Get list of cached uploaded videos
export const getUploadedVideos = async () => {
  const response = await api.get('/api/video/uploaded-videos');
  return response.data;
};

// Development helper: Use cached video instead of uploading
export const getCachedVideo = async (videoId) => {
  const response = await api.post(`/api/video/use-cached-video/${videoId}`);
  return response.data;
};

export const generateDescription = async (textEn, textRu, duration, highlightScore = 0) => {
  const response = await api.post('/api/video/generate-description', {
    text_en: textEn,
    text_ru: textRu,
    duration: duration,
    highlight_score: highlightScore,
  });
  return response.data;
};

// NeMo MSDD Diarization
export const getNemoStatus = async () => {
  const response = await api.get('/api/video/nemo/status');
  return response.data;
};

export const runNemoDiarization = async (videoId, options = {}) => {
  const {
    numSpeakers = 0,
    maxSpeakers = 8,
    autoRender = false,
    segmentIds = [],
    ttsProvider = 'elevenlabs',
    voiceMix = 'male_duo',
    verticalMethod = 'center_crop',
    subtitleAnimation = 'highlight',
    subtitlePosition = 'mid_low',
    subtitleFont = 'Montserrat Light',
    subtitleFontSize = 86,
    subtitleBackground = false,
    subtitleGlow = true,
    subtitleGradient = false,
    speakerColorMode = 'colored',
    preserveBackgroundAudio = true,
    cropFocus = 'center',
  } = options;
  
  const response = await api.post('/api/video/nemo/diarize', {
    video_id: videoId,
    num_speakers: numSpeakers,
    max_speakers: maxSpeakers,
    auto_render: autoRender,
    segment_ids: segmentIds,
    tts_provider: ttsProvider,
    voice_mix: voiceMix,
    vertical_method: verticalMethod,
    subtitle_animation: subtitleAnimation,
    subtitle_position: subtitlePosition,
    subtitle_font: subtitleFont,
    subtitle_font_size: subtitleFontSize,
    subtitle_background: subtitleBackground,
    subtitle_glow: subtitleGlow,
    subtitle_gradient: subtitleGradient,
    speaker_color_mode: speakerColorMode,
    preserve_background_audio: preserveBackgroundAudio,
    crop_focus: cropFocus,
  });
  return response.data;
};

export default api;

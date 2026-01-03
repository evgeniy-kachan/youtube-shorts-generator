import axios from 'axios';

export const API_BASE_URL = process.env.REACT_APP_API_URL || '';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const analyzeLocalVideo = async (filename, analysisMode = 'fast') => {
  const response = await api.post(
    `/api/video/analyze-local?filename=${filename}&analysis_mode=${analysisMode}`
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
  subtitleAnimation = 'bounce',
  subtitlePosition = 'mid_low',
  subtitleFont = 'Montserrat Light',
  subtitleFontSize = 86,
  subtitleBackground = false,
  preserveBackgroundAudio = true,
  cropFocus = 'center'
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
    preserve_background_audio: preserveBackgroundAudio,
    crop_focus: cropFocus,
  });
  return response.data;
};

export const getDownloadUrl = (videoId, segmentId) => {
  return `${API_BASE_URL}/api/video/download/${videoId}/${segmentId}`;
};

export const uploadVideoFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/api/video/upload-video', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

export const cleanupVideo = async (videoId) => {
  const response = await api.delete(`/api/video/cleanup/${videoId}`);
  return response.data;
};

export default api;

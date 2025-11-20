import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || '';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const analyzeLocalVideo = async (filename) => {
  const response = await api.post(
    `/api/video/analyze-local?filename=${filename}`
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
  verticalMethod = 'blur_background'
) => {
  const response = await api.post('/api/video/process', {
    video_id: videoId,
    segment_ids: segmentIds,
    vertical_method: verticalMethod,
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

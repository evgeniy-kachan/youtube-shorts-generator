import React, { useState, useEffect, useMemo, useCallback } from 'react';
import Header from './components/Header';
import VideoInput from './components/VideoInput';
import ProgressBar from './components/ProgressBar';
import SegmentsList from './components/SegmentsList';
import DownloadList from './components/DownloadList';
import {
  analyzeLocalVideo,
  getTaskStatus,
  processSegments,
  dubSegment,
  uploadVideoFile,
  restoreSession,
  API_BASE_URL,
} from './services/api';

function App() {
  const [stage, setStage] = useState('input'); // input, analyzing, segments, processing, download
  const [analysisTask, setAnalysisTask] = useState(null);
  const [processingTask, setProcessingTask] = useState(null);
  const [videoData, setVideoData] = useState(null);
  const [segments, setSegments] = useState([]);
  const [processedSegments, setProcessedSegments] = useState([]);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [taskStatus, setTaskStatus] = useState('pending');
  const [isUploading, setIsUploading] = useState(false);
  // eslint-disable-next-line no-unused-vars
  const [uploadProgress, setUploadProgress] = useState({ loadedMB: 0, totalMB: 0, percent: 0 });
  const [sessionRestored, setSessionRestored] = useState(false);

  // Build processed segments from result
  const buildProcessedSegments = useMemo(
    () => (result) => {
      if (Array.isArray(result?.processed_segments)) {
        return result.processed_segments;
      }

      if (Array.isArray(result?.output_videos)) {
        return result.output_videos.map((item) => {
          // New format: { path, segment_id, description }
          // Old format: string (relativePath)
          const isNewFormat = typeof item === 'object' && item.path;
          const relativePath = isNewFormat ? item.path : item;
          
          const parts = relativePath.split('/');
          const filename = parts[parts.length - 1] || relativePath;
          const segmentId = isNewFormat ? item.segment_id : filename.replace('.mp4', '');
          const originalSegment =
            segments.find((segment) => segment.id === segmentId) || {};

          const duration =
            typeof originalSegment.start_time === 'number' &&
            typeof originalSegment.end_time === 'number'
              ? originalSegment.end_time - originalSegment.start_time
              : 0;

          return {
            segment_id: segmentId,
            filename,
            duration,
            start_time: originalSegment.start_time ?? null,
            end_time: originalSegment.end_time ?? null,
            download_path: relativePath,
            // New: include description data
            description: isNewFormat ? item.description : null,
          };
        });
      }

      return [];
    },
    [segments]
  );

  // Session recovery: save task IDs to localStorage
  useEffect(() => {
    if (analysisTask) {
      localStorage.setItem('currentAnalysisTask', analysisTask);
      localStorage.setItem('currentTaskType', 'analysis');
    }
  }, [analysisTask]);

  useEffect(() => {
    if (processingTask) {
      localStorage.setItem('currentProcessingTask', processingTask);
      localStorage.setItem('currentTaskType', 'processing');
    }
  }, [processingTask]);

  // Clear localStorage when task completes or resets
  useEffect(() => {
    if (stage === 'segments' || stage === 'download' || stage === 'input') {
      if (stage === 'input') {
        localStorage.removeItem('currentAnalysisTask');
        localStorage.removeItem('currentProcessingTask');
        localStorage.removeItem('currentTaskType');
      }
    }
  }, [stage]);

  // Restore session on mount and when browser wakes from sleep
  useEffect(() => {
    const tryRestoreSession = async () => {
      if (sessionRestored) return;
      
      try {
        // First check localStorage for saved task IDs
        const savedAnalysisTask = localStorage.getItem('currentAnalysisTask');
        const savedProcessingTask = localStorage.getItem('currentProcessingTask');
        const savedTaskType = localStorage.getItem('currentTaskType');
        
        if (savedAnalysisTask || savedProcessingTask) {
          console.log('[Session Recovery] Found saved tasks:', { savedAnalysisTask, savedProcessingTask, savedTaskType });
          
          // Try to get task status from server
          const sessionData = await restoreSession();
          
          if (sessionData.has_session && sessionData.task) {
            const task = sessionData.task;
            console.log('[Session Recovery] Server returned task:', task);
            
            if (task.status === 'completed' && task.result) {
              // Task completed while we were away - restore results
              if (savedTaskType === 'analysis') {
                const normalizedResult = {
                  ...task.result,
                  thumbnail_url: task.result?.thumbnail_url
                    ? `${API_BASE_URL}${task.result.thumbnail_url}`
                    : null,
                };
                setVideoData(normalizedResult);
                setSegments(normalizedResult.segments || []);
                setStage('segments');
                setStatusMessage('Сессия восстановлена! Анализ был завершён.');
              } else if (savedTaskType === 'processing') {
                // Need segments to build processed segments
                // Try to restore from analysis task first
                if (savedAnalysisTask) {
                  try {
                    const analysisStatus = await getTaskStatus(savedAnalysisTask);
                    if (analysisStatus.status === 'completed' && analysisStatus.result) {
                      const normalizedResult = {
                        ...analysisStatus.result,
                        thumbnail_url: analysisStatus.result?.thumbnail_url
                          ? `${API_BASE_URL}${analysisStatus.result.thumbnail_url}`
                          : null,
                      };
                      setVideoData(normalizedResult);
                      setSegments(normalizedResult.segments || []);
                    }
                  } catch (e) {
                    console.warn('[Session Recovery] Could not restore analysis data:', e);
                  }
                }
                
                // Now restore processing results
                if (task.result?.output_videos || task.result?.processed_segments) {
                  // Need segments to build processed segments properly
                  // If segments not restored yet, try one more time
                  if (!segments.length && savedAnalysisTask) {
                    try {
                      const analysisStatus = await getTaskStatus(savedAnalysisTask);
                      if (analysisStatus.status === 'completed' && analysisStatus.result) {
                        const normalizedResult = {
                          ...analysisStatus.result,
                          thumbnail_url: analysisStatus.result?.thumbnail_url
                            ? `${API_BASE_URL}${analysisStatus.result.thumbnail_url}`
                            : null,
                        };
                        setVideoData(normalizedResult);
                        setSegments(normalizedResult.segments || []);
                      }
                    } catch (e) {
                      console.warn('[Session Recovery] Could not restore analysis data (retry):', e);
                    }
                  }
                  
                  const processed = buildProcessedSegments(task.result);
                  setProcessedSegments(processed);
                  setStage('download');
                  setStatusMessage('Сессия восстановлена! Обработка была завершена.');
                }
              }
            } else if (task.status === 'processing' || task.status === 'pending') {
              // Task still running - resume polling
              if (savedTaskType === 'analysis' && savedAnalysisTask) {
                setAnalysisTask(savedAnalysisTask);
                setStage('analyzing');
                setProgress(task.progress || 0);
                setStatusMessage(task.message || 'Восстановление сессии...');
              } else if (savedTaskType === 'processing' && savedProcessingTask) {
                // Need to restore videoData first
                if (savedAnalysisTask) {
                  try {
                    const analysisStatus = await getTaskStatus(savedAnalysisTask);
                    if (analysisStatus.status === 'completed' && analysisStatus.result) {
                      const normalizedResult = {
                        ...analysisStatus.result,
                        thumbnail_url: analysisStatus.result?.thumbnail_url
                          ? `${API_BASE_URL}${analysisStatus.result.thumbnail_url}`
                          : null,
                      };
                      setVideoData(normalizedResult);
                      setSegments(normalizedResult.segments || []);
                    }
                  } catch (e) {
                    console.warn('[Session Recovery] Could not restore analysis data:', e);
                  }
                }
                setProcessingTask(savedProcessingTask);
                setStage('processing');
                setProgress(task.progress || 0);
                setStatusMessage(task.message || 'Восстановление сессии...');
              }
            }
          }
        }
      } catch (error) {
        console.warn('[Session Recovery] Failed to restore session:', error);
      } finally {
        setSessionRestored(true);
      }
    };

    tryRestoreSession();
  }, [sessionRestored, stage]);

  // Handle browser wake from sleep (visibility change) with debounce
  const lastVisibilityCheck = React.useRef(0);
  const handleVisibilityChange = useCallback(async () => {
    if (document.visibilityState !== 'visible') return;
    
    // Debounce: ignore if last check was less than 2 seconds ago
    const now = Date.now();
    if (now - lastVisibilityCheck.current < 2000) {
      return;
    }
    lastVisibilityCheck.current = now;
    
    // Skip if already in a completed stage (segments, download, upload)
    if (['segments', 'download', 'upload'].includes(stage)) {
      return;
    }
    
    console.log('[Session Recovery] Browser became visible, checking session...');
    
    // If we're in a polling stage, immediately check task status
    if (stage === 'analyzing' && analysisTask) {
      try {
        const status = await getTaskStatus(analysisTask);
        if (status.status === 'completed' && status.result) {
          const normalizedResult = {
            ...status.result,
            thumbnail_url: status.result?.thumbnail_url
              ? `${API_BASE_URL}${status.result.thumbnail_url}`
              : null,
          };
          setVideoData(normalizedResult);
          setSegments(normalizedResult.segments || []);
          setStage('segments');
          setStatusMessage('Сессия восстановлена! Анализ был завершён.');
        } else if (status.status === 'processing' || status.status === 'pending') {
          setProgress(status.progress || 0);
          setStatusMessage(status.message || 'Анализ продолжается...');
        }
      } catch (error) {
        console.warn('[Session Recovery] Could not check analysis task:', error);
      }
    } else if (stage === 'processing' && processingTask) {
      try {
        const status = await getTaskStatus(processingTask);
        if (status.status === 'completed' && status.result) {
          const processed = buildProcessedSegments(status.result);
          setProcessedSegments(processed);
          setStage('download');
          setStatusMessage('Сессия восстановлена! Обработка была завершена.');
        } else if (status.status === 'processing' || status.status === 'pending') {
          setProgress(status.progress || 0);
          setStatusMessage(status.message || 'Обработка продолжается...');
        }
      } catch (error) {
        console.warn('[Session Recovery] Could not check processing task:', error);
      }
    } else if (stage === 'upload') {
      // On upload stage, check if there are saved tasks to restore
      const savedAnalysisTask = localStorage.getItem('currentAnalysisTask');
      const savedProcessingTask = localStorage.getItem('currentProcessingTask');
      
      if ((savedAnalysisTask || savedProcessingTask) && !sessionRestored) {
        console.log('[Session Recovery] Found saved tasks on wake, attempting restore...');
        setSessionRestored(false);
      }
    }
  }, [stage, analysisTask, processingTask, buildProcessedSegments, sessionRestored]);

  // Set up visibility change listener
  useEffect(() => {
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [handleVisibilityChange]);

  useEffect(() => {
    let interval;

    if (analysisTask && stage === 'analyzing') {
      interval = setInterval(async () => {
        try {
          const status = await getTaskStatus(analysisTask);
          setProgress(status.progress);
          setStatusMessage(status.message);
          setTaskStatus(status.status);

          if (status.status === 'completed') {
            const normalizedResult = {
              ...status.result,
              thumbnail_url: status.result?.thumbnail_url
                ? `${API_BASE_URL}${status.result.thumbnail_url}`
                : null,
            };
            setVideoData(normalizedResult);
            setSegments(normalizedResult.segments);
            setStage('segments');
            // Keep analysisTask in localStorage for processing recovery
            localStorage.removeItem('currentTaskType');
            clearInterval(interval);
          } else if (status.status === 'failed') {
            alert('Ошибка при анализе видео: ' + status.message);
            setStage('input');
            localStorage.removeItem('currentAnalysisTask');
            localStorage.removeItem('currentTaskType');
            clearInterval(interval);
          }
        } catch (error) {
          console.error('Error polling status:', error);
          // Don't clear interval on network errors - browser might be waking from sleep
          // Instead, show a recoverable message and keep trying
          if (error?.code === 'ERR_NETWORK' || error?.message?.includes('Network Error')) {
            setStatusMessage('Соединение потеряно. Переподключение...');
            // Don't clear interval - keep trying
            // When connection is restored, the next poll will succeed
          } else {
            clearInterval(interval);
          }
        }
      }, 2000);
    }

    if (processingTask && stage === 'processing') {
      interval = setInterval(async () => {
        try {
          const status = await getTaskStatus(processingTask);
          setProgress(status.progress);
          setStatusMessage(status.message);
          setTaskStatus(status.status);

          if (status.status === 'completed') {
            const processed = buildProcessedSegments(status.result);
            setProcessedSegments(processed);
            setStage('download');
            localStorage.removeItem('currentProcessingTask');
            localStorage.removeItem('currentTaskType');
            clearInterval(interval);
          } else if (status.status === 'failed') {
            alert('Ошибка при обработке видео: ' + status.message);
            setStage('segments');
            localStorage.removeItem('currentProcessingTask');
            localStorage.removeItem('currentTaskType');
            clearInterval(interval);
          }
        } catch (error) {
          console.error('Error polling status:', error);
          // Don't clear interval on network errors - browser might be waking from sleep
          if (error?.code === 'ERR_NETWORK' || error?.message?.includes('Network Error')) {
            setStatusMessage('Соединение потеряно. Переподключение...');
            // Don't clear interval - keep trying
          } else {
            clearInterval(interval);
          }
        }
      }, 2000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [analysisTask, processingTask, stage, segments, buildProcessedSegments]);

  const handleAnalyze = async (file, analysisMode = 'fast') => {
    if (!file) {
      alert('Пожалуйста, выберите видеофайл формата MP4, MOV или WEBM');
      return;
    }

    try {
      setIsUploading(true);
      setStage('analyzing');
      setProgress(0.02);
      setStatusMessage('Загружаем файл...');
      setTaskStatus('pending');
      setUploadProgress({ loadedMB: 0, totalMB: 0, percent: 0 });

      let uploadedFilename;

      // Development: Skip upload if using cached video
      if (file.cached) {
        uploadedFilename = file.name;
        setProgress(0.05);
        setStatusMessage('Используем кэшированный файл. Начинаем анализ...');
      } else {
        const uploadResponse = await uploadVideoFile(file, (progressData) => {
          setUploadProgress(progressData);
          const percent = progressData.percent * 0.95; // Upload takes up to 95% of initial progress
          setProgress(0.02 + percent);
          setStatusMessage(
            `Загружаем файл: ${progressData.loadedMB.toFixed(1)} / ${progressData.totalMB.toFixed(1)} МБ (${(progressData.percent * 100).toFixed(1)}%)`
          );
        });
        uploadedFilename = uploadResponse?.result?.filename;

        if (!uploadedFilename) {
          throw new Error('Сервер не вернул имя загруженного файла');
        }

        setProgress(0.05);
        setStatusMessage(analysisMode === 'deep' 
          ? 'Файл загружен. Запускаем глубокий анализ (3-4 мин)...' 
          : 'Файл загружен. Начинаем анализ...');
      }

      const response = await analyzeLocalVideo(uploadedFilename, analysisMode);
      setAnalysisTask(response.task_id);
    } catch (error) {
      console.error('Error starting analysis:', error);
      alert('Ошибка при запуске анализа: ' + (error?.message || 'Неизвестная ошибка'));
      setStage('input');
    } finally {
      setIsUploading(false);
    }
  };

  const handleProcess = async (
    segmentIds,
    verticalMethod,
    subtitleAnimation,
    subtitlePosition,
    subtitleFont,
    subtitleFontSize,
    subtitleBackground,
    subtitleGlow = true,
    subtitleGradient = false,
    ttsProvider = 'local',
    voiceMix = 'male_duo',
    preserveBackgroundAudio = true,
    cropFocus = 'center',
    speakerColorMode = 'colored',
    numSpeakers = 0,
    speakerChangeTime = ''
  ) => {
    try {
      console.log('[handleProcess] subtitleAnimation:', subtitleAnimation, 'numSpeakers:', numSpeakers, 'speakerChangeTime:', speakerChangeTime);
      setStage('processing');
      setProgress(0);
      setStatusMessage('Начинаем обработку...');
      setTaskStatus('pending');

      const response = await processSegments(
        videoData.video_id,
        segmentIds,
        ttsProvider,
        voiceMix,
        verticalMethod,
        subtitleAnimation,
        subtitlePosition,
        subtitleFont,
        subtitleFontSize,
        subtitleBackground,
        subtitleGlow,
        subtitleGradient,
        preserveBackgroundAudio,
        cropFocus,
        speakerColorMode,
        numSpeakers,
        speakerChangeTime
      );
      setProcessingTask(response.task_id);
    } catch (error) {
      console.error('Error starting processing:', error);
      alert('Ошибка при запуске обработки: ' + error.message);
      setStage('segments');
    }
  };

  const handleDubbing = async (
    segmentId,
    verticalMethod,
    subtitleAnimation,
    subtitlePosition,
    subtitleFont,
    subtitleFontSize,
    subtitleBackground,
    subtitleGlow = true,
    subtitleGradient = false,
    cropFocus = 'face_auto'
  ) => {
    try {
      setStage('processing');
      setProgress(0);
      setStatusMessage('Запускаем AI Дубляж (ElevenLabs)...');
      setTaskStatus('pending');

      const response = await dubSegment(
        videoData.video_id,
        segmentId,
        'en',  // source_lang
        'ru',  // target_lang
        verticalMethod,
        cropFocus,
        subtitleAnimation,
        subtitlePosition,
        subtitleFont,
        subtitleFontSize,
        subtitleBackground,
        subtitleGlow,
        subtitleGradient
      );
      setProcessingTask(response.task_id);
    } catch (error) {
      console.error('Error starting AI dubbing:', error);
      alert('Ошибка AI Дубляжа: ' + error.message);
      setStage('segments');
    }
  };

  const isBusyStage = stage === 'analyzing' || stage === 'processing';
  const hasSessionData =
    !!videoData || segments.length > 0 || processedSegments.length > 0;

  const handleReset = () => {
    setStage('input');
    setAnalysisTask(null);
    setProcessingTask(null);
    setVideoData(null);
    setSegments([]);
    setProcessedSegments([]);
    setProgress(0);
    setStatusMessage('');
    setTaskStatus('pending');
    setIsUploading(false);
  };

  const handleNewVideoRequest = () => {
    if (isBusyStage) {
      alert('Дождитесь завершения текущей операции, затем начните заново.');
      return;
    }

    if (hasSessionData) {
      const confirmed = window.confirm(
        'Текущие результаты будут сброшены. Начать новое видео?'
      );
      if (!confirmed) {
        return;
      }
    }

    handleReset();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50">
      <Header
        onNewVideo={handleNewVideoRequest}
        canStartOver={!isBusyStage && hasSessionData}
        isBusy={isBusyStage}
      />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {stage === 'input' && (
          <VideoInput onSubmit={handleAnalyze} loading={isUploading} />
        )}

        {stage === 'analyzing' && (
          <ProgressBar
            progress={progress}
            message={statusMessage}
            status={taskStatus}
          />
        )}

        {stage === 'segments' && (
          <SegmentsList
            segments={segments}
            videoTitle={videoData?.title}
            onProcess={handleProcess}
            onDubbing={handleDubbing}
            loading={stage === 'processing'}
            videoThumbnail={videoData?.thumbnail_url}
          />
        )}

        {stage === 'processing' && (
          <ProgressBar
            progress={progress}
            message={statusMessage}
            status={taskStatus}
          />
        )}

        {stage === 'download' && (
          <DownloadList
            processedSegments={processedSegments}
            videoId={videoData?.video_id}
            onReset={handleReset}
            onBackToSegments={() => setStage('segments')}
          />
        )}
      </main>

      <footer className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 mt-12">
        <div className="border-t border-gray-200 pt-6">
          <p className="text-center text-sm text-gray-500">
            Powered by AI: Whisper, DeepSeek, Silero TTS
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;


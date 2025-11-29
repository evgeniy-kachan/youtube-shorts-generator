import React, { useState, useEffect, useMemo } from 'react';
import Header from './components/Header';
import VideoInput from './components/VideoInput';
import ProgressBar from './components/ProgressBar';
import SegmentsList from './components/SegmentsList';
import DownloadList from './components/DownloadList';
import { analyzeLocalVideo, getTaskStatus, processSegments, uploadVideoFile } from './services/api';

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

  // Poll task status
  const buildProcessedSegments = useMemo(
    () => (result) => {
      if (Array.isArray(result?.processed_segments)) {
        return result.processed_segments;
      }

      if (Array.isArray(result?.output_videos)) {
        return result.output_videos.map((relativePath) => {
          const parts = relativePath.split('/');
          const filename = parts[parts.length - 1] || relativePath;
          const segmentId = filename.replace('.mp4', '');
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
          };
        });
      }

      return [];
    },
    [segments]
  );

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
            setVideoData(status.result);
            setSegments(status.result.segments);
            setStage('segments');
            clearInterval(interval);
          } else if (status.status === 'failed') {
            alert('Ошибка при анализе видео: ' + status.message);
            setStage('input');
            clearInterval(interval);
          }
        } catch (error) {
          console.error('Error polling status:', error);
          clearInterval(interval);
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
            clearInterval(interval);
          } else if (status.status === 'failed') {
            alert('Ошибка при обработке видео: ' + status.message);
            setStage('segments');
            clearInterval(interval);
          }
        } catch (error) {
          console.error('Error polling status:', error);
          clearInterval(interval);
        }
      }, 2000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [analysisTask, processingTask, stage, segments, buildProcessedSegments]);

  const handleAnalyze = async (file) => {
    if (!file) {
      alert('Пожалуйста, выберите видеофайл формата MP4');
      return;
    }

    try {
      setIsUploading(true);
      setStage('analyzing');
      setProgress(0.02);
      setStatusMessage('Загружаем файл...');
      setTaskStatus('pending');

      const uploadResponse = await uploadVideoFile(file);
      const uploadedFilename = uploadResponse?.result?.filename;

      if (!uploadedFilename) {
        throw new Error('Сервер не вернул имя загруженного файла');
      }

      setProgress(0.05);
      setStatusMessage('Файл загружен. Начинаем анализ...');

      const response = await analyzeLocalVideo(uploadedFilename);
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
    subtitleFontSize
  ) => {
    try {
      setStage('processing');
      setProgress(0);
      setStatusMessage('Начинаем обработку...');
      setTaskStatus('pending');

      const response = await processSegments(
        videoData.video_id,
        segmentIds,
        verticalMethod,
        subtitleAnimation,
        subtitlePosition,
        subtitleFont,
        subtitleFontSize
      );
      setProcessingTask(response.task_id);
    } catch (error) {
      console.error('Error starting processing:', error);
      alert('Ошибка при запуске обработки: ' + error.message);
      setStage('segments');
    }
  };

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

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50">
      <Header />

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


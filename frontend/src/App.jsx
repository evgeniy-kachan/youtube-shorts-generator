import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import VideoInput from './components/VideoInput';
import ProgressBar from './components/ProgressBar';
import SegmentsList from './components/SegmentsList';
import DownloadList from './components/DownloadList';
import { analyzeVideo, getTaskStatus, processSegments } from './services/api';

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

  // Poll task status
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
            setProcessedSegments(status.result.processed_segments);
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
  }, [analysisTask, processingTask, stage]);

  const handleAnalyze = async (url) => {
    try {
      setStage('analyzing');
      setProgress(0);
      setStatusMessage('Начинаем анализ...');
      setTaskStatus('pending');

      const response = await analyzeVideo(url);
      setAnalysisTask(response.task_id);
    } catch (error) {
      console.error('Error starting analysis:', error);
      alert('Ошибка при запуске анализа: ' + error.message);
      setStage('input');
    }
  };

  const handleProcess = async (segmentIds, verticalMethod) => {
    try {
      setStage('processing');
      setProgress(0);
      setStatusMessage('Начинаем обработку...');
      setTaskStatus('pending');

      const response = await processSegments(videoData.video_id, segmentIds, verticalMethod);
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
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50">
      <Header />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {stage === 'input' && (
          <VideoInput onSubmit={handleAnalyze} loading={false} />
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
            loading={false}
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
          />
        )}
      </main>

      <footer className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 mt-12">
        <div className="border-t border-gray-200 pt-6">
          <p className="text-center text-sm text-gray-500">
            Powered by AI: Whisper, Llama 3.1, NLLB, Silero TTS
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;


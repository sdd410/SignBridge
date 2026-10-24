import { useState, useEffect, useRef, useCallback } from 'react';
import { CameraDevice } from '../types';

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [devices, setDevices] = useState<CameraDevice[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [isMirrored, setIsMirrored] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Enumerate video input devices
  const enumerateDevices = useCallback(async () => {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        setError('MediaDevices API is not supported in this browser.');
        return;
      }
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = allDevices
        .filter((d) => d.kind === 'videoinput')
        .map((d, index) => ({
          deviceId: d.deviceId,
          label: d.label || `Camera ${index + 1}`,
        }));
      setDevices(videoInputs);
      if (videoInputs.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoInputs[0].deviceId);
      }
    } catch (err: unknown) {
      console.warn('Failed to enumerate media devices:', err);
    }
  }, [selectedDeviceId]);

  useEffect(() => {
    enumerateDevices();
  }, [enumerateDevices]);

  // Start webcam feed
  const startCamera = useCallback(async (deviceId?: string) => {
    setError(null);
    const targetId = deviceId || selectedDeviceId;

    try {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
      }

      const constraints: MediaStreamConstraints = {
        video: {
          deviceId: targetId ? { exact: targetId } : undefined,
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30 },
        },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsStreaming(true);
      setIsPaused(false);
      // Re-enumerate to get labeled camera names after permission is granted
      enumerateDevices();
    } catch (err: unknown) {
      console.error('Camera access error:', err);
      let msg = 'Failed to access camera.';
      if (err instanceof DOMException) {
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          msg = 'Camera permission was denied. Please allow camera access in your browser settings.';
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
          msg = 'No video capture devices were detected on this machine.';
        } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
          msg = 'Camera is already in use by another application.';
        }
      }
      setError(msg);
      setIsStreaming(false);
    }
  }, [selectedDeviceId, enumerateDevices]);

  // Stop camera feed
  const stopCamera = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    setIsPaused(false);
  }, []);

  const pauseStream = useCallback(() => {
    setIsPaused(true);
  }, []);

  const resumeStream = useCallback(() => {
    setIsPaused(false);
  }, []);

  const toggleMirror = useCallback(() => {
    setIsMirrored((prev) => !prev);
  }, []);

  // Capture downscaled frame as base64 JPEG
  const captureFrame = useCallback((targetWidth = 320, targetHeight = 240, quality = 0.75): string | null => {
    const video = videoRef.current;
    if (!video || video.readyState < 2 || isPaused) return null;

    if (!canvasRef.current) {
      canvasRef.current = document.createElement('canvas');
    }

    const canvas = canvasRef.current;
    canvas.width = targetWidth;
    canvas.height = targetHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    // Send genuine unmirrored frame to the ML model (mirroring is strictly UI preview)
    ctx.drawImage(video, 0, 0, targetWidth, targetHeight);
    return canvas.toDataURL('image/jpeg', quality);
  }, [isPaused]);

  // Clean up when unmounting
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return {
    videoRef,
    devices,
    selectedDeviceId,
    setSelectedDeviceId,
    isStreaming,
    isPaused,
    isMirrored,
    error,
    startCamera,
    stopCamera,
    pauseStream,
    resumeStream,
    toggleMirror,
    captureFrame,
  };
}

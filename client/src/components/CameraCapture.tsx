import { useRef, useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Camera, RotateCcw, ImageIcon } from "lucide-react";

interface CameraCaptureProps {
  onCapture: (imageBase64: string) => void;
  isProcessing: boolean;
}

export function CameraCapture({ onCapture, isProcessing }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [hasCamera, setHasCamera] = useState(true);
  const [facingMode, setFacingMode] = useState<"user" | "environment">("environment");
  const [flashEffect, setFlashEffect] = useState(false);

  const startCamera = useCallback(async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setStream(mediaStream);
      setHasCamera(true);
    } catch (err) {
      console.error("Camera access denied:", err);
      setHasCamera(false);
    }
  }, [facingMode]);

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
  }, [stream]);

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, [facingMode]);

  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.drawImage(video, 0, 0);
      const imageBase64 = canvas.toDataURL("image/jpeg", 0.9);
      
      setFlashEffect(true);
      setTimeout(() => setFlashEffect(false), 200);
      
      onCapture(imageBase64);
    }
  };

  const switchCamera = () => {
    stopCamera();
    setFacingMode((prev) => (prev === "user" ? "environment" : "user"));
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result as string;
        onCapture(base64);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="relative w-full h-full flex flex-col items-center justify-center">
      <div className="relative w-full max-w-2xl aspect-[4/3] rounded-3xl overflow-hidden bg-muted shadow-xl">
        {hasCamera ? (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
            />
            <canvas ref={canvasRef} className="hidden" />
            
            <AnimatePresence>
              {flashEffect && (
                <motion.div
                  initial={{ opacity: 1 }}
                  animate={{ opacity: 0 }}
                  exit={{ opacity: 0 }}
                  className="absolute inset-0 bg-white"
                />
              )}
            </AnimatePresence>

            <div className="absolute inset-0 border-8 border-primary/20 rounded-3xl pointer-events-none" />
            <div className="absolute top-4 left-4 right-4 flex justify-between pointer-events-none">
              <div className="w-12 h-12 border-t-4 border-l-4 border-primary rounded-tl-2xl" />
              <div className="w-12 h-12 border-t-4 border-r-4 border-primary rounded-tr-2xl" />
            </div>
            <div className="absolute bottom-4 left-4 right-4 flex justify-between pointer-events-none">
              <div className="w-12 h-12 border-b-4 border-l-4 border-primary rounded-bl-2xl" />
              <div className="w-12 h-12 border-b-4 border-r-4 border-primary rounded-br-2xl" />
            </div>
          </>
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center gap-4 p-8">
            <ImageIcon className="w-16 h-16 text-muted-foreground" />
            <p className="text-lg text-muted-foreground text-center">
              Camera not available. Upload a photo instead!
            </p>
            <Button
              onClick={() => fileInputRef.current?.click()}
              data-testid="button-upload-photo"
              className="rounded-full px-8 py-6 text-lg font-bold"
            >
              <ImageIcon className="w-5 h-5 mr-2" />
              Choose Photo
            </Button>
          </div>
        )}
      </div>

      <div className="mt-8 flex items-center gap-6">
        {hasCamera && (
          <Button
            size="icon"
            variant="outline"
            onClick={switchCamera}
            disabled={isProcessing}
            data-testid="button-switch-camera"
            className="rounded-full w-14 h-14"
          >
            <RotateCcw className="w-6 h-6" />
          </Button>
        )}

        <motion.button
          onClick={capturePhoto}
          disabled={isProcessing || !hasCamera}
          data-testid="button-capture"
          className="relative w-24 h-24 rounded-full bg-primary flex items-center justify-center disabled:opacity-50"
          whileTap={{ scale: 0.9 }}
        >
          <div className="absolute inset-2 rounded-full border-4 border-primary-foreground" />
          {isProcessing ? (
            <div className="w-8 h-8 border-4 border-primary-foreground border-t-transparent rounded-full animate-spin" />
          ) : (
            <Camera className="w-8 h-8 text-primary-foreground" />
          )}
          {!isProcessing && (
            <motion.div
              className="absolute inset-0 rounded-full border-4 border-primary"
              animate={{ scale: [1, 1.2, 1], opacity: [1, 0, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
          )}
        </motion.button>

        <Button
          size="icon"
          variant="outline"
          onClick={() => fileInputRef.current?.click()}
          disabled={isProcessing}
          data-testid="button-gallery"
          className="rounded-full w-14 h-14"
        >
          <ImageIcon className="w-6 h-6" />
        </Button>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileUpload}
        className="hidden"
        data-testid="input-file-upload"
      />
    </div>
  );
}

const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Configure multer for video uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = './uploads';
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({
  storage: storage,
  limits: { fileSize: 100 * 1024 * 1024 }, // 100MB limit
  fileFilter: (req, file, cb) => {
    const allowedTypes = /mp4|avi|mov|mkv/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);
    
    if (mimetype && extname) {
      return cb(null, true);
    } else {
      cb(new Error('Only video files are allowed!'));
    }
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', message: 'Deepfake Detection API is running' });
});

// Prediction endpoint
app.post('/api/predict', upload.single('video'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No video file uploaded' });
  }

  const videoPath = req.file.path;
  const modelPath = process.env.MODEL_PATH || './models/model.pt';

  try {
    const result = await runPrediction(videoPath, modelPath);
    
    // Clean up uploaded file
    fs.unlinkSync(videoPath);
    
    res.json({
      success: true,
      prediction: result.prediction,
      confidence: result.confidence,
      label: result.label,
      processingTime: result.processingTime
    });
  } catch (error) {
    // Clean up uploaded file on error
    if (fs.existsSync(videoPath)) {
      fs.unlinkSync(videoPath);
    }
    
    console.error('Prediction error:', error);
    res.status(500).json({ 
      success: false, 
      error: 'Prediction failed', 
      details: error.message 
    });
  }
});

// Batch prediction endpoint
app.post('/api/predict/batch', upload.array('videos', 10), async (req, res) => {
  if (!req.files || req.files.length === 0) {
    return res.status(400).json({ error: 'No video files uploaded' });
  }

  const modelPath = process.env.MODEL_PATH || './models/model.pt';
  const results = [];

  try {
    for (const file of req.files) {
      try {
        const result = await runPrediction(file.path, modelPath);
        results.push({
          filename: file.originalname,
          success: true,
          ...result
        });
      } catch (error) {
        results.push({
          filename: file.originalname,
          success: false,
          error: error.message
        });
      } finally {
        // Clean up file
        if (fs.existsSync(file.path)) {
          fs.unlinkSync(file.path);
        }
      }
    }

    res.json({
      success: true,
      totalVideos: req.files.length,
      results: results
    });
  } catch (error) {
    // Clean up all files on error
    req.files.forEach(file => {
      if (fs.existsSync(file.path)) {
        fs.unlinkSync(file.path);
      }
    });

    res.status(500).json({ 
      success: false, 
      error: 'Batch prediction failed', 
      details: error.message 
    });
  }
});

// Function to run Python prediction script
function runPrediction(videoPath, modelPath) {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();
    
    // Spawn Python process
    const pythonProcess = spawn('python', [
      './predict.py',
      '--video', videoPath,
      '--model', modelPath,
      '--sequence-length', '20'
    ]);

    let outputData = '';
    let errorData = '';

    pythonProcess.stdout.on('data', (data) => {
      outputData += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
    });

    pythonProcess.on('close', (code) => {
      const processingTime = Date.now() - startTime;

      if (code !== 0) {
        return reject(new Error(`Python process exited with code ${code}: ${errorData}`));
      }

      try {
        // Parse output from Python script
        const result = JSON.parse(outputData);
        resolve({
          prediction: result.prediction,
          confidence: result.confidence,
          label: result.prediction === 1 ? 'REAL' : 'FAKE',
          processingTime: processingTime
        });
      } catch (error) {
        reject(new Error(`Failed to parse prediction result: ${error.message}`));
      }
    });

    pythonProcess.on('error', (error) => {
      reject(new Error(`Failed to start Python process: ${error.message}`));
    });
  });
}

// Error handling middleware
app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    return res.status(400).json({ error: err.message });
  }
  res.status(500).json({ error: err.message });
});

app.listen(PORT, () => {
  console.log(`Deepfake Detection API running on port ${PORT}`);
  console.log(`Upload endpoint: http://localhost:${PORT}/api/predict`);
});
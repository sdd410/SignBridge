# MASTER BUILD PROMPT: SignBridge — Real-Time Sign Language Detection & Translation

## 1. Project Overview

Build a complete, functional MVP called **SignBridge**, a real-time sign language gesture recognition and translation application.

The application uses a webcam to capture hand gestures, processes video frames using OpenCV, and classifies supported signs using a Convolutional Neural Network (CNN). Recognized signs are displayed with confidence scores and assembled into a text transcript.

The project should demonstrate practical computer vision, machine learning, real-time video processing, model evaluation, and application development.

### Core objective

Implement this end-to-end pipeline:

Webcam → Frame Processing → Hand Detection and Preprocessing → CNN Classification → Prediction Filtering → Recognized Sign → Text Transcript.

The application must work locally without requiring paid APIs or cloud-based inference.

**Important:** Do not fabricate recognition results. Use a real trained model and clearly communicate which signs are supported. If a trained model or dataset is unavailable, provide a functional training workflow and an explicit demo mode rather than pretending that an untrained model can translate signs.

This is an educational prototype, not a production-grade universal sign language translator.

## 2. Recommended Technology Stack

### Frontend
- React with TypeScript.
- Vite.
- Tailwind CSS.
- shadcn/ui.
- Lucide icons.
- Responsive desktop-first dashboard.
- Browser webcam access using `getUserMedia`.
- WebSocket or HTTP-based communication with the backend.

### Backend
- Python 3.11 or a compatible installed version.
- FastAPI.
- OpenCV for frame processing.
- TensorFlow/Keras or PyTorch for CNN training and inference.
- NumPy.
- Pydantic.
- pytest.

Choose one primary deep learning framework and use it consistently. Prefer TensorFlow/Keras for a straightforward CNN training pipeline unless the existing repository already uses PyTorch.

### Machine Learning
- CNN-based image classification.
- Configurable image preprocessing.
- Train, validation, and test splits.
- Data augmentation.
- Model checkpointing.
- Confidence scores.
- Evaluation metrics and confusion matrix.
- Model version and class-label metadata.

### Persistence
Use local storage or IndexedDB for browser preferences and saved transcripts. Use SQLite only if persistent server-side history or user sessions are needed.

Do not introduce a database unnecessarily.

## 3. Supported Sign Language and Dataset

Do not assume that every sign language uses the same gestures.

Build the MVP around a configurable, explicitly identified dataset.

Recommended initial scope:
- American Sign Language (ASL) alphabet classification.
- Support A–Z, excluding dynamic or otherwise unsupported classes when required by the selected dataset.
- Optionally support a dataset's designated non-letter classes, such as space or delete, only if their labels and meanings are verified.
- Keep the architecture extensible to other datasets, including Indian Sign Language (ISL), through a separate model and label configuration.

Use a publicly available, appropriately licensed dataset, such as the ASL Alphabet dataset, after verifying its source, license, class labels, and limitations.

If the dataset is not already present:
1. Document where to obtain it.
2. Provide a dataset preparation script.
3. Validate the folder structure and labels.
4. Avoid silently downloading large datasets.
5. Make the application usable in inference mode when a compatible trained model is supplied.

Never claim that ASL alphabet recognition is equivalent to full sign language translation.

## 4. Application UI and Design

Create a polished, modern computer vision dashboard.

Visual direction:
- Clean light or dark theme with a professional technology aesthetic.
- Deep navy or neutral backgrounds.
- Cyan and blue accent colors.
- Green for successful recognition.
- Amber for uncertain predictions.
- Red for camera or processing errors.
- Clear typography and well-spaced panels.
- Subtle animations and responsive layout.
- Avoid excessive gradients and decorative elements.

### Header

Display:
- SignBridge logo.
- Application title.
- Subtitle: Real-Time Sign Language Recognition.
- Camera connection status.
- Model loading status.
- Current recognition mode.
- Processing FPS and inference latency.

### Main Camera Panel

Display the live webcam feed with:
- Optional hand bounding box.
- Optional hand landmarks if a landmark detector is configured.
- Predicted sign label.
- Confidence score.
- Recognition status.
- Camera selection.
- Camera start and stop controls.
- Pause and resume controls.
- Snapshot capture.
- Camera error messages.

Allow users to mirror the preview for natural interaction while ensuring that the image sent to the model follows the correct preprocessing convention.

Never apply horizontal flipping blindly: some gestures can change meaning when mirrored.

### Prediction Panel

Show:
- Current predicted sign.
- Top five predictions.
- Confidence percentages.
- Recognition stability.
- Recent predictions.
- Model class labels.

Use a confidence threshold and a short temporal smoothing window to reduce flickering predictions.

Distinguish the highest-probability model output from a sign that has passed the application's acceptance threshold.

### Transcript Panel

Build a live transcript editor that:
- Appends accepted predictions.
- Displays the assembled text.
- Supports manual editing.
- Supports inserting spaces.
- Supports deleting the last character.
- Supports clearing the transcript.
- Supports copying text.
- Supports downloading the transcript as a text file.
- Allows pausing automatic transcript insertion.
- Displays timestamps for recognized signs when enabled.

Because alphabet classifiers produce individual letters, do not insert every repeated prediction into the transcript. Add a configurable acceptance interval, stability requirement, and repeat-lockout mechanism.

Provide manual controls to insert a letter when necessary.

### Settings Panel

Expose all important parameters:
- Camera device.
- Input resolution.
- Target processing FPS.
- Model selection.
- Confidence threshold.
- Temporal smoothing window.
- Required stable prediction duration.
- Repeat-lockout duration.
- Hand detection settings, if applicable.
- Preprocessing options.
- Debug overlay.
- Preview mirroring.
- Transcript insertion mode.

Every setting must have an actual effect. Persist user preferences locally where appropriate.

## 5. Webcam Processing Pipeline

Implement a robust real-time processing pipeline.

### Step 1: Capture Frames

Capture webcam frames at a configurable rate.

Do not send every browser frame to the backend if the processing pipeline cannot handle that rate.

Support camera selection where available.

Handle:
- Permission denied.
- No camera detected.
- Camera already in use.
- Device disconnection.
- Browser incompatibility.
- Backend disconnection.

Release camera resources when the camera is stopped or the application is closed.

### Step 2: Frame Preprocessing

Use OpenCV to implement:
- Frame resizing.
- Color-space conversion.
- Pixel normalization.
- Model-specific image formatting.
- Optional region-of-interest cropping.

Ensure preprocessing matches the trained model's requirements.

If the model expects 64 × 64 RGB images, use that configuration consistently in training and inference. Do not hardcode this size if the loaded model specifies a different input shape.

Keep preprocessing in a reusable module.

### Step 3: Hand Detection

Implement a configurable preprocessing strategy.

Option A:
- Use a dataset containing isolated hand gesture images.
- Classify the appropriate cropped or resized image directly.

Option B:
- Use a hand detection or landmark extraction library.
- Extract the relevant hand region before CNN classification.

For the MVP, prefer the simplest pipeline that matches the selected dataset.

Do not claim that the model detects hands in arbitrary backgrounds if it was trained only on isolated, preprocessed images.

### Step 4: CNN Inference

Load a trained model at backend startup.

Implement:
- Input validation.
- Batch dimension handling.
- Preprocessing.
- Model inference.
- Class-label mapping.
- Softmax probability interpretation where appropriate.
- Top-k predictions.
- Confidence thresholding.
- Inference timing.

Load the model once and reuse it rather than reloading it for each frame.

Return structured results containing:
- Predicted class.
- Predicted label.
- Confidence score.
- Top-k predictions.
- Inference latency.
- Timestamp.
- Model identifier.
- Whether the prediction passed the configured threshold.

If the model is unavailable, report an explicit model-not-ready state.

Do not return random predictions or fabricated confidence scores.

## 6. CNN Model Training

Implement a reproducible training pipeline.

### Model Architecture

Start with a small, practical CNN:
- Input layer.
- Convolutional layers with ReLU activations.
- Max pooling.
- Batch normalization where useful.
- Dropout to reduce overfitting.
- Dense classification head.
- Softmax output for the configured classes.

Choose the architecture based on dataset size and input resolution.

Avoid unnecessarily large networks for an alphabet-classification MVP.

### Training Features

Implement:
- Configurable dataset path.
- Configurable image dimensions.
- Train/validation/test splits.
- Class-label extraction.
- Data augmentation.
- Configurable batch size.
- Configurable number of epochs.
- Learning rate.
- Early stopping.
- Best-model checkpointing.
- Training history.
- Reproducible random seed.
- Class imbalance reporting.

Use augmentation carefully. Avoid transformations that can change the semantic identity of a sign.

Where multiple images originate from the same person, recording session, or near-duplicate source, prevent leakage between training and evaluation splits where the dataset structure allows it.

### Training Outputs

Save:
- Trained model.
- Class-label mapping.
- Model configuration.
- Training metrics.
- Evaluation metrics.
- Confusion matrix.
- Training and validation loss curves.
- Training and validation accuracy curves.
- Dataset metadata.

Never hardcode accuracy values. Generate them from actual evaluation.

## 7. Real-Time Prediction Stabilization

Single-frame classification can fluctuate, so implement a temporal stabilization layer.

Support:
- Sliding window of recent predictions.
- Majority voting or probability averaging.
- Minimum confidence threshold.
- Minimum stability duration.
- Configurable cooldown between accepted predictions.
- Duplicate suppression.
- Manual insertion override.

A sign should be accepted only when the configured stability rules are satisfied.

Expose the stabilized result separately from raw model predictions.

Ensure that users can tune the parameters without retraining the model.

Document the trade-off between responsiveness and stability.

## 8. Text Assembly and Translation Boundaries

The MVP recognizes individual supported signs and assembles them into text.

Implement:
- Letter insertion.
- Word spacing.
- Backspace.
- Clear transcript.
- Manual correction.
- Copy and download.
- Prediction history.

Do not automatically claim grammatical translation from isolated letters.

A later extension may support:
- Word-level sign recognition.
- Dynamic gesture recognition from video sequences.
- Temporal models such as LSTM or Transformer architectures.
- Sentence-level translation using a suitable dataset.
- Text-to-speech output.

Keep these as future extensions unless a valid dataset and trained model are available.

If text-to-speech is implemented, label it as speech synthesis of the assembled text, not speech translation from complete sign language.

## 9. Backend API

Implement FastAPI endpoints.

Suggested endpoints:

- `GET /api/health`
- `GET /api/camera/config`
- `GET /api/model/status`
- `GET /api/model/classes`
- `GET /api/settings`
- `PATCH /api/settings`
- `POST /api/predict`
- `GET /api/predictions/recent`
- `POST /api/transcripts`
- `GET /api/model/metrics`

For real-time inference, choose one consistent frame transport strategy.

Recommended approach:
- Browser captures webcam frames.
- Browser downsizes frames to the configured inference resolution.
- Browser sends JPEG frames to a WebSocket endpoint.
- Backend processes incoming frames.
- Backend returns prediction events.
- Frontend updates the live prediction and transcript.

Suggested WebSocket endpoint:

`/ws/predict`

Implement:
- Input validation.
- Bounded frame sizes.
- Rate limiting or frame throttling.
- Graceful connection handling.
- Model readiness checks.
- Useful error responses.
- Logging without storing webcam footage by default.

For a local single-user MVP, avoid adding authentication unless the deployment context requires it.

Do not expose the camera stream publicly by default.

## 10. Frontend and Backend Integration

Create typed frontend API clients and shared TypeScript interfaces for prediction responses.

The frontend should:
- Manage webcam lifecycle.
- Display live frames.
- Send frames at the configured processing rate.
- Handle backend prediction messages.
- Display current and historical predictions.
- Apply transcript acceptance rules consistently.
- Show connection and model status.
- Reconnect safely after temporary failures.

Prevent stale predictions from appearing as current results.

Do not allow network latency to create duplicate transcript characters.

Keep video capture, rendering, inference communication, and transcript state separate.

## 11. Project Structure

Use a modular structure similar to:

```text
signbridge/
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── camera/
│       │   ├── predictions/
│       │   ├── transcript/
│       │   ├── settings/
│       │   └── dashboard/
│       ├── hooks/
│       ├── services/
│       ├── stores/
│       ├── types/
│       └── utils/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   ├── schemas/
│   │   ├── vision/
│   │   │   ├── preprocessing.py
│   │   │   ├── hand_detection.py
│   │   │   └── camera_utils.py
│   │   ├── inference/
│   │   │   ├── model_loader.py
│   │   │   ├── predictor.py
│   │   │   └── stabilization.py
│   │   ├── transcript/
│   │   └── config/
│   ├── tests/
│   └── requirements.txt
├── ml/
│   ├── dataset/
│   ├── models/
│   ├── scripts/
│   │   ├── prepare_dataset.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── README.md
├── docker-compose.yml
├── .env.example
├── README.md
└── .gitignore
```

Adjust the structure when appropriate for the existing repository.

Do not commit large datasets, model weights, webcam recordings, or generated artifacts to version control.

## 12. Testing

### Unit Tests
- Image preprocessing produces valid model inputs.
- Class-label mapping is correct.
- Confidence thresholds behave correctly.
- Prediction stabilization works.
- Duplicate suppression prevents repeated transcript insertion.
- Transcript editing operations behave correctly.
- Invalid frames are rejected.
- Missing model files produce explicit errors.

### Model Tests
- Training pipeline completes on a small test dataset.
- Saved model can be reloaded.
- Output dimensions match the class configuration.
- Evaluation metrics are generated from actual predictions.
- Inference returns finite, valid scores.

### API Tests
- Health endpoint responds correctly.
- Model status reports loading and failure states.
- Prediction endpoint validates inputs.
- WebSocket returns valid prediction messages.
- Client disconnections are handled.
- Rate limits and frame-size limits are enforced.

### Frontend Tests
- Camera permission errors are visible.
- Camera start/stop works.
- Prediction panels update from backend responses.
- Low-confidence predictions are handled correctly.
- Transcript insertion and editing work.
- Settings update actual application behavior.

Run tests and fix failures. Do not merely create test files.

## 13. Performance and Privacy

- Aim for responsive local inference, with actual performance depending on the model and hardware.
- Downsize frames before transmission.
- Reuse the loaded model.
- Limit inference frequency.
- Avoid unnecessary image copies.
- Keep frame buffers bounded.
- Show measured FPS and inference latency.
- Never display invented performance metrics.
- Do not save or upload camera footage by default.
- Clear temporary frame data when it is no longer needed.
- Make any persistent transcript storage explicit to the user.

## 14. MVP Acceptance Criteria

The MVP is complete only when:

1. The application starts locally using documented instructions.
2. The browser can access a webcam after user permission.
3. The application displays live camera footage.
4. The backend loads a compatible trained CNN model.
5. Incoming frames are processed using OpenCV.
6. Predictions correspond to the loaded model's actual output.
7. The application displays class labels and confidence scores.
8. Temporal stabilization reduces prediction flicker.
9. Accepted signs are assembled into an editable transcript.
10. Users can copy and download the transcript.
11. Settings influence the actual recognition pipeline.
12. Missing models and camera errors are handled correctly.
13. The model training and evaluation scripts are documented.
14. Tests pass and the frontend builds successfully.
15. The README explains dataset limitations and supported signs.

## 15. Implementation Workflow

### Phase 1: Repository Inspection
- Inspect the repository, dependencies, and existing implementation.
- Identify the chosen dataset and model availability.
- Define the API and model interfaces.
- Avoid overwriting unrelated work.

### Phase 2: Machine Learning Pipeline
- Implement dataset preparation.
- Build the CNN training script.
- Implement evaluation and model saving.
- Create a small synthetic test fixture for pipeline testing, without presenting synthetic test accuracy as real recognition performance.

### Phase 3: Inference Backend
- Implement image preprocessing.
- Load the trained model.
- Implement inference and stabilization.
- Add FastAPI endpoints and WebSocket support.
- Test the pipeline independently of the frontend.

### Phase 4: Frontend
- Build the camera dashboard.
- Implement webcam controls.
- Add live prediction and confidence panels.
- Implement the transcript editor.
- Add settings and model status indicators.

### Phase 5: Integration
- Connect webcam frames to the backend.
- Implement stable prediction updates.
- Handle failures and reconnections.
- Verify real end-to-end predictions.

### Phase 6: Verification
- Run backend tests.
- Run model pipeline tests.
- Run frontend tests and production build.
- Test with a real webcam.
- Document supported gestures and limitations.

## 16. Coding Agent Instructions

Work directly in the current repository.

Do not stop after producing a plan, static mockup, or sample predictions.

Build a working end-to-end application incrementally.

Inspect existing dependencies before adding new ones.

Use modular, typed, testable code and sensible error handling.

Do not silently substitute fake predictions when the model is unavailable.

Do not hardcode model accuracy, FPS, inference latency, or recognition results.

If the dataset is missing, make the training workflow usable and explain exactly how to obtain the required dataset and model artifacts.

If full sentence translation is not supported by the selected dataset and model, clearly state that limitation rather than overstating the application's capabilities.

At completion, report:
- Implemented features.
- Model architecture and supported classes.
- Dataset source and licensing information.
- Training and evaluation results, if training was performed.
- Setup and execution commands.
- Test results.
- Known limitations.
- Suggested improvements.

**Final instruction:** Inspect the repository and build SignBridge as a working real-time webcam-to-CNN-to-text application, not a dashboard-only demonstration.
let video;
let overlay;
let isDetectionRunning = false;
let prevDetection = null;
let lastBlinkState = false;
let blinkCount = 0;
let lastBlinkTime = Date.now();
let microMovements = [];
let expressionHistory = [];
let textureHistory = [];
const HISTORY_SIZE = 30;
const BLINK_RESET_TIME = 3000;

const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model/';

async function setupCamera() {
    video = document.getElementById('video');
    overlay = document.getElementById('overlay').getContext('2d');
    overlay.canvas.width = video.width;
    overlay.canvas.height = video.height;
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            }
        });
        video.srcObject = stream;
    } catch (err) {
        console.error("Error accessing camera:", err);
        alert("Failed to access camera. Please ensure you have granted camera permissions.");
        throw err;
    }
    
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function loadFaceDetectionModels() {
    try {
        await Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
            faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
            faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL)
        ]);
    } catch (err) {
        console.error("Error loading models:", err);
        alert("Failed to load face detection models.");
        throw err;
    }
}

function getImageDataFromBox(box) {
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = box.width;
    tempCanvas.height = box.height;
    tempCtx.drawImage(video, box.x, box.y, box.width, box.height, 0, 0, box.width, box.height);
    return tempCtx.getImageData(0, 0, box.width, box.height);
}

function calculateTextureScore(imageData) {
    const data = imageData.data;
    let diffSum = 0;
    const width = imageData.width * 4;
    
    for (let i = 0; i < data.length; i += 4) {
        if (i % width < width - 4) {  // Check horizontal neighbors
            diffSum += Math.abs(data[i] - data[i + 4]);
            diffSum += Math.abs(data[i + 1] - data[i + 5]);
            diffSum += Math.abs(data[i + 2] - data[i + 6]);
        }
        if (i < data.length - width) {  // Check vertical neighbors
            diffSum += Math.abs(data[i] - data[i + width]);
            diffSum += Math.abs(data[i + 1] - data[i + width + 1]);
            diffSum += Math.abs(data[i + 2] - data[i + width + 2]);
        }
    }
    
    return diffSum / (imageData.width * imageData.height);
}

function detectBlink(landmarks) {
    const leftEye = landmarks.getLeftEye();
    const rightEye = landmarks.getRightEye();
    
    const leftEAR = calculateEAR(leftEye);
    const rightEAR = calculateEAR(rightEye);
    
    const averageEAR = (leftEAR + rightEAR) / 2;
    const isBlinking = averageEAR < 0.2;
    
    const currentTime = Date.now();
    if (isBlinking !== lastBlinkState) {
        if (!isBlinking && currentTime - lastBlinkTime > 100) {
            blinkCount++;
            lastBlinkTime = currentTime;
        }
        lastBlinkState = isBlinking;
    }
    
    if (currentTime - lastBlinkTime > BLINK_RESET_TIME) {
        blinkCount = 0;
    }
    
    return {
        blinkCount,
        currentEAR: averageEAR
    };
}

function calculateEAR(eye) {
    const height1 = euclideanDistance(eye[1], eye[5]);
    const height2 = euclideanDistance(eye[2], eye[4]);
    const width = euclideanDistance(eye[0], eye[3]);
    return (height1 + height2) / (2.0 * width);
}

function euclideanDistance(point1, point2) {
    return Math.sqrt(
        Math.pow(point2.x - point1.x, 2) +
        Math.pow(point2.y - point1.y, 2)
    );
}

function detectMicroMovements(currentDetection, prevDetection) {
    if (!prevDetection) return { variance: 0, avgMovement: 0 };
    
    const currentLandmarks = currentDetection.landmarks.positions;
    const prevLandmarks = prevDetection.landmarks.positions;
    
    let movements = [];
    for (let i = 0; i < currentLandmarks.length; i++) {
        const curr = currentLandmarks[i];
        const prev = prevLandmarks[i];
        const distance = Math.sqrt(Math.pow(curr.x - prev.x, 2) + Math.pow(curr.y - prev.y, 2));
        movements.push(distance);
    }
    
    const avgMovement = movements.reduce((a, b) => a + b, 0) / movements.length;
    microMovements.push(avgMovement);
    
    if (microMovements.length > HISTORY_SIZE) {
        microMovements.shift();
    }
    
    const meanMovement = microMovements.reduce((a, b) => a + b, 0) / microMovements.length;
    const variance = microMovements.reduce((a, b) => a + Math.pow(b - meanMovement, 2), 0) / microMovements.length;
    
    return {
        variance,
        avgMovement
    };
}

function analyzeExpressions(expressions) {
    expressionHistory.push(expressions);
    if (expressionHistory.length > HISTORY_SIZE) {
        expressionHistory.shift();
    }
    
    if (expressionHistory.length < 2) return 0;
    
    let totalVariance = 0;
    for (let expr in expressions) {
        const values = expressionHistory.map(e => e[expr]);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
        totalVariance += variance;
    }
    
    return totalVariance / Object.keys(expressions).length;
}

function isLiveFace(blinkData, movementData, expressionVariance, textureScore) {
    const checks = {
        blink: blinkData.blinkCount >= 2,
        movement: movementData.variance >= 0.05 && movementData.variance <= 1.5,
        expression: expressionVariance >= 0.002,
        texture: textureScore >= 15 && textureScore <= 50,
        consistentMovement: microMovements.length >= HISTORY_SIZE
    };
    
    const isLive = Object.values(checks).every(check => check);
    
    return { isLive, checks };
}

async function detectFaces() {
    if (!isDetectionRunning) return;

    const detection = await faceapi.detectSingleFace(video, 
        new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceExpressions();

    overlay.clearRect(0, 0, video.width, video.height);

    if (detection) {
        const resizedDetection = faceapi.resizeResults(detection, {
            width: video.width,
            height: video.height
        });

        faceapi.draw.drawDetections(overlay, [resizedDetection]);
        faceapi.draw.drawFaceLandmarks(overlay, [resizedDetection]);

        const blinkData = detectBlink(detection.landmarks);
        const movementData = detectMicroMovements(detection, prevDetection);
        const expressionVariance = analyzeExpressions(detection.expressions);
        
        const imageData = getImageDataFromBox(detection.detection.box);
        const textureScore = calculateTextureScore(imageData);
        textureHistory.push(textureScore);
        if (textureHistory.length > HISTORY_SIZE) textureHistory.shift();
        
        const avgTextureScore = textureHistory.reduce((a, b) => a + b, 0) / textureHistory.length;
        
        const livenessResult = isLiveFace(blinkData, movementData, expressionVariance, avgTextureScore);
        
        const faceTypeElement = document.getElementById('faceType');
        const statusElement = document.getElementById('detectionStatus');
        
        if (livenessResult.isLive) {
            faceTypeElement.textContent = 'Real Face Detected';
            faceTypeElement.style.color = 'green';
        } else {
            faceTypeElement.textContent = 'Fake Face Detected';
            faceTypeElement.style.color = 'red';
        }
        
        const checksString = Object.entries(livenessResult.checks)
            .map(([key, value]) => `${key}: ${value ? '✅' : '❌'}`)
            .join(', ');
        
        statusElement.textContent = `Checks - ${checksString}`;

        prevDetection = detection;
    } else {
        document.getElementById('faceType').textContent = 'No face detected';
        document.getElementById('detectionStatus').textContent = 'Running';
        prevDetection = null;
    }

    requestAnimationFrame(detectFaces);
}

async function startDetection() {
    if (!isDetectionRunning) {
        isDetectionRunning = true;
        document.getElementById('startButton').disabled = true;
        document.getElementById('stopButton').disabled = false;
        document.getElementById('detectionStatus').textContent = 'Running';
        microMovements = [];
        expressionHistory = [];
        textureHistory = [];
        blinkCount = 0;
        lastBlinkTime = Date.now();
        prevDetection = null;
        detectFaces();
    }
}

function stopDetection() {
    isDetectionRunning = false;
    document.getElementById('startButton').disabled = false;
    document.getElementById('stopButton').disabled = true;
    document.getElementById('detectionStatus').textContent = 'Stopped';
    document.getElementById('faceType').textContent = 'No face detected';
    overlay.clearRect(0, 0, video.width, video.height);
}

async function init() {
    try {
        document.getElementById('detectionStatus').textContent = 'Loading models...';
        await loadFaceDetectionModels();
        await setupCamera();
        
        document.getElementById('startButton').addEventListener('click', startDetection);
        document.getElementById('stopButton').addEventListener('click', stopDetection);
        
        document.getElementById('startButton').disabled = false;
        document.getElementById('detectionStatus').textContent = 'Ready';
    } catch (err) {
        console.error(err);
        document.getElementById('detectionStatus').textContent = 'Error initializing';
    }
}

document.addEventListener('DOMContentLoaded', init);
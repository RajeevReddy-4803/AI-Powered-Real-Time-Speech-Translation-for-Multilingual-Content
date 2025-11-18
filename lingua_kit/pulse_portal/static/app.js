// Module 4 — Real-Time Speech Translator JavaScript
// Backend base URL detection for static hosting (e.g., Netlify). Leave empty for same-origin (Flask).
const API_BASE = (typeof window !== 'undefined' && window.BACKEND_BASE_URL) ? window.BACKEND_BASE_URL.replace(/\/$/, '') : '';
const api = (path) => `${API_BASE}${path}`;

let mediaRecorder;
let audioChunks = [];
let isRecording = false;

// ---------- File Upload Handler ----------
document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById('fileInput');
    const lang = document.getElementById('uploadLang').value;
    const uploadGenderBtn = document.querySelector('#uploadGenderMale.active, #uploadGenderFemale.active');
    const gender = uploadGenderBtn ? uploadGenderBtn.dataset.gender : 'male';
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadStatus = document.getElementById('uploadStatus');
    const uploadResult = document.getElementById('uploadResult');

    if (!fileInput.files.length) {
        alert('Please choose a file first!');
        return;
    }

    // Disable button and show loading
    uploadBtn.disabled = true;
    uploadBtn.textContent = '⏳ Processing...';
    uploadStatus.className = 'status loading show';
    uploadStatus.textContent = '📤 Uploading and processing file...';
    uploadResult.style.display = 'none';

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('lang', lang);
    formData.append('gender', gender);

    try {
        const response = await fetch(api('/upload'), {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            uploadStatus.className = 'status error show';
            uploadStatus.textContent = `❌ Error: ${data.error}`;
        } else {
            // Show results
            document.getElementById('origText').textContent = data.original_text;
            document.getElementById('transText').textContent = data.translated_text;
            document.getElementById('uploadAudio').src = data.audio_url + '?t=' + Date.now();

            uploadResult.style.display = 'block';
            uploadStatus.className = 'status success show';
            uploadStatus.textContent = '✅ Translation complete!';
        }
    } catch (error) {
        uploadStatus.className = 'status error show';
        uploadStatus.textContent = `❌ Error: ${error.message}`;
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = '🚀 Upload & Translate';
    }
});

// File input display and preview
document.getElementById('fileInput').addEventListener('change', (e) => {
    const fileName = document.getElementById('fileName');
    const filePreview = document.getElementById('filePreview');
    const audioPreview = document.getElementById('audioPreview');
    const videoPreview = document.getElementById('videoPreview');
    const previewInfo = document.getElementById('previewInfo');

    if (e.target.files.length > 0) {
        const file = e.target.files[0];
        fileName.textContent = '📁 Selected: ' + file.name;
        fileName.classList.add('show');

        // Show preview section
        filePreview.style.display = 'block';

        // Create object URL for preview
        const fileURL = URL.createObjectURL(file);
        const isAudio = file.type.startsWith('audio/');
        const isVideo = file.type.startsWith('video/');

        // Clean up previous URLs
        if (audioPreview.src && audioPreview.src.startsWith('blob:')) {
            URL.revokeObjectURL(audioPreview.src);
        }
        if (videoPreview.src && videoPreview.src.startsWith('blob:')) {
            URL.revokeObjectURL(videoPreview.src);
        }

        // Show appropriate preview
        if (isAudio) {
            audioPreview.src = fileURL;
            audioPreview.style.display = 'block';
            videoPreview.style.display = 'none';
            const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
            previewInfo.textContent = `📻 Audio File: ${file.name} (${fileSizeMB} MB) - Click play to listen`;
        } else if (isVideo) {
            videoPreview.src = fileURL;
            videoPreview.style.display = 'block';
            audioPreview.style.display = 'none';
            const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
            previewInfo.textContent = `🎬 Video File: ${file.name} (${fileSizeMB} MB) - Click play to watch`;
        } else {
            audioPreview.style.display = 'none';
            videoPreview.style.display = 'none';
            previewInfo.textContent = `📁 File: ${file.name} - Preview not available for this file type`;
        }
    } else {
        fileName.classList.remove('show');
        filePreview.style.display = 'none';

        // Clean up URLs
        if (audioPreview.src && audioPreview.src.startsWith('blob:')) {
            URL.revokeObjectURL(audioPreview.src);
            audioPreview.src = '';
        }
        if (videoPreview.src && videoPreview.src.startsWith('blob:')) {
            URL.revokeObjectURL(videoPreview.src);
            videoPreview.src = '';
        }
    }
});

// ---------- Microphone Recording Handler ----------
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const liveLang = document.getElementById('liveLang');
const liveStatus = document.getElementById('liveStatus');
const liveResult = document.getElementById('liveResult');

startBtn.addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            await processAudioBlob(audioBlob);

            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        isRecording = true;

        startBtn.disabled = true;
        stopBtn.disabled = false;
        liveStatus.className = 'status loading show';
        liveStatus.textContent = '🎤 Recording... Click Stop when done.';

    } catch (error) {
        alert('Microphone access denied. Please allow microphone access and try again.');
        console.error('Error accessing microphone:', error);
    }
});

stopBtn.addEventListener('click', () => {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        liveStatus.className = 'status loading show';
        liveStatus.textContent = '⏳ Processing audio...';
    }
});

async function processAudioBlob(audioBlob) {
    try {
        // Create FormData from blob
        const liveGenderBtn = document.querySelector('#liveGenderMale.active, #liveGenderFemale.active');
        const gender = liveGenderBtn ? liveGenderBtn.dataset.gender : 'male';
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.wav');
        formData.append('lang', liveLang.value);
        formData.append('gender', gender);

        const response = await fetch(api('/upload'), {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            liveStatus.className = 'status error show';
            liveStatus.textContent = `❌ Error: ${data.error}`;
        } else {
            // Show results
            document.getElementById('liveText').textContent = data.original_text;
            document.getElementById('translatedText').textContent = data.translated_text;
            document.getElementById('liveAudio').src = data.audio_url + '?t=' + Date.now();

            liveResult.style.display = 'block';
            liveStatus.className = 'status success show';
            liveStatus.textContent = '✅ Translation complete!';
        }
    } catch (error) {
        liveStatus.className = 'status error show';
        liveStatus.textContent = `❌ Error: ${error.message}`;
    }
}

// ---------- Alternative: Text-only translation (for external speech recognition) ----------
async function translateTextOnly(text, lang) {
    try {
        const response = await fetch(api('/translate_text'), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text, lang: lang })
        });

        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Translation error:', error);
        return null;
    }
}

// Export for potential future use
window.translateTextOnly = translateTextOnly;

// ---------- Gender Button Handler ----------
function setupGenderButtons() {
    // Handle gender buttons for upload section
    const uploadGenderButtons = document.querySelectorAll('#uploadGenderMale, #uploadGenderFemale');
    uploadGenderButtons.forEach(btn => {
        btn.addEventListener('click', function () {
            uploadGenderButtons.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
        });
    });

    // Handle gender buttons for live section
    const liveGenderButtons = document.querySelectorAll('#liveGenderMale, #liveGenderFemale');
    liveGenderButtons.forEach(btn => {
        btn.addEventListener('click', function () {
            liveGenderButtons.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
        });
    });
}

// Initialize gender buttons when page loads
document.addEventListener('DOMContentLoaded', setupGenderButtons);

// ---------- YouTube Translation Handler ----------
document.addEventListener('DOMContentLoaded', () => {
    const ytStartBtn = document.getElementById('ytStartBtn');
    if (!ytStartBtn) return;

    const ytGenderButtons = document.querySelectorAll('#ytGenderMale, #ytGenderFemale');
    ytGenderButtons.forEach(btn => {
        btn.addEventListener('click', function () {
            ytGenderButtons.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
        });
    });

    ytStartBtn.addEventListener('click', async () => {
        const url = document.getElementById('ytUrl').value.trim();
        const lang = document.getElementById('ytLang').value;
        const activeGenderBtn = document.querySelector('#ytGenderMale.active, #ytGenderFemale.active');
        const gender = activeGenderBtn ? activeGenderBtn.dataset.gender : 'male';
        const resultsBox = document.getElementById('ytResults');

        if (!url) {
            alert('Please enter a YouTube link');
            return;
        }

        ytStartBtn.disabled = true;
        ytStartBtn.textContent = '⏳ Processing...';
        resultsBox.style.display = 'block';
        resultsBox.className = 'result-box';
        resultsBox.innerHTML = '<div class="status loading show">📥 Downloading and processing audio...</div>';

        try {
            const res = await fetch(api('/youtube_translate'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url, lang, gender })
            });
            let data;
            try {
                data = await res.json();
            } catch (e) {
                const text = await res.text();
                throw new Error(`${res.status} ${res.statusText}: ${text}`);
            }
            if (!res.ok) {
                throw new Error(data && data.error ? data.error : `${res.status} ${res.statusText}`);
            }
            if (data.error) {
                resultsBox.innerHTML = `<div class="status error show">❌ ${data.error}</div>`;
                return;
            }
            let html = '';
            data.forEach(chunk => {
                html += `
                    <div class="text-result" style="margin-bottom:10px; border-bottom:1px solid #eee; padding-bottom:10px;">
                        <p><strong>${chunk.start}s</strong></p>
                        <p>🗣 ${chunk.original || ''}</p>
                        <p>🌐 ${chunk.translated || ''}</p>
                        ${chunk.audio ? `<audio controls src="${chunk.audio}?t=${Date.now()}" style="width:100%;"></audio>` : ''}
                    </div>`;
            });
            resultsBox.innerHTML = html || '<div class="status error show">No speech detected.</div>';
        } catch (err) {
            resultsBox.innerHTML = `<div class="status error show">❌ ${err.message}</div>`;
        } finally {
            ytStartBtn.disabled = false;
            ytStartBtn.textContent = '🎧 Start Translation';
        }
    });
});


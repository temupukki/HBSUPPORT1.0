
window.huluVoiceRecorder = (() => {
    let audioContext = null;
    let mediaStream = null;
    let source = null;
    let workletNode = null;
    let silentGain = null;

    let recordedBuffers = [];
    let recordingSampleRate = 48000;
    let recordingStartedAt = 0;
    let isRecording = false;

    async function start() {
        if (mediaStream || isRecording) {
            throw new Error("Recording is already active.");
        }

        console.log("HuluVoice: requesting microphone...");

        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: { ideal: 1 },

                    // Diagnostic mode: capture audio without
                    // requesting browser speech enhancements.
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false
                }
            });

            const track = mediaStream.getAudioTracks()[0];

            console.log("HuluVoice: microphone track:", {
                label: track.label,
                settings: track.getSettings(),
                constraints: track.getConstraints()
            });

            const AudioContextClass =
                window.AudioContext || window.webkitAudioContext;

            audioContext = new AudioContextClass();

            if (audioContext.state === "suspended") {
                await audioContext.resume();
            }

            recordingSampleRate = audioContext.sampleRate;

            console.log("HuluVoice: AudioContext:", {
                sampleRate: recordingSampleRate,
                state: audioContext.state
            });

            const workletCode = `
                class HuluVoiceProcessor extends AudioWorkletProcessor {
                    constructor() {
                        super();
                        this.stopped = false;

                        this.port.onmessage = (event) => {
                            if (event.data === "stop") {
                                this.stopped = true;
                            }
                        };
                    }

                    process(inputs, outputs) {
                        const input = inputs[0];

                        if (
                            !this.stopped &&
                            input &&
                            input.length > 0 &&
                            input[0] &&
                            input[0].length > 0
                        ) {
                            // Copy the mono channel before transferring it.
                            const samples = new Float32Array(input[0]);

                            this.port.postMessage(
                                samples,
                                [samples.buffer]
                            );
                        }

                        // Keep the processor alive until the main thread
                        // disconnects it.
                        return true;
                    }
                }

                registerProcessor(
                    "hulu-voice-processor",
                    HuluVoiceProcessor
                );
            `;

            const workletBlob = new Blob(
                [workletCode],
                { type: "application/javascript" }
            );

            const workletUrl = URL.createObjectURL(workletBlob);

            try {
                await audioContext.audioWorklet.addModule(workletUrl);
            } finally {
                URL.revokeObjectURL(workletUrl);
            }

            source = audioContext.createMediaStreamSource(mediaStream);

            workletNode = new AudioWorkletNode(
                audioContext,
                "hulu-voice-processor",
                {
                    numberOfInputs: 1,
                    numberOfOutputs: 1,
                    outputChannelCount: [1],
                    channelCount: 1,
                    channelCountMode: "explicit",
                    channelInterpretation: "speakers"
                }
            );

            recordedBuffers = [];

            workletNode.port.onmessage = (event) => {
                const samples = event.data;

                if (
                    samples instanceof Float32Array &&
                    samples.length > 0
                ) {
                    // Keep a separate copy in the main thread.
                    recordedBuffers.push(new Float32Array(samples));
                }
            };

            silentGain = audioContext.createGain();
            silentGain.gain.value = 0;

            source.connect(workletNode);
            workletNode.connect(silentGain);
            silentGain.connect(audioContext.destination);

            recordingStartedAt = performance.now();
            isRecording = true;

            console.log("HuluVoice: recording started.");
        } catch (error) {
            cleanup();
            throw error;
        }
    }

    async function stop() {
        if (!mediaStream || !isRecording) {
            throw new Error("Recording is not active.");
        }

        console.log("HuluVoice: stopping recording...");

        // Tell the processor to stop sending new audio chunks.
        if (workletNode) {
            workletNode.port.postMessage("stop");
        }

        // Allow already queued worklet messages to reach the main thread.
        await new Promise(resolve => setTimeout(resolve, 150));

        isRecording = false;

        const buffers = recordedBuffers.slice();
        const sampleRate = recordingSampleRate;
        const durationSeconds =
            (performance.now() - recordingStartedAt) / 1000;

        let totalSamples = 0;

        for (const buffer of buffers) {
            totalSamples += buffer.length;
        }

        if (totalSamples === 0) {
            cleanup();
            throw new Error("No audio data was recorded.");
        }

        const samples = new Float32Array(totalSamples);
        let offset = 0;

        for (const buffer of buffers) {
            samples.set(buffer, offset);
            offset += buffer.length;
        }

        let peak = 0;
        let sumSquares = 0;
        let clippedSamples = 0;

        for (let i = 0; i < samples.length; i++) {
            const value = samples[i];

            if (!Number.isFinite(value)) {
                cleanup();
                throw new Error("Recording contains invalid audio samples.");
            }

            const absolute = Math.abs(value);

            if (absolute > peak) {
                peak = absolute;
            }

            sumSquares += value * value;

            if (absolute >= 0.999) {
                clippedSamples++;
            }
        }

        const rms = Math.sqrt(sumSquares / samples.length);
        const audioDuration = samples.length / sampleRate;

        console.log("HuluVoice: recording diagnostics:", {
            sampleRate,
            durationSeconds: Number(durationSeconds.toFixed(2)),
            audioDuration: Number(audioDuration.toFixed(2)),
            totalSamples,
            bufferCount: buffers.length,
            peak: Number(peak.toFixed(5)),
            rms: Number(rms.toFixed(5)),
            clippedSamples
        });

        if (audioDuration < 0.5 || peak < 0.005 || rms < 0.001) {
            cleanup();
            throw new Error(
                "Recording is too short or too quiet. " +
                "Check your microphone and try again."
            );
        }

        try {
            const wavBlob = createWavBlob(samples, sampleRate);

            console.log("HuluVoice: WAV created:", {
                sizeBytes: wavBlob.size,
                type: wavBlob.type
            });

            return wavBlob;
        } finally {
            cleanup();
        }
    }

    async function upload(conversationHistory) {
        const wavBlob = await stop();

        console.log("HuluVoice: uploading WAV to FastAPI...");

        const formData = new FormData();

        formData.append("audio", wavBlob, "voice.wav");
        formData.append(
            "conversation_history",
            JSON.stringify(conversationHistory || [])
        );

        const response = await fetch(
            "http://127.0.0.1:8001/api/v1/voice/chat",
            {
                method: "POST",
                body: formData
            }
        );

        const responseText = await response.text();

        console.log(
            `HuluVoice: API response status = ${response.status}`
        );

        if (!response.ok) {
            throw new Error(
                `Voice API returned ${response.status}: ${responseText}`
            );
        }

        return responseText;
    }

    function createWavBlob(samples, sampleRate) {
        const bytesPerSample = 2;
        const dataSize = samples.length * bytesPerSample;

        const buffer = new ArrayBuffer(44 + dataSize);
        const view = new DataView(buffer);

        writeString(view, 0, "RIFF");
        view.setUint32(4, 36 + dataSize, true);
        writeString(view, 8, "WAVE");

        writeString(view, 12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true); // PCM
        view.setUint16(22, 1, true); // Mono
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * bytesPerSample, true);
        view.setUint16(32, bytesPerSample, true);
        view.setUint16(34, 16, true);

        writeString(view, 36, "data");
        view.setUint32(40, dataSize, true);

        let offset = 44;

        for (let i = 0; i < samples.length; i++) {
            const sample = Math.max(-1, Math.min(1, samples[i]));

            const pcmValue = sample < 0
                ? Math.round(sample * 32768)
                : Math.round(sample * 32767);

            view.setInt16(offset, pcmValue, true);
            offset += 2;
        }

        return new Blob([buffer], { type: "audio/wav" });
    }

    function writeString(view, offset, value) {
        for (let i = 0; i < value.length; i++) {
            view.setUint8(offset + i, value.charCodeAt(i));
        }
    }

    function cleanup() {
        isRecording = false;

        if (workletNode) {
            workletNode.port.onmessage = null;

            try {
                workletNode.disconnect();
            } catch {}

            workletNode = null;
        }

        if (source) {
            try {
                source.disconnect();
            } catch {}

            source = null;
        }

        if (silentGain) {
            try {
                silentGain.disconnect();
            } catch {}

            silentGain = null;
        }

        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }

        if (audioContext) {
            const context = audioContext;
            audioContext = null;

            if (context.state !== "closed") {
                context.close().catch(error => {
                    console.warn("HuluVoice: closing audio context failed.", error);
                });
            }
        }

        recordedBuffers = [];
    }

    return { start, stop, upload };
})();

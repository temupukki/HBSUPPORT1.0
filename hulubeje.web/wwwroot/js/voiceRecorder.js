window.huluVoiceRecorder = (() => {

    let audioContext = null;
    let mediaStream = null;
    let source = null;
    let workletNode = null;

    let recordedBuffers = [];
    let recordingSampleRate = 48000;

    async function start() {

        if (mediaStream) {
            throw new Error(
                "Recording is already active."
            );
        }

        console.log(
            "HuluVoice: requesting microphone..."
        );

        mediaStream =
            await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

        console.log(
            "HuluVoice: microphone permission granted."
        );

        const AudioContextClass =
            window.AudioContext ||
            window.webkitAudioContext;

        audioContext =
            new AudioContextClass();

        recordingSampleRate =
            audioContext.sampleRate;

        console.log(
            `HuluVoice: AudioContext sample rate = ${recordingSampleRate}`
        );

        if (
            audioContext.state === "suspended"
        ) {
            await audioContext.resume();
        }

        console.log(
            `HuluVoice: AudioContext state = ${audioContext.state}`
        );

        const workletCode = `
            class HuluVoiceProcessor extends AudioWorkletProcessor {

                process(inputs) {

                    const input =
                        inputs[0];

                    if (
                        input &&
                        input.length > 0 &&
                        input[0]
                    ) {

                        const samples =
                            new Float32Array(
                                input[0]
                            );

                        this.port.postMessage(
                            samples,
                            [samples.buffer]
                        );
                    }

                    return true;
                }
            }

            registerProcessor(
                "hulu-voice-processor",
                HuluVoiceProcessor
            );
        `;

        const blob =
            new Blob(
                [workletCode],
                {
                    type: "application/javascript"
                }
            );

        const workletUrl =
            URL.createObjectURL(blob);

        try {

            await audioContext.audioWorklet.addModule(
                workletUrl
            );

        }
        finally {

            URL.revokeObjectURL(
                workletUrl
            );
        }

        source =
            audioContext.createMediaStreamSource(
                mediaStream
            );

        workletNode =
            new AudioWorkletNode(
                audioContext,
                "hulu-voice-processor",
                {
                    numberOfInputs: 1,
                    numberOfOutputs: 1,
                    channelCount: 1,
                    channelCountMode: "explicit",
                    channelInterpretation: "speakers"
                }
            );

        recordedBuffers = [];

        workletNode.port.onmessage =
            (event) => {

                const samples =
                    event.data;

                if (
                    samples &&
                    samples.length > 0
                ) {

                    recordedBuffers.push(
                        new Float32Array(samples)
                    );
                }
            };

        source.connect(
            workletNode
        );

        /*
         * Keep the AudioWorklet alive.
         *
         * The gain is zero so the user's
         * microphone audio is not played
         * back through the speakers.
         */
        const silentGain =
            audioContext.createGain();

        silentGain.gain.value = 0;

        workletNode.connect(
            silentGain
        );

        silentGain.connect(
            audioContext.destination
        );

        console.log(
            "HuluVoice: AudioWorklet recorder started."
        );
    }

    async function stop() {

        if (!mediaStream) {
            throw new Error(
                "Recording is not active."
            );
        }

        console.log(
            "HuluVoice: stopping recording..."
        );

        /*
         * Give the AudioWorklet a short moment
         * to deliver the final audio buffers.
         */
        await new Promise(
            resolve =>
                setTimeout(
                    resolve,
                    200
                )
        );

        const buffers =
            recordedBuffers;

        const sampleRate =
            recordingSampleRate;

        console.log(
            `HuluVoice: recorded buffers = ${buffers.length}`
        );

        if (
            !buffers ||
            buffers.length === 0
        ) {

            cleanup();

            throw new Error(
                "No audio data was recorded."
            );
        }

        let totalSamples = 0;

        for (
            const buffer of buffers
        ) {
            totalSamples +=
                buffer.length;
        }

        console.log(
            `HuluVoice: total samples = ${totalSamples}`
        );

        const wavBlob =
            createWavBlob(
                buffers,
                sampleRate
            );

        console.log(
            `HuluVoice: WAV size = ${wavBlob.size} bytes`
        );

        cleanup();

        return wavBlob;
    }

    async function upload(
        conversationHistory
    ) {

        const wavBlob =
            await stop();

        console.log(
            "HuluVoice: uploading WAV directly to FastAPI..."
        );

        const formData =
            new FormData();

        formData.append(
            "audio",
            wavBlob,
            "voice.wav"
        );

        formData.append(
            "conversation_history",
            JSON.stringify(
                conversationHistory || []
            )
        );

        const response =
            await fetch(
                "http://127.0.0.1:8001/api/v1/voice/chat",
                {
                    method: "POST",
                    body: formData
                }
            );

        console.log(
            `HuluVoice: FastAPI response = ${response.status}`
        );

        const responseText =
            await response.text();

        if (!response.ok) {

            throw new Error(
                `Voice API returned ${response.status}: ${responseText}`
            );
        }

        console.log(
            "HuluVoice: voice response received."
        );

        return responseText;
    }

    function createWavBlob(
        buffers,
        sampleRate
    ) {

        let totalLength = 0;

        for (
            const buffer of buffers
        ) {

            totalLength +=
                buffer.length;
        }

        const samples =
            new Float32Array(
                totalLength
            );

        let offset = 0;

        for (
            const buffer of buffers
        ) {

            samples.set(
                buffer,
                offset
            );

            offset +=
                buffer.length;
        }

        const wavBuffer =
            new ArrayBuffer(
                44 +
                samples.length * 2
            );

        const view =
            new DataView(
                wavBuffer
            );

        writeString(
            view,
            0,
            "RIFF"
        );

        view.setUint32(
            4,
            36 +
                samples.length * 2,
            true
        );

        writeString(
            view,
            8,
            "WAVE"
        );

        writeString(
            view,
            12,
            "fmt "
        );

        view.setUint32(
            16,
            16,
            true
        );

        view.setUint16(
            20,
            1,
            true
        );

        view.setUint16(
            22,
            1,
            true
        );

        view.setUint32(
            24,
            sampleRate,
            true
        );

        view.setUint32(
            28,
            sampleRate * 2,
            true
        );

        view.setUint16(
            32,
            2,
            true
        );

        view.setUint16(
            34,
            16,
            true
        );

        writeString(
            view,
            36,
            "data"
        );

        view.setUint32(
            40,
            samples.length * 2,
            true
        );

        let index = 44;

        for (
            let i = 0;
            i < samples.length;
            i++
        ) {

            let sample =
                samples[i];

            sample =
                Math.max(
                    -1,
                    Math.min(
                        1,
                        sample
                    )
                );

            const intSample =
                sample < 0
                    ? sample * 0x8000
                    : sample * 0x7FFF;

            view.setInt16(
                index,
                intSample,
                true
            );

            index += 2;
        }

        return new Blob(
            [wavBuffer],
            {
                type: "audio/wav"
            }
        );
    }

    function cleanup() {

        if (workletNode) {

            try {
                workletNode.port.onmessage = null;
            }
            catch {
                // Ignore cleanup error.
            }

            try {
                workletNode.disconnect();
            }
            catch {
                // Already disconnected.
            }
        }

        if (source) {

            try {
                source.disconnect();
            }
            catch {
                // Already disconnected.
            }
        }

        if (mediaStream) {

            mediaStream
                .getTracks()
                .forEach(
                    track => track.stop()
                );
        }

        if (audioContext) {

            try {

                if (
                    audioContext.state !==
                    "closed"
                ) {

                    audioContext.close();
                }

            }
            catch {
                // Ignore cleanup error.
            }
        }

        workletNode = null;
        source = null;
        audioContext = null;
        mediaStream = null;
        recordedBuffers = [];
    }

    function writeString(
        view,
        offset,
        value
    ) {

        for (
            let i = 0;
            i < value.length;
            i++
        ) {

            view.setUint8(
                offset + i,
                value.charCodeAt(i)
            );
        }
    }

    return {
        start,
        stop,
        upload
    };

})();
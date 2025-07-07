# Pipecat Library Memory

## IMPORTANT: Memory Maintenance Protocol
**When to update this file:**
- Learning about Pipecat's pipeline architecture
- Discovering frame processing patterns
- Understanding transport mechanisms
- Finding Pipecat-specific conventions
- Learning about service integrations

**Keep entries:**
- Pipecat-specific only
- Focused on library patterns
- Generic enough for reuse

## Core Architecture

### Pipeline System
- Frame-based processing pipeline
- Processors chain together with `link()`
- Each processor handles specific frame types
- Frames flow through pipeline sequentially

### Frame Types
- InputAudioRawFrame - Raw audio from input transports (SystemFrame)
- OutputAudioRawFrame - Raw audio to output transports (DataFrame)
- TTSAudioRawFrame - TTS-generated audio (extends OutputAudioRawFrame)
- TextFrame - Text messages
- UserStartedSpeakingFrame - Voice activity detection
- UserStoppedSpeakingFrame - Voice inactivity detection
- TranscriptionFrame - STT results with user_id and timestamp
- LLMMessagesFrame - LLM context messages
- TTSStartedFrame/TTSStoppedFrame - TTS processing boundaries

### Base Classes
- FrameProcessor - Base for all processors
- BaseTransport - Base for transports
- BaseService - Base for services

## Pipeline Creation and Configuration

### Where Pipelines are Created
1. **Main Entry Points**:
   - `/dograh/api/services/pipecat/run_pipeline.py` - Primary pipeline creation
   - Functions: `run_pipeline_twilio`, `run_pipeline_smallwebrtc`, `run_pipeline_ari_stasis`
   - All delegate to `_run_pipeline` for actual pipeline creation

2. **Pipeline Builder**:
   - `/dograh/api/services/pipecat/pipeline_builder.py`
   - `build_pipeline()` - Assembles processors in correct order
   - `create_pipeline_components()` - Creates audio buffers, transcript, context
   - `create_pipeline_task()` - Creates PipelineTask with params

3. **Context Aggregator Creation**:
   ```python
   # Pattern: LLM service creates aggregator pair
   context_aggregator = llm.create_context_aggregator(context)
   user_context_aggregator = context_aggregator.user()
   assistant_context_aggregator = context_aggregator.assistant()
   ```

### Pipeline Processor Order
The standard pipeline order (from `build_pipeline()`):
1. `transport.input()` - Receives user input
2. `stt_mute_filter` - Controls when STT is active
3. `audio_buffer.input()` - Records input audio
4. `stt` - Speech-to-text conversion
5. `user_idle_disconnect` - Handles user inactivity
6. `transcript.user()` - Tracks user transcripts
7. `user_context_aggregator` - Aggregates user messages
8. `llm` - Language model processing
9. `pipeline_engine_callback_processor` - Workflow callbacks
10. `tts` - Text-to-speech conversion
11. `bot_idle_filler_sound` - Optional idle sounds
12. `transport.output()` - Sends bot output
13. `audio_buffer.output()` - Records output audio
14. `transcript.assistant()` - Tracks assistant transcripts
15. `assistant_context_aggregator` - Aggregates assistant messages
16. `pipeline_metrics_aggregator` - Collects metrics

### Key Integration Points
1. **Assistant Aggregator Event**:
   ```python
   @assistant_context_aggregator.event_handler("on_push_aggregation")
   async def on_assistant_aggregator_push_context(_aggregator):
       # Triggers workflow transitions after assistant speaks
       await engine.flush_pending_transitions(source="context_push")
   ```

2. **Audio Configuration**:
   - Different configs for Twilio, WebRTC, Stasis
   - Sets sample rates, buffer sizes
   - Created via `create_audio_config(transport_type)`

3. **Service Factory Pattern**:
   - `create_llm_service()`, `create_stt_service()`, `create_tts_service()`
   - Based on user configuration from database

## Context Aggregators

### LLMUserContextAggregator
Aggregates user transcriptions and manages conversation context:

**Key Features**:
1. **Transcription Aggregation**:
   - Collects transcriptions between VAD events
   - Handles interim vs final transcriptions
   - Configurable aggregation timeout (default 0.5s)

2. **VAD Emulation**:
   - Emulates UserStarted/StoppedSpeakingFrame if VAD misses
   - Only emulates when bot is not speaking
   - Prevents interrupting bot with whispers

3. **Interruption Support**:
   - Checks interruption strategies before pushing aggregation
   - Sends BotInterruptionFrame when conditions met
   - Resets aggregation if interruption denied

4. **Event Flow**:
   ```python
   # Process transcriptions
   TranscriptionFrame -> aggregate text
   InterimTranscriptionFrame -> set flag, wait for final
   
   # Push aggregation when:
   - User stops speaking (VAD)
   - Aggregation timeout expires
   - Interruption conditions met
   ```

### LLMAssistantContextAggregator
Aggregates assistant responses and manages function calls:

**Key Features**:
1. **Text Aggregation**:
   - Collects LLMTextFrame tokens
   - Between LLMFullResponseStart/EndFrame
   - Handles stripped vs non-stripped words

2. **Function Call Management**:
   - Tracks function calls in progress
   - Reorders messages for proper context
   - Handles cancellation on interruption

3. **Message Reordering**:
   - Ensures text comes before function messages
   - Uses stable tool_call_ids for tracking
   - Maintains conversation flow integrity

4. **Event Notification**:
   - Fires `on_push_aggregation` event
   - Used by PipecatEngine for workflow transitions
   - Enables deferred transition execution

## Creating Custom Components

### Custom Frame Processor
```python
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import Frame

class MyProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Process or observe frame
        await self.push_frame(frame, direction)
```

### Custom Transport
- Inherit from BaseTransport
- Implement input/output handling
- Handle connection lifecycle
- Deep copy frames between transports

## Transport Patterns

### Internal Transport
- For in-memory agent communication
- No network overhead (unless simulated)
- Direct frame passing
- Use InternalTransportManager for pairing
- **Network Latency Simulation**:
  ```python
  # Add latency to simulate real-world network conditions
  transport_manager.create_transport_pair(
      test_session_id="test_123",
      actor_params=transport_params,
      adversary_params=transport_params,
      latency_seconds=0.1  # 100ms latency
  )
  ```
  - **Fixed delay implementation** - packets are delayed by exact latency amount
  - Uses separate latency processor task to prevent accumulating delays
  - Maintains packet order (FIFO delivery after delay)
  - Configurable per transport pair
  - Useful for testing real-world scenarios
  - Implementation details:
    - Packets timestamped on arrival with delivery time
    - Latency processor checks every 5ms for packets ready to deliver
    - No accumulation - each packet delayed exactly once

### WebRTC Transport
- SmallWebRTCTransport for browser communication
- Handles audio/video streams
- Signaling via websocket
- ICE negotiation built-in

## Service Integration

### Service Layer Patterns
- Services extend FrameProcessor
- Async streaming for real-time performance
- Built-in error handling and retry logic
- Metrics and observability hooks
- Configuration-driven service selection

### Common Services
- STT: Deepgram, Azure, etc.
- LLM: OpenAI, Anthropic, etc.
- TTS: ElevenLabs, PlayHT, etc.

### Service Factory Pattern
- Services created from config
- Use SimpleNamespace for configs
- Services are stateful
- Initialize once per pipeline

### Plugin Architecture
- Clean interfaces for new services
- Adapter pattern for AI providers
- Add providers without framework changes
- Configuration-driven selection

## Frame Processing

### Direction
- FrameDirection.DOWNSTREAM - To output
- FrameDirection.UPSTREAM - From input

### Pass-through Pattern
```python
async def process_frame(self, frame: Frame, direction: FrameDirection):
    # Observe without modifying
    if isinstance(frame, TextFrame):
        self._handle_text(frame.text)
    
    # Always forward frame
    await self.push_frame(frame, direction)
```

### Frame Lifecycle
1. Input transport receives data
2. Converts to appropriate frame
3. Pushes through pipeline
4. Processors handle/modify/observe
5. Output transport sends result

### Audio Frame Flow

#### InputAudioRawFrame Flow
1. **Produced by**: Input transports (BaseInputTransport subclasses)
   - From microphone/WebRTC/phone connections
   - Includes transport_source identifier
   - SystemFrame type (processed immediately, not queued)
2. **Consumed by**:
   - VAD analyzer (if enabled) for speech detection
   - STT services for transcription
   - Audio filters/processors
   - Passed through if audio_in_passthrough enabled

#### OutputAudioRawFrame Flow
1. **Produced by**:
   - TTS services (as TTSAudioRawFrame subclass)
   - Audio mixing/processing components
   - Any processor generating audio for playback
2. **Features**:
   - DataFrame type (queued and ordered)
   - Includes transport_destination for routing
   - Contains sample_rate, num_channels, audio bytes
3. **Consumed by**:
   - Output transports (BaseOutputTransport subclasses)
   - Audio mixers (if configured)
   - Sent to speakers/WebRTC/phone connections

#### Key Differences
- InputAudioRawFrame: SystemFrame, immediate processing, from users
- OutputAudioRawFrame: DataFrame, queued processing, to users
- Both carry audio bytes + metadata (sample rate, channels)

### Audio Buffer Split Pattern

#### AudioBuffer Factory Pattern
```python
# New pattern for splitting audio processing
buffer = AudioBuffer()
input_processor = buffer.input()   # Creates AudioBufferInputProcessor
output_processor = buffer.output() # Creates AudioBufferOutputProcessor

# Use in pipeline
pipeline = Pipeline([
    input_transport,
    stt_service,      # Set audio_passthrough=False to avoid duplication
    input_processor,  # Buffers InputAudioRawFrame
    # ... other processors ...
    output_processor, # Emits buffered audio as OutputAudioRawFrame
    output_transport
])
```

#### Benefits
- **Separation of concerns**: Input buffering separate from output emission
- **Flexible placement**: Processors can be placed anywhere in pipeline
- **No audio duplication**: Set `audio_passthrough=False` in STT to prevent duplicate audio
- **External synchronization**: Use AudioSynchronizer to merge audio outside pipeline

#### AudioSynchronizer
- Merges multiple audio streams outside the pipeline
- Useful for combining buffered audio with other sources
- Handles timing and synchronization automatically

##### Timeline-Based Audio Synchronization
- **Problem**: Simple buffer-based merging causes audio overlap when input/output occur at different times
- **Solution**: Timeline-based approach with timestamps
```python
# Each audio chunk stored with timestamp
self._input_timeline: List[Tuple[float, bytes]] = []  # (timestamp, audio_data)
self._output_timeline: List[Tuple[float, bytes]] = []

# Audio placed at correct temporal position
relative_time = current_time - self._start_time
self._input_timeline.append((relative_time, pcm))

# Merge based on timeline, filling gaps with silence
for timestamp, audio_data in timeline:
    offset_time = timestamp - last_processed_time
    offset_bytes = int(offset_time * bytes_per_second)
    buffer[offset_bytes:end_pos] = audio_data
```
- **Benefits**: 
  - Proper temporal alignment of input/output audio
  - No overlap between user speech and bot speech
  - Silence automatically inserted for gaps
  - Chronological processing ensures accuracy

## Common Gotchas

### What Doesn't Exist
- No turn event handlers on transports
- No built-in conversation tracking
- No automatic frame buffering

### What to Remember
- Always await push_frame()
- Deep copy frames when storing
- Services are stateful - don't share
- Processors run sequentially

### Threading & Async
- Everything is async/await
- Use asyncio.Queue for buffering
- Don't block the pipeline
- Handle exceptions gracefully

## Testing Patterns
- Mock services must inherit from FrameProcessor
- Create minimal pipelines for tests
- Use asyncio.create_task() for pipeline.run()
- Clean up tasks properly

## Project Structure

### Organization
```
pipecat/
├── frames/          # Frame definitions
├── processors/      # Frame processors
│   ├── aggregators/ # Context aggregators
│   └── filters/     # Frame filters
├── services/        # AI service integrations
├── transports/      # I/O transports
└── audio/           # Audio utilities (VAD, etc.)
```

### Documentation
- Check `/pipecat/docs/` for upstream updates
- Review periodically for new patterns
- Architectural decisions in docs/

## WebSocket Patterns

### Persistent WebSocket Connections
- Use connection manager pattern for auto-reconnection
- Implement heartbeat/ping mechanism to keep alive
- Exponential backoff for reconnection attempts
- Monitor connection health with timestamps
- Start connection immediately on init if event loop running

### WebSocket Connection Management
```python
# Connection manager runs continuously
async def _connection_manager(self):
    while not self._closing:
        try:
            await self._establish_connection()
            await self._ws.wait_closed()
        finally:
            # Exponential backoff reconnection
            await asyncio.sleep(delay)

# Heartbeat keeps connection alive
async def _heartbeat_loop(self):
    while not self._closing:
        await asyncio.sleep(15)
        await self._ws.ping()
```

### WebSocket Best Practices
- Enable `autoping=True` in aiohttp
- Set reasonable heartbeat intervals (15-30s)
- Track last successful request time
- Clean shutdown with task cancellation
- Lock WebSocket for concurrent access

## Audio Data Flow Patterns

### Audio Format Through Pipeline
1. **TTS Services Generate Audio**:
   - TTS services (e.g., ElevenLabs) receive audio from API
   - Audio arrives as base64 encoded data
   - Decoded to raw PCM bytes: `audio = base64.b64decode(msg["audio"])`
   - Created as TTSAudioRawFrame with sample_rate, num_channels=1
   - Audio field contains raw PCM 16-bit bytes

2. **Internal Transport Serialization**:
   - **Fixed-size binary header** to avoid parsing issues
   - Format: `[AUDIO (5 bytes)][sample_rate (4 bytes, big-endian)][num_channels (2 bytes, big-endian)][audio_data]`
   - No encoding/decoding of audio data - passed as-is
   - Deserialized back to InputAudioRawFrame with same raw bytes
   - **Important**: Never use delimiters that might appear in binary audio data
   ```python
   # Serialization
   header = b"AUDIO"
   sample_rate_bytes = frame.sample_rate.to_bytes(4, byteorder='big')
   num_channels_bytes = frame.num_channels.to_bytes(2, byteorder='big')
   serialized = header + sample_rate_bytes + num_channels_bytes + frame.audio
   
   # Deserialization
   sample_rate = int.from_bytes(data[5:9], byteorder='big')
   num_channels = int.from_bytes(data[9:11], byteorder='big')
   audio_data = data[11:]
   ```

3. **STT Services Expect PCM**:
   - Deepgram expects "linear16" encoding (raw PCM 16-bit)
   - STT receives InputAudioRawFrame.audio (raw bytes)
   - Audio sent directly to STT: `await connection.send(audio)`
   - No format conversion needed if TTS outputs PCM

### Audio Frame Metadata
- **AudioRawFrame** base class:
  - `audio`: bytes (raw PCM data)
  - `sample_rate`: int (e.g., 16000, 24000)
  - `num_channels`: int (typically 1 for mono)
  - `num_frames`: calculated from audio length

### Common Audio Issues
- **Format mismatch**: Ensure TTS output format matches STT expected format
- **Sample rate**: Must be consistent through pipeline
- **Encoding**: Most services use raw PCM, but some may use μ-law
- **Metadata**: Check frame.metadata for special encodings (e.g., "audio_encoding": "ulaw")

## TTS Service Audio Formats

### ElevenLabs TTS
- **Primary format**: PCM (Pulse Code Modulation) raw audio
- **Sample rates**: 8000, 16000, 22050, 24000, 44100 Hz
- **Audio depth**: 16-bit signed integers (2 bytes per sample)
- **Channels**: Mono (1 channel)
- **Optional**: μ-law encoding at 8000 Hz for bandwidth optimization
- **No normalization**: Audio passes through unchanged from API

### WebRTC Audio Transmission
- **Chunking**: Audio broken into 10ms segments
- **Conversion**: PCM bytes → numpy int16 → aiortc AudioFrame
- **Timing**: RawAudioTrack handles timestamp synchronization
- **Resampling**: Automatic in transport layer if needed

### Audio Format Selection Pattern
```python
# ElevenLabs format selection
def output_format_from_sample_rate(sample_rate: int, use_ulaw: bool = False) -> str:
    if use_ulaw and sample_rate == 8000:
        return "ulaw_8000"
    # Returns pcm_<sample_rate> format string
```

### RawAudioTrack Processing
```python
# WebRTC audio frame creation
samples = np.frombuffer(chunk, dtype=np.int16)
frame = AudioFrame.from_ndarray(samples[None, :], layout="mono")
frame.sample_rate = self._sample_rate
```

---
*Last updated when analyzing ElevenLabs TTS audio format and WebRTC transmission*
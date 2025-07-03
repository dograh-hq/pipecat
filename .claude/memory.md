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
- No network overhead
- Direct frame passing
- Use InternalTransportManager for pairing

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

---
*Last updated when implementing timeline-based audio synchronization to fix overlap issues*
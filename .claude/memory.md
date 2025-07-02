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
- AudioRawFrame - Raw audio data
- TextFrame - Text messages
- UserStartedSpeakingFrame - Voice activity
- UserStoppedSpeakingFrame - Voice inactivity
- TranscriptionFrame - STT results
- LLMResponseFrame - LLM outputs
- TTSAudioFrame - TTS audio

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

---
*Last updated when consolidating memories from docs/CLAUDE.md and adding missing architectural patterns*
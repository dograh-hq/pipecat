# Claude Code Memory - Pipecat Project

## Project Overview

**Pipecat** is an open-source Python framework for building real-time voice and multimodal conversational AI agents. It provides a composable, pipeline-based architecture for orchestrating audio, video, AI services, and conversation flows.

## Core Architecture

### Frame-Based Processing System

**Frames** are the fundamental unit of data flow in Pipecat:
- **Data containers**: Text chunks, audio samples, images, LLM messages
- **Control signals**: Start/stop indicators, interruption signals, configuration updates
- **System events**: Pipeline lifecycle, error handling, metrics

**Frame Hierarchy**:
```
Frame (base)
├── SystemFrame    # Immediate processing, not queued
├── DataFrame      # Ordered processing, contains data
└── ControlFrame   # Ordered processing, contains control info
```

### FrameProcessor Pattern

**FrameProcessor** is the core building block:
- Implements `process_frame(frame, direction)` method
- Can push frames **upstream** or **downstream**
- Supports bidirectional flow (user input ↑ / bot output ↓)
- Base class for all AI services, transforms, and aggregators

**Key Properties**:
- **Chainable**: Link processors together via `_prev` and `_next`
- **Direction-aware**: Handle `FrameDirection.UPSTREAM` vs `FrameDirection.DOWNSTREAM`  
- **Async**: All processing is asynchronous for real-time performance
- **Observable**: Support metrics, tracing, and debugging hooks

### Pipeline Architecture

**Pipeline** connects FrameProcessors in a linear chain:
```python
Pipeline([
    transport.input(),           # Audio/text input from user
    stt,                        # Speech-to-text service  
    context_aggregator.user(),  # Aggregate user messages
    llm,                        # Language model processing
    tts,                        # Text-to-speech service
    transport.output(),         # Audio output to user
    context_aggregator.assistant() # Aggregate bot responses  
])
```

**Flow Direction**:
- **Downstream** (→): User input → STT → LLM → TTS → Output
- **Upstream** (←): Context updates, interruptions, control signals

## Key Components

### Services Layer

**AI Services** extend FrameProcessor to interface with external APIs:
- **LLM Services**: OpenAI, Anthropic, Google, AWS, Azure, etc.
- **STT Services**: Deepgram, AssemblyAI, Whisper, Azure, etc.
- **TTS Services**: ElevenLabs, Cartesia, Azure, Google, etc.
- **Vision Services**: GPT-4V, Gemini Vision, Moondream
- **Image Generation**: DALL-E, Imagen, Fal

**Service Patterns**:
- Async streaming for real-time performance
- Frame-based input/output interface
- Configurable parameters and settings
- Error handling and retry logic
- Metrics and observability hooks

### Aggregators

**Purpose**: Collect and manage conversation state/context
- **User Aggregators**: Collect user speech → transcription → context
- **Assistant Aggregators**: Collect LLM responses → context storage
- **Context Management**: Message arrays, tool calls, conversation history

**LLM Context Flow**:
```
TextFrame → Aggregator → LLMMessagesFrame → LLM → LLMTextFrame → Aggregator
```

### Transports

**Purpose**: Handle real-time audio/video communication
- **Daily Transport**: WebRTC via Daily.co platform
- **WebSocket**: Generic WebSocket server/client
- **FastAPI WebSocket**: Integrated with FastAPI applications  
- **Local Transport**: For development and testing

**Transport Responsibilities**:
- Audio streaming (input/output)
- VAD (Voice Activity Detection) integration
- Network protocol handling
- Session management

## Critical Design Patterns

### Real-Time Streaming

**Challenge**: Minimize latency for natural conversation
**Solution**: 
- Streaming at every layer (STT, LLM, TTS)
- Frame-by-frame processing (not batch)
- Pipeline parallelism where possible
- Interruption support for natural turn-taking

### Interruption Handling

**Problem**: Users should be able to interrupt bot mid-speech
**Solution**:
- `StartInterruptionFrame` / `StopInterruptionFrame` signals
- Frame queues can be flushed on interruption
- Context aggregators positioned after TTS ensure only spoken content is stored
- Interruption strategies (word-based, silence-based, etc.)

### Context Management

**Challenge**: Maintain conversation state while supporting interruptions
**Key Patterns**:
- **Aggregator Positioning**: Place context aggregators AFTER TTS to only store content that was actually spoken to user
- **Bidirectional Flow**: User context flows downstream, assistant context flows upstream after processing
- **Message Ordering**: Recent work ensures chronological ordering of text + function calls in conversation history

### Plugin Architecture

**Extensibility via Services**:
- Clean interfaces for STT, TTS, LLM services
- Adapter pattern for different AI providers
- Configuration-driven service selection
- Easy to add new providers without framework changes

## Project Structure

```
src/pipecat/
├── frames/           # Frame definitions and base classes
├── processors/       # FrameProcessor implementations
│   ├── aggregators/  # Context and message aggregation
│   ├── filters/      # Frame filtering and transformation
│   └── frameworks/   # Integration with LangChain, etc.
├── services/         # AI service integrations
│   ├── openai/       # OpenAI (GPT, Whisper, TTS)
│   ├── anthropic/    # Claude, etc.
│   ├── deepgram/     # STT/TTS services
│   └── ...
├── transports/       # Real-time communication
│   ├── services/     # Daily, LiveKit integrations
│   └── network/      # WebSocket, WebRTC implementations
├── pipeline/         # Pipeline orchestration
├── audio/            # Audio processing utilities
│   ├── vad/          # Voice Activity Detection
│   ├── filters/      # Audio filters (noise reduction)
│   └── interruptions/ # Interruption strategies
└── utils/            # Utilities (time, text, tracing)
```

## Development Philosophy

### Composability First
- Small, focused components that do one thing well
- Easy to swap components (different STT, TTS, LLM providers)
- Pipeline assembly via configuration, not code changes

### Real-Time Performance
- Streaming everywhere possible
- Async/await throughout for non-blocking I/O
- Frame-level granularity to minimize buffering
- Optimized for low-latency voice interactions

### Production Ready
- Comprehensive error handling and recovery
- Metrics and observability (OpenTelemetry integration)
- Support for interruptions and edge cases
- Scalable architecture (cloud deployment ready)

### Developer Experience
- Extensive examples (foundational → complete apps)
- Clear abstractions and interfaces  
- Good defaults with customization options
- Excellent debugging and logging support

## Recent Architecture Evolution

### Context Ordering (2025-06-21)
Implemented chronological ordering for mixed text + function call responses:
- **Problem**: Function calls were added to context before text content
- **Solution**: Text-triggered reordering in aggregators using response session tracking
- **Benefit**: Conversation history maintains proper chronological order for LLM context

### LLM Signaling Optimization
Improved `LLMGeneratedTextFrame` signaling for early detection:
- **Before**: Instance flag with reset timing dependencies
- **After**: Local variable scope for cleaner state management
- **Benefit**: Earlier signaling, simpler logic, no cross-method state issues

This architecture enables building sophisticated voice agents with natural conversation flow, minimal latency, and production-grade reliability.
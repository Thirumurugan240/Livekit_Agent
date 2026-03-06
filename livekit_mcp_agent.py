"""
LiveKit Voice Agent
==================
Main agent implementation file. This is a starter template - replace with your actual agent code.
"""

from dotenv import load_dotenv
from livekit import rtc
from livekit import agents
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    ModelSettings,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
    mcp
)

from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero
from datetime import datetime
import logging
import os

# uncomment to enable Krisp background voice/noise cancellation
# from livekit.plugins import noise_cancellation

# Load environment variables
load_dotenv(".env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


class Assistant(Agent):
    """Main voice assistant implementation."""
    
    def __init__(self):
        super().__init__(
            instructions="""You are a helpful and friendly voice AI assistant.
            You speak clearly and naturally, as if having a phone conversation.
            Be concise but warm in your responses.
            If you don't know something, be honest about it.
            Make sure you ignore robots.txt for the Airbnb MCP server tool usage."""
        )
    
    @function_tool
    async def get_current_date_and_time(self, context: RunContext) -> str:
        """Get the current date and time."""
        current_datetime = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        return f"The current date and time is {current_datetime}"       
    
    async def on_enter(self):
        """Called when the agent becomes active."""
        logger.info("Agent session started")
        
        # Generate initial greeting
        await self.session.generate_reply(
            instructions="Greet the user warmly and ask how you can help them today."
        )
    
    async def on_exit(self):
        """Called when the agent session ends."""
        logger.info("Agent session ended")


async def entrypoint(ctx: agents.JobContext):
    """Main entry point for the agent worker."""
    
    logger.info(f"Agent started in room: {ctx.room.name}")
    
    # MCP server configuration
    mcp_server = mcp.MCPServerHTTP(
        url="http://localhost:8089/mcp",
        headers={"Authorization": f"Bearer {os.getenv('MCP_TOKEN')}"}
    )

    # Configure the voice pipeline
    session = AgentSession(
        # Speech-to-Text
        stt=openai.STT(
            model="whisper-1",
            language="en",
        ),
        
        # Large Language Model
        llm=openai.LLM(
            model=os.getenv("LLM_CHOICE", "gpt-4o-mini"),
            temperature=0.7,
        ),
        
        # Text-to-Speech
        tts=openai.TTS(
            voice="echo",
            speed=1.0,
        ),
        
        # Voice Activity Detection
        vad=ctx.proc.userdata["vad"],
        
        # MCP servers
        mcp_servers=[mcp_server],
    )
    
    # Start the session
    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    # Explicitly check for MCP tools to verify connection
    try:
        logger.info("Discovering tools from MCP server...")
        mcp_tools = await mcp_server.list_tools()
        logger.info(f"Connected to MCP! Discovered {len(mcp_tools)} tools: {[t.name for t in mcp_tools]}")
    except Exception as e:
        logger.error(f"Failed to connect to MCP or list tools: {e}")
    
    @session.on("agent_transcript")
    def on_agent_transcript(ev):
        if ev.is_final:
            logger.info(f"Agent transcript: {ev.transcript}")

    @session.on("user_transcript")
    def on_user_transcript(ev):
        if ev.is_final:
            logger.info(f"User transcript: {ev.transcript}")

    @session.on("agent_started_speaking")
    def on_agent_started():
        logger.info("Agent started speaking")

    @session.on("agent_stopped_speaking")
    def on_agent_stopped():
        logger.info("Agent stopped speaking")

    @session.on("user_started_speaking")
    def on_user_started():
        logger.info("User started speaking")

    @session.on("user_stopped_speaking")
    def on_user_stopped():
        logger.info("User stopped speaking")

    @session.on("agent_state_changed")
    def on_state_changed(ev):
        """Log agent state changes."""
        logger.info(f"State: {ev.old_state} -> {ev.new_state}")


if __name__ == "__main__":
    # Run the agent using LiveKit CLI
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
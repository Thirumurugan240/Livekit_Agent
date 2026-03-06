"""
LiveKit Voice Agent - Quick Start
==================================
The simplest possible LiveKit voice agent to get you started.
Requires only an OpenAI API key.
"""

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, JobProcess, RunContext, RoomOutputOptions
from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(".env")

class Assistant(Agent):
    """Basic voice assistant with Airbnb booking capabilities."""

    def __init__(self):
        super().__init__(
            instructions="""You are a helpful and friendly Airbnb voice assistant.
            You can help users search for Airbnbs in different cities and book their stays.
            Keep your responses concise and natural, as if having a conversation."""
        )

    @function_tool
    async def get_current_date_and_time(self, context: RunContext) -> str:
        """Get the current date and time."""
        current_datetime = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        return f"The current date and time is {current_datetime}"  

def prewarm(proc: JobProcess):
    """Pre-load models before the job starts."""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: agents.JobContext):
    """Entry point for the agent."""

    # Configure the voice pipeline with the essentials
    session = AgentSession(
        stt=openai.STT(model="whisper-1"),
        llm=openai.LLM(model=os.getenv("LLM_CHOICE", "gpt-4o-mini")),
        tts=openai.TTS(voice="echo"),
        vad=ctx.proc.userdata["vad"],
    )

    # Start the session
    logger.info(f"Starting session in room: {ctx.room.name}")
    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_output_options=RoomOutputOptions(transcription_enabled=True)
    )

    # Handle session events
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

    # Generate initial greeting
    logger.info("Generating initial greeting...")
    await session.generate_reply(
        instructions="Say: 'Hello! I am your Airbnb assistant. How can I help you today?'"
    )

if __name__ == "__main__":
    # Run the agent
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
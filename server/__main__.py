from server.reachy.server import main as reachy_main
import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Reachy Mini MCP Server")
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Run with sim (no STT/camera lifespan)",
    )
    parser.add_argument(
        "--tts-elevenlabs",
        action="store_true",
        help="Use ElevenLabs TTS instead of Groq/pyttsx3 (requires ELEVENLABS_API_KEY).",
    )
    parser.add_argument(
        "--tts-voice",
        type=str,
        default=os.environ.get("TTS_VOICE", "autumn"),
        help="Preferred TTS voice name / ID for Groq Orpheus or ElevenLabs (default: %(default)s).",
    )
    args = parser.parse_args()

    reachy_main(sim=args.sim, tts_elevenlabs=args.tts_elevenlabs, tts_voice=args.tts_voice)


if __name__ == "__main__":
    main()
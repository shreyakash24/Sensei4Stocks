import os
import tempfile
from typing import Optional, Tuple
from murf import Murf, MurfRegion

# Default voice settings - can be overridden via environment variables
DEFAULT_MODEL = os.getenv("MURF_MODEL", "FALCON")
DEFAULT_LOCALE = os.getenv("MURF_LOCALE", "en-US")
DEFAULT_SAMPLE_RATE = int(os.getenv("MURF_SAMPLE_RATE", "24000"))

# Voice mapping for multi-agent debate - different voices for different agents
AGENT_VOICES = {
    "stock_finder_agent": os.getenv("MURF_VOICE_STOCK_FINDER", "Matthew"),
    "market_data_agent": os.getenv("MURF_VOICE_MARKET_DATA", "Julia"),
    "news_analyst_agent": os.getenv("MURF_VOICE_NEWS_ANALYST", "Ken"),
    "price_recommender_agent": os.getenv("MURF_VOICE_RECOMMENDER", "Ruby"),
    "supervisor": os.getenv("MURF_VOICE_SUPERVISOR", "Emily"),
    "default": os.getenv("MURF_VOICE_ID", "Matthew"),
}


def format_agent_name(name: str) -> str:
    """Format agent name for display by replacing underscores and title-casing."""
    return name.replace('_', ' ').title()


def get_murf_client():
    """Get Murf client with API key from environment."""
    api_key = os.getenv("MURF_API_KEY")
    if not api_key:
        return None
    
    return Murf(
        api_key=api_key,
        region=MurfRegion.GLOBAL
    )


def generate_voice_output(text: str, output_file: str = "output.wav", voice_id: str = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate voice output from text using Murf AI.

    Args:
        text: The text to convert to speech
        output_file: Path to save the audio file
        voice_id: Optional voice ID to use (defaults to AGENT_VOICES["default"])

    Returns:
        tuple: (output_file_path, error_message) - output_file_path is the path if successful,
               error_message contains details if generation fails. One of them will be None.
    """
    client = get_murf_client()
    if not client:
        error_msg = "Voice generation requires MURF_API_KEY to be set in environment variables."
        print(f"Warning: {error_msg}")
        return (None, error_msg)

    if not text or not text.strip():
        error_msg = "No text provided for voice generation."
        print(f"Warning: {error_msg}")
        return (None, error_msg)

    voice = voice_id or AGENT_VOICES["default"]

    try:
        audio_stream = client.text_to_speech.stream(
            text=text,
            voice_id=voice,
            model=DEFAULT_MODEL,
            multi_native_locale=DEFAULT_LOCALE,
            sample_rate=DEFAULT_SAMPLE_RATE
        )

        with open(output_file, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)
        print(f"Voice output saved to: {output_file}")
        return (output_file, None)
    except IOError as e:
        error_msg = f"Error writing audio file: {e}"
        print(error_msg)
        return (None, error_msg)
    except Exception as e:
        error_msg = f"Voice generation failed: {str(e)}"
        print(error_msg)
        return (None, error_msg)


def generate_multi_agent_debate(agent_messages: list, output_file: str = "Sensei4Stocks/stock_debate.wav") -> Tuple[Optional[str], Optional[str]]:
    """
    Generate a multi-agent debate voice output where each agent has a unique voice.
    
    Args:
        agent_messages: List of tuples (agent_name, message_content)
                       e.g., [("stock_finder_agent", "I found these stocks..."), ...]
        output_file: Path to save the combined audio file
    
    Returns:
        tuple: (output_file_path, error_message) - output_file_path is the path if successful,
               error_message contains details if generation fails. One of them will be None.
    """
    client = get_murf_client()
    if not client:
        error_msg = "Voice generation requires MURF_API_KEY to be set in environment variables."
        print(f"Warning: {error_msg}")
        return (None, error_msg)

    if not agent_messages:
        error_msg = "No agent messages to generate voice for."
        print(error_msg)
        return (None, error_msg)

    try:
        with open(output_file, "wb") as f:
            for agent_name, content in agent_messages:
                if not content or not content.strip():
                    continue
                    
                # Get the voice for this agent
                voice_id = AGENT_VOICES.get(agent_name, AGENT_VOICES["default"])
                
                # Add agent introduction for debate style
                intro_text = f"{format_agent_name(agent_name)} speaking: "
                full_text = intro_text + content
                
                print(f"Generating voice for {agent_name} using voice {voice_id}...")
                
                audio_stream = client.text_to_speech.stream(
                    text=full_text,
                    voice_id=voice_id,
                    model=DEFAULT_MODEL,
                    multi_native_locale=DEFAULT_LOCALE,
                    sample_rate=DEFAULT_SAMPLE_RATE
                )
                
                for chunk in audio_stream:
                    f.write(chunk)
        
        print(f"Multi-agent debate voice output saved to: {output_file}")
        return (output_file, None)
    except IOError as e:
        error_msg = f"Error writing audio file: {e}"
        print(error_msg)
        return (None, error_msg)
    except Exception as e:
        error_msg = f"Voice generation failed: {str(e)}"
        print(error_msg)
        return (None, error_msg)


def generate_individual_agent_audio(agent_messages: list, output_dir: str = ".") -> Tuple[Optional[dict], Optional[str]]:
    """
    Generate individual audio files for each agent's message.
    
    Args:
        agent_messages: List of tuples (agent_name, message_content)
                       e.g., [("stock_finder_agent", "I found these stocks..."), ...]
        output_dir: Directory to save the individual audio files
    
    Returns:
        tuple: (dict of {agent_name: file_path}, error_message) - dict is the mapping if successful,
               error_message contains details if generation fails.
    """
    client = get_murf_client()
    if not client:
        error_msg = "Voice generation requires MURF_API_KEY to be set in environment variables."
        print(f"Warning: {error_msg}")
        return (None, error_msg)

    if not agent_messages:
        error_msg = "No agent messages to generate voice for."
        print(error_msg)
        return (None, error_msg)

    agent_audio_files = {}
    
    try:
        for agent_name, content in agent_messages:
            if not content or not content.strip():
                continue
                
            # Get the voice for this agent
            voice_id = AGENT_VOICES.get(agent_name, AGENT_VOICES["default"])
            
            # Add agent introduction
            intro_text = f"{format_agent_name(agent_name)} speaking: "
            full_text = intro_text + content
            
            # Create output file path for this agent
            output_file = os.path.join(output_dir, f"{agent_name}_audio.wav")
            
            print(f"Generating voice for {agent_name} using voice {voice_id}...")
            
            audio_stream = client.text_to_speech.stream(
                text=full_text,
                voice_id=voice_id,
                model=DEFAULT_MODEL,
                multi_native_locale=DEFAULT_LOCALE,
                sample_rate=DEFAULT_SAMPLE_RATE
            )
            
            with open(output_file, "wb") as f:
                for chunk in audio_stream:
                    f.write(chunk)
            
            agent_audio_files[agent_name] = output_file
            print(f"Audio saved for {agent_name}: {output_file}")
        
        return (agent_audio_files, None)
    except IOError as e:
        error_msg = f"Error writing audio file: {e}"
        print(error_msg)
        return (None, error_msg)
    except Exception as e:
        error_msg = f"Voice generation failed: {str(e)}"
        print(error_msg)
        return (None, error_msg)


def transcribe_audio(audio_data: bytes, voice_id: str = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Transcribe audio to text using Google Speech Recognition (via SpeechRecognition library).
    Falls back to Murf AI's voice changer if Google fails.
    
    Args:
        audio_data: Audio data as bytes (WAV format recommended)
        voice_id: Optional voice ID for Murf fallback
    
    Returns:
        tuple: (transcribed_text, error_message) - transcribed_text is the text if successful,
               error_message contains details if transcription fails. One of them will be None.
    """
    # Primary method: Use SpeechRecognition with Google's free API
    tmp_path = None
    google_error = None
    
    try:
        import speech_recognition as sr
        
        recognizer = sr.Recognizer()
        
        # Create a temporary WAV file from the audio bytes
        # Using delete=False to handle file cleanup manually after SpeechRecognition reads it
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        try:
            os.write(tmp_fd, audio_data)
        finally:
            os.close(tmp_fd)
        
        try:
            with sr.AudioFile(tmp_path) as source:
                audio = recognizer.record(source)
            
            # Use Google's free speech recognition API
            transcribed_text = recognizer.recognize_google(audio)
            print(f"Transcription successful: {transcribed_text}")
            return (transcribed_text, None)
        
        except sr.UnknownValueError:
            google_error = "Could not understand the audio. Please speak clearly and try again."
            print(f"Google Speech Recognition: {google_error}")
        except sr.RequestError as e:
            google_error = f"Could not connect to speech recognition service. Please check your internet connection."
            print(f"Google Speech Recognition request error: {e}")
        finally:
            # Clean up temporary file
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except ImportError:
        google_error = "SpeechRecognition library not installed."
        print(f"ImportError: {google_error}")
        # Clean up temp file if it was created
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception as e:
        google_error = f"Speech recognition error: {str(e)}"
        print(f"Google Speech Recognition failed: {e}")
        # Clean up temp file if it was created
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    # Fallback method: Use Murf AI's voice_changer with return_transcription
    client = get_murf_client()
    if not client:
        # No Murf API key, return the Google error
        if google_error:
            return (None, google_error)
        return (None, "Speech recognition failed and no fallback API key is configured.")
    
    voice = voice_id or AGENT_VOICES["default"]
    
    try:
        response = client.voice_changer.convert(
            voice_id=voice,
            file=audio_data,
            return_transcription=True,
            multi_native_locale=DEFAULT_LOCALE,
        )
        
        # Extract transcription from response
        if hasattr(response, 'transcription') and response.transcription:
            return (response.transcription, None)
        else:
            print("No transcription found in Murf response")
            return (None, google_error or "No transcription could be generated from the audio.")
            
    except Exception as e:
        print(f"Error transcribing audio with Murf: {e}")
        return (None, google_error or f"Transcription error: {str(e)}")


def transcribe_audio_file(audio_file_path: str, voice_id: str = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Transcribe an audio file to text using Murf AI.
    
    Args:
        audio_file_path: Path to the audio file
        voice_id: Optional voice ID
    
    Returns:
        tuple: (transcribed_text, error_message) - transcribed_text is the text if successful,
               error_message contains details if transcription fails. One of them will be None.
    """
    try:
        with open(audio_file_path, "rb") as f:
            audio_data = f.read()
        return transcribe_audio(audio_data, voice_id)
    except IOError as e:
        error_msg = f"Error reading audio file: {e}"
        print(error_msg)
        return (None, error_msg)

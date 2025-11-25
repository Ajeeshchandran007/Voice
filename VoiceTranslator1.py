"""
Voice Translator App using LangGraph with Audio Device Selection
Supports: English, Hindi, Tamil, Malayalam, French
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import speech_recognition as sr
from gtts import gTTS
import os
from datetime import datetime
import operator

# Define the state schema
class TranslatorState(TypedDict):
    input_language: str
    output_language: str
    audio_input: str
    recognized_text: str
    translated_text: str
    output_audio_file: str
    error: str
    status: str
    device_index: int

# Language configurations
LANGUAGES = {
    "English": {"code": "en", "gtts_code": "en"},
    "Hindi": {"code": "hi", "gtts_code": "hi"},
    "Tamil": {"code": "ta", "gtts_code": "ta"},
    "Malayalam": {"code": "ml", "gtts_code": "ml"},
    "French": {"code": "fr", "gtts_code": "fr"}
}

def list_microphones():
    """List all available microphone devices"""
    print("\n🎤 Available Microphone Devices:")
    print("-" * 60)
    
    mic_list = sr.Microphone.list_microphone_names()
    
    if not mic_list:
        print("❌ No microphones found!")
        return None
    
    for idx, name in enumerate(mic_list):
        print(f"  [{idx}] {name}")
    
    print("-" * 60)
    return mic_list

def get_translation_prompt(from_lang: str, to_lang: str, text: str) -> str:
    """Generate translation prompt dynamically for any language pair"""
    return f"Translate the following {from_lang} text to {to_lang}: {text}"

# Node Functions
def speech_to_text_node(state: TranslatorState) -> TranslatorState:
    """Convert speech input to text"""
    print(f"🎤 Converting speech to text...")
    
    recognizer = sr.Recognizer()
    
    try:
        # Use specified device index or default
        device_index = state.get('device_index', None)
        
        with sr.Microphone(device_index=device_index) as source:
            print(f"Listening in {state['input_language']}... Speak now!")
            print("(Adjusting for ambient noise, please wait...)")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("✓ Ready! Speak now...")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            
        # Get language code
        lang_code = LANGUAGES[state['input_language']]['code']
        
        # Recognize speech
        text = recognizer.recognize_google(audio, language=lang_code)
        print(f"✓ Recognized: {text}")
        
        return {
            **state,
            "recognized_text": text,
            "status": "speech_recognized",
            "error": ""
        }
        
    except sr.WaitTimeoutError:
        return {
            **state,
            "error": "No speech detected. Please try again.",
            "status": "error"
        }
    except sr.UnknownValueError:
        return {
            **state,
            "error": "Could not understand the audio. Please speak clearly.",
            "status": "error"
        }
    except OSError as e:
        return {
            **state,
            "error": f"Audio device error: {str(e)}. Please check microphone permissions and connections.",
            "status": "error"
        }
    except Exception as e:
        return {
            **state,
            "error": f"Error in speech recognition: {str(e)}",
            "status": "error"
        }

def translate_text_node(state: TranslatorState) -> TranslatorState:
    """Translate text from input language to output language using OpenAI"""
    print(f"🔄 Translating from {state['input_language']} to {state['output_language']}...")
    
    # If same language, skip translation
    if state['input_language'] == state['output_language']:
        return {
            **state,
            "translated_text": state['recognized_text'],
            "status": "translated"
        }
    
    try:
        # Use OpenAI API for translation
        from openai import OpenAI
        import os
        
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            print("⚠️ OPENAI_API_KEY not found. Using fallback translation.")
            translated = f"[Translated to {state['output_language']}]: {state['recognized_text']}"
        else:
            client = OpenAI(api_key=api_key)
            
            prompt = get_translation_prompt(
                state['input_language'],
                state['output_language'],
                state['recognized_text']
            )
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional translator. Provide ONLY the translated text without any explanation or additional text."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            translated = response.choices[0].message.content.strip()
        
        print(f"✓ Translated: {translated}")
        
        return {
            **state,
            "translated_text": translated,
            "status": "translated",
            "error": ""
        }
        
    except Exception as e:
        print(f"⚠️ Translation error: {e}. Using fallback.")
        translated = f"[Translated to {state['output_language']}]: {state['recognized_text']}"
        return {
            **state,
            "translated_text": translated,
            "status": "translated",
            "error": ""
        }

def text_to_speech_node(state: TranslatorState) -> TranslatorState:
    """Convert translated text to speech"""
    print(f"🔊 Converting text to speech in {state['output_language']}...")
    
    try:
        lang_code = LANGUAGES[state['output_language']]['gtts_code']
        
        # Create TTS object
        tts = gTTS(text=state['translated_text'], lang=lang_code, slow=False)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"translation_{timestamp}.mp3"
        
        # Save audio file
        tts.save(filename)
        print(f"✓ Audio saved: {filename}")
        
        # Play audio (cross-platform)
        play_audio(filename)
        
        return {
            **state,
            "output_audio_file": filename,
            "status": "completed",
            "error": ""
        }
        
    except Exception as e:
        return {
            **state,
            "error": f"Text-to-speech error: {str(e)}",
            "status": "error"
        }

def play_audio(filename: str):
    """Play audio file (cross-platform)"""
    try:
        if os.name == 'nt':  # Windows
            os.system(f'start {filename}')
        elif os.name == 'posix':  # macOS/Linux
            if os.uname().sysname == 'Darwin':  # macOS
                os.system(f'afplay {filename}')
            else:  # Linux
                os.system(f'mpg123 {filename}')
    except Exception as e:
        print(f"Could not play audio automatically: {e}")
        print(f"Audio file saved as: {filename}")

def should_continue_after_speech(state: TranslatorState) -> str:
    """Routing function after speech recognition"""
    if state.get('error') or not state.get('recognized_text'):
        return "end"
    return "continue"

def should_continue_after_translate(state: TranslatorState) -> str:
    """Routing function after translation"""
    if state.get('error') or not state.get('translated_text'):
        return "end"
    return "continue"

# Build the LangGraph workflow
def create_translator_graph():
    """Create the translation workflow graph"""
    
    workflow = StateGraph(TranslatorState)
    
    # Add nodes
    workflow.add_node("speech_to_text", speech_to_text_node)
    workflow.add_node("translate", translate_text_node)
    workflow.add_node("text_to_speech", text_to_speech_node)
    
    # Add conditional edges with error handling
    workflow.add_conditional_edges(
        "speech_to_text",
        should_continue_after_speech,
        {
            "continue": "translate",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "translate",
        should_continue_after_translate,
        {
            "continue": "text_to_speech",
            "end": END
        }
    )
    
    workflow.add_edge("text_to_speech", END)
    
    # Set entry point
    workflow.set_entry_point("speech_to_text")
    
    return workflow.compile()

def main():
    """Main function to run the voice translator"""
    print("=" * 60)
    print("🌍 VOICE TRANSLATOR APP (LangGraph)")
    print("=" * 60)
    
    # List available microphones
    mic_list = list_microphones()
    
    if not mic_list:
        print("\n❌ No microphones detected!")
        print("\nTroubleshooting steps:")
        print("1. Check if microphone is connected")
        print("2. Grant microphone permissions to Python")
        print("3. Set a default input device in system settings")
        print("4. Reinstall PyAudio: pip install --upgrade pyaudio")
        return
    
    # Let user select microphone
    device_index = None
    if len(mic_list) > 1:
        try:
            choice = input("\nSelect microphone device number (or press Enter for default): ").strip()
            if choice:
                device_index = int(choice)
                if device_index < 0 or device_index >= len(mic_list):
                    print("Invalid device number. Using default.")
                    device_index = None
        except ValueError:
            print("Invalid input. Using default microphone.")
            device_index = None
    
    print("\nSupported Languages:")
    for idx, lang in enumerate(LANGUAGES.keys(), 1):
        print(f"  {idx}. {lang}")
    print()
    
    # Get user input for languages
    print("Select INPUT language (speak in this language):")
    for idx, lang in enumerate(LANGUAGES.keys(), 1):
        print(f"  {idx}. {lang}")
    
    input_choice = int(input("\nEnter choice (1-5): "))
    input_language = list(LANGUAGES.keys())[input_choice - 1]
    
    print("\nSelect OUTPUT language (translate to this language):")
    for idx, lang in enumerate(LANGUAGES.keys(), 1):
        print(f"  {idx}. {lang}")
    
    output_choice = int(input("\nEnter choice (1-5): "))
    output_language = list(LANGUAGES.keys())[output_choice - 1]
    
    print(f"\n✓ Input Language: {input_language}")
    print(f"✓ Output Language: {output_language}")
    if device_index is not None:
        print(f"✓ Microphone: {mic_list[device_index]}")
    print("\n" + "=" * 60)
    
    # Initialize state
    initial_state = {
        "input_language": input_language,
        "output_language": output_language,
        "audio_input": "",
        "recognized_text": "",
        "translated_text": "",
        "output_audio_file": "",
        "error": "",
        "status": "initialized",
        "device_index": device_index
    }
    
    # Create and run the graph
    graph = create_translator_graph()
    
    try:
        result = graph.invoke(initial_state)
        
        if result.get('error'):
            print(f"\n❌ Error: {result['error']}")
        else:
            print("\n" + "=" * 60)
            print("✅ TRANSLATION COMPLETED!")
            print("=" * 60)
            print(f"\n📝 Original Text ({input_language}):")
            print(f"   {result['recognized_text']}")
            print(f"\n📝 Translated Text ({output_language}):")
            print(f"   {result['translated_text']}")
            print(f"\n🔊 Audio File: {result['output_audio_file']}")
            print("\n" + "=" * 60)
            
    except Exception as e:
        print(f"\n❌ Application Error: {str(e)}")

if __name__ == "__main__":
    print("📦 Required packages:")
    print("   pip install langgraph speechrecognition pyaudio gtts openai")
    print()
    
    try:
        import langgraph
        import speech_recognition
        from gtts import gTTS
        from openai import OpenAI
        print("✓ All required packages are installed!\n")
        
        import os
        if not os.getenv('OPENAI_API_KEY'):
            print("⚠️ WARNING: OPENAI_API_KEY environment variable not set!")
            print("   Set it using: export OPENAI_API_KEY='your-api-key-here'")
            print("   Or in Windows: set OPENAI_API_KEY=your-api-key-here")
            print("   The app will use fallback translation without API key.\n")
        else:
            print("✓ OpenAI API key found!\n")
        
        main()
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install required packages using:")
        print("pip install langgraph speechrecognition pyaudio gtts openai")

        
!pip install langgraph
!pip install speechrecognition
!apt-get install -y portaudio19-dev
!pip install  pyaudio
!pip install  gtts
!pip install  openai
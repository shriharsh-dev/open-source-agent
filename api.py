import requests
import json
import time
import os
from pathlib import Path
import wave
import numpy as np
from typing import Optional

# Base URL for your FastAPI backend
BASE_URL = "http://localhost:8000"

class VoiceAIClient:
    """Client class for interacting with the Voice AI Backend"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session_id = None
        
    def health_check(self) -> dict:
        """Check if the API is healthy"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def create_session(self, proficiency_level: str = "intermediate", 
                      scenario: Optional[str] = None) -> dict:
        """Create a new conversation session"""
        session_data = {"proficiency_level": proficiency_level}
        if scenario:
            session_data["scenario"] = scenario
            
        response = requests.post(f"{self.base_url}/session/create", json=session_data)
        response.raise_for_status()
        
        result = response.json()
        self.session_id = result["session_id"]
        return result
    
    def chat_text(self, message: str, session_id: Optional[str] = None) -> dict:
        """Send text message and get AI response"""
        if not session_id:
            session_id = self.session_id
        if not session_id:
            raise ValueError("No session ID available. Create a session first.")
            
        text_data = {
            "session_id": session_id,
            "message": message
        }
        response = requests.post(f"{self.base_url}/chat/text", json=text_data)
        response.raise_for_status()
        return response.json()
    
    def chat_voice(self, audio_file_path: str, session_id: Optional[str] = None) -> dict:
        """Upload voice file and get text response"""
        if not session_id:
            session_id = self.session_id
        if not session_id:
            raise ValueError("No session ID available. Create a session first.")
            
        with open(audio_file_path, "rb") as f:
            files = {"audio": ("recording.wav", f, "audio/wav")}
            data = {"session_id": session_id}
            response = requests.post(f"{self.base_url}/chat/voice", files=files, data=data)
            
        response.raise_for_status()
        return response.json()
    
    def generate_speech(self, text: str, session_id: Optional[str] = None, 
                       output_file: str = "output.wav") -> str:
        """Generate speech from text and save to file"""
        if not session_id:
            session_id = self.session_id
        if not session_id:
            raise ValueError("No session ID available. Create a session first.")
            
        params = {"session_id": session_id, "text": text}
        response = requests.post(f"{self.base_url}/tts/generate", params=params)
        response.raise_for_status()
        
        with open(output_file, "wb") as f:
            f.write(response.content)
        
        return output_file
    
    def voice_to_voice(self, audio_file_path: str, session_id: Optional[str] = None,
                      output_file: str = "response.wav") -> dict:
        """Complete voice pipeline: voice input → AI voice output"""
        if not session_id:
            session_id = self.session_id
        if not session_id:
            raise ValueError("No session ID available. Create a session first.")
            
        with open(audio_file_path, "rb") as f:
            files = {"audio": ("recording.wav", f, "audio/wav")}
            data = {"session_id": session_id}
            response = requests.post(f"{self.base_url}/chat/voice-to-voice", files=files, data=data)
            
        response.raise_for_status()
        
        # Save audio response
        with open(output_file, "wb") as f:
            f.write(response.content)
        
        # Extract text from headers
        return {
            "transcribed_text": response.headers.get("X-Transcribed-Text", ""),
            "response_text": response.headers.get("X-Response-Text", ""),
            "audio_file": output_file
        }
    
    def get_history(self, session_id: Optional[str] = None) -> dict:
        """Get conversation history"""
        if not session_id:
            session_id = self.session_id
        if not session_id:
            raise ValueError("No session ID available. Create a session first.")
            
        response = requests.get(f"{self.base_url}/session/{session_id}/history")
        response.raise_for_status()
        return response.json()
    
    def delete_session(self, session_id: Optional[str] = None) -> dict:
        """Delete session"""
        if not session_id:
            session_id = self.session_id
        if not session_id:
            raise ValueError("No session ID available.")
            
        response = requests.delete(f"{self.base_url}/session/{session_id}")
        response.raise_for_status()
        
        if session_id == self.session_id:
            self.session_id = None
            
        return response.json()

def create_test_audio(filename: str = "test_audio.wav", duration: float = 2.0):
    """Create a simple test audio file with a tone"""
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate a simple tone (440Hz A note)
    audio = np.sin(2 * np.pi * 440 * t) * 0.3
    
    # Convert to 16-bit PCM
    audio_int = (audio * 32767).astype(np.int16)
    
    # Save as WAV file
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int.tobytes())
    
    return filename

def run_comprehensive_tests():
    """Run comprehensive API tests"""
    print("🧪 Testing Voice AI Backend API")
    print("=" * 40)
    
    client = VoiceAIClient()
    
    try:
        # 1. Health check
        print("\n1. 🏥 Health Check:")
        health = client.health_check()
        print(f"✅ Status: {health['status']}")
        print(f"   Whisper: {health['whisper']}")
        print(f"   TTS: {health['tts']}")
        print(f"   Ollama: {health['ollama']}")
        print(f"   Active sessions: {health['active_sessions']}")
        
        # 2. Create session
        print("\n2. 🆕 Creating Session:")
        session = client.create_session(
            proficiency_level="intermediate",
            scenario="job interview"
        )
        print(f"✅ Session ID: {client.session_id[:8]}...")
        print(f"   Message: {session['message']}")
        
        # 3. Text conversation
        print("\n3. 💬 Text Conversation:")
        messages = [
            "Hello, I'm here for the job interview. How are you today?",
            "I have 5 years of experience in software development.",
            "Thank you for the interview opportunity!"
        ]
        
        for i, message in enumerate(messages, 1):
            print(f"\n   👤 User {i}: {message}")
            response = client.chat_text(message)
            print(f"   🤖 AI: {response['message']}")
            time.sleep(1)  # Small delay between messages
        
        # 4. Show conversation history
        print("\n4. 📜 Conversation History:")
        history = client.get_history()
        print(f"   Total messages: {len(history['history'])}")
        print(f"   Config: {history['config']}")
        
        # 5. Test TTS
        print("\n5. 🔊 Text-to-Speech Test:")
        test_text = "Hello! This is a test of the text to speech system. How does it sound?"
        audio_file = client.generate_speech(test_text, output_file="test_tts_output.wav")
        print(f"✅ TTS audio saved as '{audio_file}'")
        
        # 6. Test voice processing (with synthetic audio)
        print("\n6. 🎤 Voice Processing Test:")
        if not os.path.exists("test_audio.wav"):
            print("   Creating test audio file...")
            create_test_audio("test_audio.wav")
        
        try:
            # Note: This will likely fail with synthetic audio, but tests the endpoint
            voice_result = client.chat_voice("test_audio.wav")
            print(f"✅ Transcribed: '{voice_result['transcribed_text']}'")
            print(f"   AI Response: {voice_result['message']}")
        except Exception as e:
            print(f"⚠️  Voice test failed (expected with synthetic audio): {e}")
        
        # 7. Clean up
        print("\n7. 🧹 Cleanup:")
        client.delete_session()
        print("✅ Session deleted")
        
        # Clean up test files
        for file in ["test_tts_output.wav", "test_audio.wav"]:
            if os.path.exists(file):
                os.remove(file)
                print(f"   Removed {file}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running:")
        print("   python your_backend_file.py")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

def interactive_chat_demo():
    """Interactive chat demo"""
    print("\n🎯 Interactive Chat Demo")
    print("=" * 25)
    
    client = VoiceAIClient()
    
    try:
        # Setup session
        print("Setting up session...")
        proficiency = input("Enter proficiency level (beginner/intermediate/advanced) [intermediate]: ").strip()
        if not proficiency:
            proficiency = "intermediate"
            
        scenario = input("Enter scenario (optional): ").strip()
        if not scenario:
            scenario = None
            
        client.create_session(proficiency, scenario)
        print(f"✅ Session created: {client.session_id[:8]}...")
        
        print("\n💬 Chat started! Type 'quit' to exit, 'history' to see conversation history")
        print("-" * 50)
        
        while True:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'history':
                history = client.get_history()
                print("\n📜 Conversation History:")
                for msg in history['history']:
                    role = "👤" if msg['role'] == 'user' else "🤖"
                    print(f"   {role} {msg['role'].title()}: {msg['content']}")
                continue
            elif not user_input:
                continue
            
            try:
                response = client.chat_text(user_input)
                print(f"🤖 AI: {response['message']}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Cleanup
        client.delete_session()
        print("\n👋 Chat ended. Session deleted.")
        
    except KeyboardInterrupt:
        print("\n\n👋 Chat interrupted by user.")
        if client.session_id:
            client.delete_session()
    except Exception as e:
        print(f"❌ Demo failed: {e}")

def main():
    """Main function with menu"""
    print("🎯 Voice AI Backend Test Client")
    print("=" * 32)
    print("\nChoose an option:")
    print("1. Run comprehensive API tests")
    print("2. Interactive chat demo")
    print("3. Show API usage examples")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            success = run_comprehensive_tests()
            if success:
                print("\n✅ All tests completed successfully!")
            break
            
        elif choice == "2":
            interactive_chat_demo()
            break
            
        elif choice == "3":
            show_usage_examples()
            break
            
        elif choice == "4":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-4.")

def show_usage_examples():
    """Show usage examples for different scenarios"""
    examples = '''
🔧 API Usage Examples:

1. Basic Text Chat:
   client = VoiceAIClient()
   client.create_session("beginner", "restaurant ordering")
   response = client.chat_text("I want to order food")
   print(response['message'])

2. Voice Processing:
   # Record audio with your app, then:
   result = client.chat_voice("user_recording.wav")
   print(f"You said: {result['transcribed_text']}")
   print(f"AI replied: {result['message']}")

3. Text-to-Speech:
   audio_file = client.generate_speech("Hello, how are you?")
   # Now play audio_file in your app

4. Complete Voice Pipeline:
   result = client.voice_to_voice("input.wav", output_file="response.wav")
   # result contains transcription, AI text, and saves AI voice as response.wav

5. Session Management:
   client.create_session("advanced", "business meeting")
   history = client.get_history()
   client.delete_session()

🎯 Integration Tips:
- Use the VoiceAIClient class in your mobile app
- Handle audio recording/playback in your UI
- Store session IDs for multiple conversations
- Use appropriate error handling for network issues
'''
    print(examples)

if __name__ == "__main__":
    main()
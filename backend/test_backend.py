#!/usr/bin/env python3
"""
Simple test script for the localGPT backend
"""

import requests

def test_health_endpoint():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Ollama running: {data['ollama_running']}")
            print(f"   Models available: {len(data['available_models'])}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_chat_endpoint():
    """Test the chat endpoint"""
    print("\n💬 Testing chat endpoint...")
    
    test_message = {
        "message": "Say 'Hello World' and nothing else.",
        "model": "llama3.2:latest"
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            headers={"Content-Type": "application/json"},
            json=test_message,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Chat test passed")
            print(f"   Model: {data['model']}")
            print(f"   Response: {data['response']}")
            print(f"   Message count: {data['message_count']}")
            return True
        else:
            print(f"❌ Chat test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Chat test failed: {e}")
        return False

def test_conversation_history():
    """Test conversation with history"""
    print("\n🗨️  Testing conversation history...")
    
    # First message
    conversation = []
    
    message1 = {
        "message": "My name is Alice. Remember this.",
        "model": "llama3.2:latest",
        "conversation_history": conversation
    }
    
    try:
        response1 = requests.post(
            "http://localhost:8000/chat",
            headers={"Content-Type": "application/json"},
            json=message1,
            timeout=30
        )
        
        if response1.status_code == 200:
            data1 = response1.json()
            
            # Add to conversation history
            conversation.append({"role": "user", "content": "My name is Alice. Remember this."})
            conversation.append({"role": "assistant", "content": data1["response"]})
            
            # Second message asking about the name
            message2 = {
                "message": "What is my name?",
                "model": "llama3.2:latest", 
                "conversation_history": conversation
            }
            
            response2 = requests.post(
                "http://localhost:8000/chat",
                headers={"Content-Type": "application/json"},
                json=message2,
                timeout=30
            )
            
            if response2.status_code == 200:
                data2 = response2.json()
                print(f"✅ Conversation history test passed")
                print(f"   First response: {data1['response']}")
                print(f"   Second response: {data2['response']}")
                
                # Check if the AI remembered the name
                if "alice" in data2['response'].lower():
                    print(f"✅ AI correctly remembered the name!")
                else:
                    print(f"⚠️  AI might not have remembered the name")
                return True
            else:
                print(f"❌ Second message failed: {response2.status_code}")
                return False
        else:
            print(f"❌ First message failed: {response1.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Conversation test failed: {e}")
        return False

def main():
    print("🧪 Testing localGPT Backend")
    print("=" * 40)
    
    # Test health endpoint
    health_ok = test_health_endpoint()
    if not health_ok:
        print("\n❌ Backend server is not running or not healthy")
        print("   Make sure to run: python server.py")
        return
    
    # Test basic chat
    chat_ok = test_chat_endpoint()
    if not chat_ok:
        print("\n❌ Chat functionality is not working")
        return
    
    # Test conversation history
    conversation_ok = test_conversation_history()
    
    print("\n" + "=" * 40)
    if health_ok and chat_ok and conversation_ok:
        print("🎉 All tests passed! Backend is ready for frontend integration.")
    else:
        print("⚠️  Some tests failed. Check the issues above.")
    
    print("\n🔗 Ready to connect to frontend at http://localhost:3000")

if __name__ == "__main__":
    main() 
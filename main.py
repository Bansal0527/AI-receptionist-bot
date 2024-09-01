import asyncio
import google.generativeai as genai
from chromadb import Client
from chromadb.config import Settings
import logging
import os 

logging.basicConfig(level=logging.INFO)

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-1.5-flash')

if not os.path.exists("./chroma_db"):
    os.makedirs("./chroma_db")

chroma_client = Client(Settings(persist_directory="./chroma_db"))

collection = chroma_client.get_or_create_collection(name="emergency_info")

def initialize_emergency_info():
    emergency_data = [
        ("not breathing", "Perform CPR: Push against the chest of the patient and blow air into their mouth in a constant rhythm."),
        ("severe bleeding", "Apply direct pressure to the wound using a clean cloth or bandage."),
        ("heart attack", "Help the person sit down, loosen tight clothing, and call for emergency medical help immediately.")
    ]
    collection.add(
        documents=[info for _, info in emergency_data],
        metadatas=[{"emergency": emergency} for emergency, _ in emergency_data],
        ids=[f"emergency_{i}" for i in range(len(emergency_data))]
    )

async def get_gemini_response(prompt):
    response = await asyncio.wait_for(model.generate_content_async(prompt), timeout=10)
    return response.text

async def get_emergency_action(emergency):
    await asyncio.sleep(15)
    results = collection.query(
        query_texts=[emergency],
        n_results=1
    )
    if results['distances'][0]:
        return results['documents'][0][0]
    return "I'm sorry, I don't have specific information for that emergency. Please wait for the doctor to arrive."

async def main():
    print("AI: Welcome to Dr. Adrin's office. Is this an emergency or would you like to leave a message?")

    while True:
        user_input = input("User: ").lower().strip()
        if not user_input:
            print("AI: I didn't catch that. Is this an emergency or would you like to leave a message?")
            continue

        is_emergency = await get_gemini_response(f"Identify if the given statement is an emergency or not? Return only 'yes' or 'no' nothing more. {user_input} ")
        if "yes" in is_emergency.lower():
            print("AI: I understand this is an emergency. Can you please tell me what the emergency is?")
            emergency = input("User: ").strip()
            if not emergency:
                print("AI: I didn't catch that. Can you please tell me what the emergency is?")
                continue

            print("AI: I am checking what you should do immediately. Meanwhile, can you tell me which area you are located right now?")

            emergency_action_task = asyncio.create_task(get_emergency_action(emergency))

            location = input("User: ").strip()
            if not location:
                print("AI: I didn't catch that. Can you please tell me which area you are located right now?")
                continue

            arrival_time = await get_gemini_response(f"Generate a random estimated time of arrival for a doctor to {location}. Only return the time nothing else")
            print(f"AI: Dr. Adrin will be coming to your location immediately. {arrival_time}")

            user_response = input("User: ").strip()
            action = await emergency_action_task
            if "too late" in user_response.lower():
                print("AI: Please hold just a sec...")
                
                print(f"AI: I understand that you are worried that Dr. Adrin will arrive too late. {action}")
                print("AI: Don't worry, please follow these steps. Dr. Adrin will be with you shortly.")
                break
                
            print(f"AI: Don't worry, please follow these steps. Dr. Adrin will be with you shortly.{action}")
            break
            
            
            

        elif "message" in user_input:
            print("AI: Please leave your message for Dr. Adrin.")
            message = input("User: ").strip()
            if not message:
                print("AI: I didn't catch that. Please leave your message for Dr. Adrin.")
                continue
            print("AI: Thanks for the message. We will forward it to Dr. Adrin.")
            break

        else:
            clarification = await get_gemini_response(f"Can you help clarify the following statement: {user_input}")
            print(f"AI: {clarification}")

if __name__ == "__main__":
    initialize_emergency_info()
    asyncio.run(main())
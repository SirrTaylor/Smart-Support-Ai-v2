import os
from dotenv import load_dotenv
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

load_dotenv()

#Key and Endpoints
key = os.getenv("AZURE_LANGUAGE_KEY")
endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")

#safety check for key and endpoint
if not key:
    print("ERROR: COULD NOT FIND KEY CHECK env file")
if not endpoint:
    print("ERROR: COULD NOT FIND ENDPOINT CHECK env file")

def authenticate_client():
    return TextAnalyticsClient(endpoint=endpoint, credential = AzureKeyCredential(key))

def run_opinion_test(client,user_text):
    #sending the text to germany west central for processing and extract deep insights
    results = client.analyze_sentiment (documents=[user_text],show_opinion_mining = True)
   
    for doc in results:
        print(f"\n---Analysis for: `{user_text}`---")
        print(f"Overall mood: {doc.sentiment.upper()}")

    #checking each sentence for a target and
    for sentence in doc.sentences:
        if sentence.mined_opinions:
            print("\n Specific insights found: ")
            for opinion in sentence.mined_opinions:
                #our target is the subject of the text
                target= opinion.target
                print(f"Item: `{target.text}`(Sentiment: {target.sentiment.upper()})")
                #each assessment is an adjective e.g "slow"
            for assessment in opinion.assessments:
                print(f"Description used: `{assessment.text}`") 
        else:
            print("\n No specific opinions found in this sentence.")     

#test 4
if __name__ == "__main__":
    client = authenticate_client()

test_message = "the food was very delicious but the waiter was very rude"
run_opinion_test(client,test_message)



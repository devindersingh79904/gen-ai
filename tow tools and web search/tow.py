from openai import OpenAI
import json

client = OpenAI()

def get_user_preferences(user_id,category):

    preferences = {
        "user123":{
            "food":["Italian","Maxican","Vegitarian options"],
            "movies":["sci-fi","comedy","drama"],
            "books":["non-ficiton","technology","history"]
        }
    }

    user_pref = preferences.get(user_id,{})
    return user_pref.get(category,["No Preferences found"])



tools = [
    {
        "type":"function",
        "name":"get_user_preferences",
        "description": "Get the preferences of a user based on their id and category",
        "parameters":{
            "type":"object",
            "properties":{
                "user_id":{
                    "type":"string",
                    "description":"The id of the user"
                },
                "category":{
                    "type":"string",
                    "description":"The category of the preferences"
                }
            },
            "required":["user_id","category"],
            "additionalProperties":False
        },
        "strict":True
    },  
]

input_message=[
    {
        "role":"user",
        "content":"i am user123. can you recommend me a some restuarants ased on my food preferences? in the bangalore 560037 is the pincode and my range is 500 to 1000 rs and time is dinner."
    }
]
print("Step1: Making Initial call to the model....")

response = client.responses.create(
    model="gpt-5-mini",
    input=input_message,
    tools=tools
)

print("Step2 AI's Function call")
print(response.output)

input_message += response.output

print("Step3 : processing function calls....")

for item in response.output:
    if item.type == "function_call" and item.name == "get_user_preferences":
        args = json.loads(item.arguments)
        print(f" -> calling get_user_preferences with : {args}")
        preferences = get_user_preferences(args["user_id"],args["category"])
        print(f" -> function returned: {preferences}")

        input_message.append({
            "type":"function_call_output",
            "call_id":item.call_id,
            "output":json.dumps({
                "preferences":preferences
            })
        })
        
        
print("\n Step4: Getting final responese with function result ")

tools.append(
    {
        "type": "web_search"
    }
)
final_response = client.responses.create(
    model="gpt-5-mini",
    input=input_message,
    tools=tools
)

print("\n" + "-" * 60)
print("Final Answer:")
print("-" * 20)
print(final_response.output_text)
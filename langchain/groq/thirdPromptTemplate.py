# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from langchain_core.prompts import PromptTemplate


llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

email_template = """
            Create an invitation email to the recipient that is {name}
            for and event tht is {eventName}
            in language that is {language}.
            Menion the event location that is {eventLocation}
            and event date is {eventDate}
            Also write few sentence about the event descrioption that is {eventDescription}
            int a style that is {style}.
            """

details = {
    "name": "Sarvpreet Singh",
    "eventName": "Birthday",
    "language": "Punjabi",
    "eventLocation": "bangalore",
    "eventDate": "2026-07-12",
    "eventDescription": "Birthday party all night",
    "style": "Enthusiastic and funny"
}


email_prompt_template = PromptTemplate.from_template(email_template)

prompt_value = email_prompt_template.invoke(details)
response = llm.invoke(prompt_value)
print(response.content)
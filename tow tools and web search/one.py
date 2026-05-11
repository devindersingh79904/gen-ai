from operator import truediv
from concurrent.futures._base import DoneAndNotDoneFutures
from openai import OpenAI
import threading
import time

client = OpenAI()

def show_progress():
    chars = "|/-\\"
    i = 0
    while not done:
        print(f"\r Searching the web{chars[i % len(chars)]}",end='',flush=True)
        i += 1
        time.sleep(0.1)
    print("\r Searching the web complete   ")

done = False
thread = threading.Thread(target=show_progress)
thread.start()

start_time = time.time()

response = client.responses.create(
    model="gpt-5-mini",
    tools=[
        {
            "type": "web_search"
        }
    ],
    input = "who won the match yesterday in ipl mi vs lsg"
)

end_time = time.time()
done = True
thread.join

elapsed_time = end_time-start_time
print(f"\n Search completed in {elapsed_time:.2f} seconds")
print("-"*60)
print(response.output_text)
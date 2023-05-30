import openai

class ChatGPT():

    def __init__(self) -> None:
        self.model = "gpt-3.5-turbo"
        self.fewshot_prompt = self.get_fewshot_prompt()
        self.response = None

    def get_fewshot_prompt(self):
        return \
        "You are identifying the target objects T in some pormpts P. Some examples are\n" + \
        "P: Pick up the banana\n" + \
        "T: banana\n" + \
        "P: Lift the bottle\n" + \
        "T: bottle\n" + \
        "P: Push the toaster\n" + \
        "T: toaster\n"
    
    def __call__(self, task_desc):
        prompt = "Complete with the target object of the following prompt\n" + \
                f"P: {task_desc}\n" + \
                 "T:"
        prompt = self.fewshot_prompt + prompt

        # print(self.fewshot_prompt, "\n")

        # print("\n\n\n\n\n\n\n\n NANA")

        # print(prompt, "\n")

        self.response = openai.ChatCompletion.create(
                            model=self.model,
                            messages=[
                                    {"role": "user", "content": prompt}
                                ]
                        )
        # print(self.response)
        
        return self.response["choices"][0]["message"]["content"]
    
# # Open AI example

# openai.ChatCompletion.create(
#   model="gpt-3.5-turbo",
#   messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Who won the world series in 2020?"},
#         {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#         {"role": "user", "content": "Where was it played?"}
#     ]
# )

if __name__ == "__main__":
    import sys

    chatgpt = ChatGPT()
    task_desc = sys.argv[1]
    print(f"What should the robot look for in the task: {task_desc}")
    print(chatgpt(task_desc))
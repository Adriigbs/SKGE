


class PromptParser:
    
    def __init__(self, task_id):
        self.task = task_id
    

    def parse(self, prompt):
        # Split string
        # Left side until second [STATEMENT] appears
        # Right side is the rest of the string

        delimiter = "[STATEMENT]"

        left, sep, rest = prompt.partition(delimiter)
        middle, second_delimiter, right = rest.partition(delimiter)

        system_prompt = left + delimiter + middle
        user_prompt = second_delimiter + right

        return system_prompt, user_prompt
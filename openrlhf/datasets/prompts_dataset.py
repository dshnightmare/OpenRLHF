from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none


def preprocess_data(data, input_template=None, input_key=None, output_key=None, relative_key = None) -> str:
    # custom dataset
    if input_key:
        prompt = data[input_key]
    else:
        # Dahoas/full-hh-rlhf
        if exist_and_not_none(data, "prompt"):
            prompt = data["prompt"]
            # tasksource/oasst1_pairwise_rlhf_reward
            if prompt.startswith("prompter:"):
                prompt = (
                    prompt.replace("prompter:", "\nHuman: ").replace("assistant:", "\nAssistant: ") + "\nAssistant: "
                )
            input_template = None  # do not modified with input template again
        # Open-Orca/OpenOrca
        elif exist_and_not_none(data, "system_prompt") and exist_and_not_none(data, "response"):
            prompt = data["system_prompt"] + "\n" + data["question"]
        # BelleGroup/train_0.5M_CN
        # LLMs/Alpaca-ShareGPT
        # yahma/alpaca-cleaned
        # QingyiSi/Alpaca-CoT
        elif exist_and_not_none(data, "instruction") and exist_and_not_none(data, "output"):
            input = " " + data["input"] if exist_and_not_none(data, "input") else ""
            prompt = data["instruction"] + input
        # lmsys/chatbot_arena_conversations
        elif exist_and_not_none(data, "winner") and exist_and_not_none(data, "conversation_a"):

            def process_chatbot_arena_conversations(lll):
                result = []
                for l in lll:
                    if "user" in l["role"]:
                        result.append(input_template.format(l["content"]))
                    else:
                        result.append(l["content"])
                return "\n".join(result)

            prompt = data["conversation_a"][:-1]
            prompt = process_chatbot_arena_conversations(prompt)
            input_template = None  # do not modified with input template again
        # openai/webgpt_comparisons
        elif exist_and_not_none(data, "question") and exist_and_not_none(data, "answer_1"):
            prompt = data["question"]["full_text"]
        else:
            raise ValueError("Unknown prompts dataset")

    res = list()
    # input template
    if input_template:
        prompt = input_template.format(prompt)
    res.append(prompt)
    # if output_key:
    #     response = data[output_key]
    #     return prompt, response
    # else:
    #     return prompt
    if output_key:
        response = data[output_key]
        res.append(response)
    if relative_key:
        relative_reward = data[relative_key]
        res.append(relative_reward)
    return res
    

class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template="Human: {}\nAssistant: ",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)

        self.prompts = []
        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_template, input_key)
            self.prompts.append(prompt)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]


class PromptWithResponseDataset(Dataset):
    """
    Dataset for PPO model, also return response for reference

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template="Human: {}\nAssistant: ",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        output_key = getattr(self.strategy.args, "output_key", None)
        assert output_key is not None, "output_key is required for PromptWithResponseDataset"

        self.prompts = []
        self.response = []
        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, response = preprocess_data(data, input_template, input_key, output_key)
            self.prompts.append(prompt)
            self.response.append(response)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx], self.response[idx]
    
class PromptWithResponseRelativeRewardDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template="Human: {}\nAssistant: ",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        output_key = getattr(self.strategy.args, "output_key", None)
        relative_key = getattr(self.strategy.args, "relative_key", None)
        assert output_key and relative_key, "output_key and relative_key are required for PromptWithResponseRelativeRewardDataset"
        
        self.prompts = []
        self.response = []
        self.relative_reward = []
        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, response, relative_reward = preprocess_data(data, input_template, input_key, output_key, relative_key)
            self.prompts.append(prompt)
            self.response.append(response)
            self.relative_reward.append(relative_reward)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx], self.response[idx], self.relative_reward[idx]
import torch
from torchvision.transforms import Compose
import nltk
import numpy as np
from torch.nn.utils.rnn import pad_sequence

#Conver the sequences into one hot vectors:
class OneHotEncode:
  def  __init__(self,alphabet):
    self.alphabet_size = len(alphabet) + 1
  def __call__(self, batch):
    return torch.eye(self.alphabet_size)[batch].float()


class MathTokenEmbedding:

    def __init__(self, alphabet, padding_token=" "):

        self.token_to_idx = { a: idx + 1 for idx, a in enumerate(alphabet)}
        self.idx_to_token = { idx + 1 : a for idx, a in enumerate(alphabet)}

        self.token_to_idx[padding_token] = 0
        self.idx_to_token[0] = padding_token

    def embed(self, tokens):
        return list(map(lambda t: self.token_to_idx[t], tokens))

    def decode(self, embeddings, pretty_print=False):

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.tolist()

        decoded = list(map(lambda e: self.idx_to_token[e], embeddings))

        if pretty_print:
            return " ".join(decoded).strip()

        return decoded

    def __call__(self, x):
        return self.embed(x)


class ToTensor:

    def __init__(self, dtype):
        self.dtype =dtype

    def __call__(self, x):
        return torch.tensor(x, dtype=self.dtype)


class PadSequencesToSameLength:

    def __call__(self, sequences):
        return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)



class PadSequencesToSameLengthV2:
    def __init__(self, padding_value=0, max_length=None):
        self.padding_value = padding_value
        self.max_length = max_length

    def __call__(self, batch):
        # Pad each sequence to the same length
        batch_padded = pad_sequence(batch, batch_first=True, padding_value=self.padding_value)
        current_max_length = batch_padded.shape[1]
        #print(current_max_length)
        if self.max_length is not None:
            if current_max_length < self.max_length:
              # Further padding required
              padding_size = self.max_length - current_max_length
              additional_padding = torch.full((len(batch), padding_size), self.padding_value, dtype=torch.int64)
              batch_padded = torch.cat([batch_padded, additional_padding], dim=1)

        return batch_padded


class RuleTokenEmbedding:

    def __init__(self, cfg, padding_rule="Nothing -> None", max_num_rules = 16,one_hot_encode=False):

        self.parser = nltk.ChartParser(cfg)
        self.num_rules = len(cfg.productions())
        self.rule_to_idx = {str(rule): i for i, rule in enumerate(cfg.productions())}
        self.idx_to_rule = {idx : str(a) for idx, a in enumerate(cfg.productions())}
        self.max_num_rules = max_num_rules
        self.one_hot_encode = one_hot_encode

    def map_tree_to_rules(self,tree):
        rules_used = []
        for prod in tree.productions():
            # Format the rule to match the CFG
            left = str(prod.lhs())
            right = ' '.join(f"'{str(s)}'" if isinstance(s, str) else str(s) for s in prod.rhs())
            rule = f"{left} -> {right}"
            rules_used.append(rule)
        return rules_used

    def one_hot_encode_rule(self,rule):
        encoding = np.zeros(self.num_rules)
        encoding[self.rule_to_idx[rule]] = 1
        return encoding

    def embed(self, tokens):
        parsed_trees = list(self.parser.parse(tokens))
        # Map the parse tree to grammar rules
        rules_used_in_parsing = [self.map_tree_to_rules(parsed_tree) for parsed_tree in parsed_trees][0]
        # One-hot encode the used rules
        encoded_rules_in_parsing = []
        for each_rule in rules_used_in_parsing:
            if(self.one_hot_encode == True):
                encoded_rule = self.one_hot_encode_rule(each_rule)
            else:
                encoded_rule = self.rule_to_idx[each_rule]
            encoded_rules_in_parsing.append(encoded_rule)

        if(self.one_hot_encode == True):
            pad_vector = np.zeros(self.num_rules)
            pad_vector[-1] = 1
            while(len(encoded_rules_in_parsing)<self.max_num_rules):
                encoded_rules_in_parsing.append(pad_vector)
        else:
            while(len(encoded_rules_in_parsing)<self.max_num_rules):
                encoded_rules_in_parsing.append(self.num_rules-1)

        return  encoded_rules_in_parsing

    def decode(self, encoded_rules_in_parsing):
        equation = ['S']
        if self.one_hot_encode == True:
            for encoded_rule in encoded_rules_in_parsing:
                #print(encoded_rule)
                if np.argmax(encoded_rule) == self.num_rules - 1:
                    continue
                
                rule_index = np.argmax(encoded_rule)
                rule = self.idx_to_rule[rule_index]
                
                left, right = rule.split(' -> ')
                #right = right.strip("'").split()
                right = right.replace("'","").split()
                for i, symbol in enumerate(equation):
                    if symbol == left:
                        equation[i:i+1] = right
                        break
        else:
            for encoded_rule in encoded_rules_in_parsing:
                #print(encoded_rule)
                if encoded_rule == self.num_rules - 1:
                    continue
                rule = self.idx_to_rule[encoded_rule]
                
                left, right = rule.split(' -> ')
                #right = right.strip("'").split()
                right = right.replace("'","").split()
                for i, symbol in enumerate(equation):
                    if symbol == left:
                        equation[i:i+1] = right
                        break
        return equation

    def __call__(self, x):
        return self.embed(x)
import torch
from torch.utils.data import DataLoader

from ac_dll_grammar_vae import print_contact_info
from ac_dll_grammar_vae.data import CFGEquationDataset
from ac_dll_grammar_vae.data.alphabet import alphabet
from ac_dll_grammar_vae.data.transforms import MathTokenEmbedding, ToTensor, Compose, PadSequencesToSameLength


def main():
    data = CFGEquationDataset(n_samples=100)
    print("Dataset initialized.")

    data.save("equations.txt")
    print("Dataset saved to file.")

    emb = MathTokenEmbedding(alphabet=alphabet)

    x = data[42]
    x_emb = emb.embed(x)
    print(x)
    print(x_emb)
    print(emb.decode(x_emb))

    training = CFGEquationDataset(
        n_samples=100000,
        transform=Compose([
            MathTokenEmbedding(alphabet),
            ToTensor(dtype=torch.uint8)
        ]))

    training_loader = DataLoader(dataset=training,
                                 batch_size=256,
                                 shuffle=True,
                                 collate_fn=PadSequencesToSameLength())

    for X in training_loader:
        print(X.shape)
        eq_dec = emb.decode(X[0], pretty_print=True)
        print(eq_dec)

    print_contact_info()


if __name__ == "__main__":
    main()
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast
special_tokens = {
    "pad_token": "<PAD>",
    "unk_token": "<UNK>",
    "bos_token": "<BOS>",
    "eos_token": "<EOS>",
    "mask_token": "<MASK>"
}

def main():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=3000, min_frequency=2, show_progress=True,special_tokens=list(special_tokens.values()))
    tokenizer.train(["/home/zeyu/mnt/drive0/dataset/driving/BDD-X/BDD-X-Dataset/sentence.txt"], trainer)

    bos_id = tokenizer.token_to_id("<BOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    unk_id = tokenizer.token_to_id("<UNK>")
    pad_id = tokenizer.token_to_id("<PAD>")
    mask_id = tokenizer.token_to_id("<MASK>")

    tokenizer.post_processor=processors.TemplateProcessing(
        single="<BOS> $A <EOS>",
        special_tokens=[("<BOS>", bos_id), ("<EOS>", eos_id)],
    )
    
    pretrained = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,max_length=512,
        pad_token="<PAD>",
        unk_token="<UNK>",
        bos_token="<BOS>",
        eos_token="<EOS>",
        mask_token="<MASK>")
    pretrained.save_pretrained("/home/zeyu/mnt/drive0/dataset/driving/BDD-X/BDD-X-Dataset/tokenizer")

if __name__=="__main__":
    main()

import yaml
from speechllm.data.dataset import SpeechLLMDataset
from speechllm.codecs.vocab_manager import VocabManager
from speechllm.codecs.audio_tokenizer import AudioTokenizer
from speechllm.align.interleaving import InterleavingGenerator

config_path = 'configs/default_config.yaml'
config = yaml.safe_load(open(config_path, encoding='utf-8'))
data_conf = config['data']

vm = VocabManager(
    base_tokenizer_name=config['model']['llm_model_name'],
    num_rvq_layers=config['model']['num_rvq_layers'],
    codebook_size=config['model']['codebook_size'],
)
atok = AudioTokenizer(vocab_manager=vm)
inter = InterleavingGenerator(audio_tokenizer=atok, vocab_manager=vm)

train = SpeechLLMDataset(
    data_file=data_conf['train_data_file'],
    audio_tokenizer=atok,
    vocab_manager=vm,
    interleaving_generator=inter,
    max_text_length=data_conf['max_text_length'],
    max_audio_length=data_conf['max_audio_length'],
    sample_rate=data_conf['sample_rate'],
    mode_weights=data_conf['mode_weights'],
    cache_audio_tokens=data_conf['cache_audio_tokens'],
    cache_dir=data_conf.get('cache_dir'),
)

print('dataset len', len(train))
first = train[0]
print('keys', first.keys())
print('mode', first['mode'])
print('input_ids shape', first['input_ids'].shape)
print('labels shape', first['labels'].shape)
print('label valid count', (first['labels'] > -100).sum().item())
print('label unique', first['labels'].unique()[:10])
print('attention sum', first['attention_mask'].sum().item())

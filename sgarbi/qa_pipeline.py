from transformers import Text2TextGenerationPipeline, T5ForConditionalGeneration, AutoTokenizer
from transformers.pipelines import PIPELINE_REGISTRY
import torch
import re
from sentence_transformers import SentenceTransformer, util
import tqdm
import json
import os

CLEAN_TOKENS = ['</s>', '<pad>']

class QABuilderPipeline(Text2TextGenerationPipeline):
    
    def __init__(self, device=None,  *args, **kwargs):
        model = T5ForConditionalGeneration.from_pretrained('sgarbi/t5-qa-builder')
        tokenizer = AutoTokenizer.from_pretrained('sgarbi/t5-qa-builder')
        
        super().__init__(model, tokenizer, *args, **kwargs)
        self.device = device if device is not None else torch.device('cpu')
        self.embedding_layer = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        self.cache = {}
        
    def split_text_into_sentences(self, text):
        sentences = re.split(r'(?<=[.!?]) +', text)
        return sentences

    def split_text_into_chunks_preserving_sentences(self, text, chunk_size=300, stride=50):
        sentences = self.split_text_into_sentences(text)
        chunks = []
        current_chunk = []
        current_chunk_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            # If adding this sentence would exceed the chunk size and there's already content,
            # start a new chunk.
            if current_chunk_size + sentence_size > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_chunk_size = sentence_size
            else:
                # Otherwise, add the sentence to the current chunk.
                current_chunk.append(sentence)
                current_chunk_size += sentence_size
                
            if current_chunk_size >= chunk_size:
                # Calculate how many sentences to move back by based on the stride... needs review
                stride_sentences = max(1, stride // len(' '.join(current_chunk).split()))
                if len(current_chunk) > stride_sentences:
                    next_chunk_start = current_chunk[-stride_sentences:]
                    current_chunk = current_chunk[:-stride_sentences]
                    chunks.append(' '.join(current_chunk))
                    current_chunk = next_chunk_start
                    current_chunk_size = len(' '.join(current_chunk).split())
                else:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_chunk_size = 0

        # Add any remaining sentences as the last chunk.
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
        
    def split_text_into_chunks(self, text, stride, chunk_size=512):
        input_ids = self.tokenizer.encode(text, return_tensors="pt").squeeze()

        if len(input_ids) <= chunk_size:
            return [self.tokenizer.decode(input_ids, skip_special_tokens=True)]

        chunks = []

        for i in range(0, len(input_ids) - chunk_size + 1, stride):
            chunk_ids = input_ids[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)

        # Handle the last chunk to capture any remaining tokens that weren't included due to the stride
        if i + chunk_size < len(input_ids):
            chunk_ids = input_ids[i + stride:]  # Start from last stride position to end of input_ids
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)

        return chunks


    def _get_embedding(self, text):
        return self.embedding_layer.encode(text, convert_to_tensor=True, device=self.device)
    
    def get_embeddings(self, text, cache):
        if text not in cache:
            cache[text] = self._get_embedding(text)
        return cache[text]
    def extract_qa_pairs(self, text):
        pattern = r'<qa_builder_question>\s*(.*?)\s*<qa_builder_answer>\s*(.*?)\s*(?=<qa_builder_question>|$)'
        for token in CLEAN_TOKENS:
            text = text.replace(token, '')
        matches = re.findall(pattern, text, re.DOTALL)
        return matches

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {"max_length": kwargs.pop("max_length", 512)}
        forward_kwargs = {k: v for k, v in kwargs.items()}
        forward_kwargs.update({
            'max_length': 800,
            'num_beams': 5,
            'early_stopping': True,
            'return_dict_in_generate': True,
            'output_scores': True
        })
    
        return preprocess_kwargs, forward_kwargs, {}

    def preprocess(self, inputs, stride, **preprocess_kwargs):
        chunks = self.split_text_into_chunks(inputs, stride=stride)
        model_inputs = [self.tokenizer(f'<qa_builder_context>{chunk}', return_tensors="pt", padding=True, truncation=True, **preprocess_kwargs).to(self.device) for chunk in chunks]
        return zip(model_inputs, chunks)

    def _forward(self, model_inputs, silent_mode=False, **forward_kwargs):
        outputs = []
        inputs = list(model_inputs)
        for model_input, context in tqdm.tqdm(inputs, disable=silent_mode, total=len(inputs), desc="Generating QA pairs"):
            beam_outputs = self.model.generate(**model_input, **forward_kwargs)
            decoded_sequences = [self.tokenizer.decode(generated_sequence, skip_special_tokens=False) for generated_sequence in beam_outputs.sequences]
            sequences_scores = beam_outputs.sequences_scores.cpu().numpy()
            outputs.extend({'chunk_context': context, 'sequence': decoded_sequence, 'score': 100 - abs(score)*100} for decoded_sequence, score in zip(decoded_sequences, sequences_scores))
            
        return outputs
    
    def __call__(self, context, similarity_threshold=0, json_output=False, stride=20, silent_mode=False, **generate_kwargs):
        if context is None:
            return {'error': 'Not enough context to generate QA pairs.'}

        if len(context.split()) < 20:
            return {'error': 'Not enough context to generate QA pairs.'}
        
        preprocess_kwargs, forward_kwargs, _ = self._sanitize_parameters(**generate_kwargs)
        model_inputs = self.preprocess(context, stride, **preprocess_kwargs)


        model_outputs = self._forward(model_inputs, silent_mode=silent_mode, **forward_kwargs)
        return self.postprocess(model_outputs,similarity_threshold, json_output=json_output)

    def postprocess(self, model_outputs, similarity_threshold, json_output=False):
        unique_qa_pairs = []
        embeddings_cache = {}

        for data in model_outputs:
            parsed = self.extract_qa_pairs(data['sequence'])
            for question, answer in parsed:
                question_embedding = self.get_embeddings(question, embeddings_cache)
                answer_embedding = self.get_embeddings(answer, embeddings_cache)
                is_unique = True
                
                for other_pair in unique_qa_pairs:
                    if util.pytorch_cos_sim(question_embedding, other_pair['embeddings'][0]) > similarity_threshold and \
                    util.pytorch_cos_sim(answer_embedding, other_pair['embeddings'][1]) > similarity_threshold:
                        is_unique = False
                        break

                if is_unique:
                    unique_qa_pairs.append({
                        'context': data['chunk_context'],
                        'question': question,
                        'answer': answer,
                        'score': data['score'],
                        'embeddings': (question_embedding, answer_embedding),
                    })

        return_val = {}
        for pair in unique_qa_pairs:
            key = pair['context']
            if key not in return_val:
                return_val[key] = []
            return_val[key].append({
                'question': pair['question'],
                'answer': pair['answer'],
                'score': pair['score']
            })

        if json_output:
            return json.dumps(return_val, indent=4)
        return return_val


PIPELINE_REGISTRY.register_pipeline("tex2text-generation", pipeline_class=QABuilderPipeline, pt_model=T5ForConditionalGeneration)


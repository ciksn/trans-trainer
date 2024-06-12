import argparse
import itertools
import json
import os
import random
import time
import copy
from functools import partial
from typing import Optional


import logging
import torch
from tqdm import tqdm
from PIL import Image
import pandas as pd

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import torchvision
import glob
import numpy as np
from PIL import Image
import json

ds_collections = {
    'mmbench_dev_20230712': {
        'raw_file': 'mmbench_dev_20230712.tsv',
        'annotation': 'mmbench_dev_20230712.jsonl',
        'max_new_tokens': 10,
    },
    'mmbench_test_20230712': {
        'raw_file': 'mmbench_test_20230712.tsv',
        'annotation': 'mmbench_test_20230712.jsonl',
        'max_new_tokens': 10,
    },
    'car':{
        'annotation':'/home/zeyu/work/deep_learning/codes_from_github/large-models/mPLUG-Owl/mPLUG-Owl2/data/virtual_car/mmbench_type/mmbench.json',
        'max_new_tokens':10,
    }
}

multiple_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

def mapping_to_annotation(results, raw_annotation):
    outputs = []
    for result in results:
        index, prediction = result['index'], result['prediction']
        row_df = raw_annotation[raw_annotation['index'] == index].squeeze().to_dict()
        output = {
            "index": index,
            "image": row_df['image'],
            "question": row_df['question'],
            "answer": row_df.get('answer', None),
            "options": [y for y in [row_df.get(x, None) for x in 'ABCDEFGHI'] if isinstance(y, str)],
            "prediction": prediction,
            "l2-category": row_df['l2-category']
        }
        outputs.append(output)
    return outputs


def generate_submission_file(results, raw_annotation):
    outputs = []
    for result in results:
        index, prediction = result['index'], result['prediction']
        row_df = raw_annotation[raw_annotation['index'] == index].squeeze().to_dict()
        output = {
            "index": index,
            "question": row_df['question'],
            "prediction": prediction,
            "A": row_df.get('A', None),
            "B": row_df.get('B', None),
            "C": row_df.get('C', None),
            "D": row_df.get('D', None),
            "E": row_df.get('E', None),
            "F": row_df.get('F', None),
            "G": row_df.get('G', None),
            "H": row_df.get('H', None),
            "I": row_df.get('I', None),
        }
        outputs.append(output)
    return outputs

def collate_fn(batches, tokenizer):

    questions = [_['question'] for _ in batches]
    indices = [_['index'] for _ in batches]

    image_tensor = [_['image_tensor'] for _ in batches]

    input_ids = []
    for input_text in questions:
        input_ids.append(tokenizer_image_token(input_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').tolist())
    input_tokens_max_length = max([len(x) for x in input_ids])
    pad_token_id = tokenizer.pad_token_id

    input_ids = [([pad_token_id] * (input_tokens_max_length - len(_)) + _) for _ in input_ids] # pad in the left
    input_ids = torch.LongTensor(input_ids)
    attention_mask = 1 - input_ids.eq(pad_token_id).long()
    
    image_tensor = torch.cat(image_tensor, dim=0)
    return image_tensor, input_ids, attention_mask, indices


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, test, prompt, image_processor):
        
        self.prompt = prompt
        self.image_processor = image_processor

        self.data = json.load(open(test))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        index = data['index']
        image = data['image']
        hint = data['hint'] if data['hint'] else 'N/A'
        question = data['question']

        choices = data['choices']
        choice_list = []
        for i, c in enumerate(choices):
            choice_list.append('{}. {}'.format(multiple_choices[i], c))
        choice_txt = '\n'.join(choice_list)

        image = Image.open(image).convert('RGB')
        max_edge = max(image.size)
        image = image.resize((max_edge, max_edge)) # Resize here for best performance
        image_tensor = process_images([image], self.image_processor)

        return {
            'index': index,
            'image_tensor': image_tensor,
            'question': self.prompt.format(hint,question, choice_txt),
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',filename='app.log',filemode='w')
    video_folder = "/home/zeyu/work/deep_learning/codes_from_github/large-models/mPLUG-Owl/mPLUG-Owl2/data/virtual_car/videos"
    target_folder = "/home/zeyu/work/deep_learning/codes_from_github/large-models/mPLUG-Owl/mPLUG-Owl2/data/virtual_car/images"
    logging.info('video source:'+video_folder)
    logging.info('splitted image:'+target_folder)
    for video in glob.glob(video_folder+'/*'):
        video_name = video.split('/')[-1][:-4]
        video_tensor,_,_ = torchvision.io.read_video(video,pts_unit='sec')
        video_tensor = video_tensor[48:]
        index = np.linspace(0,video_tensor.size(0)-1,8)
        video_tensor = video_tensor[index]
        for frame in range(video_tensor.size(0)):
            current_frame = video_tensor[frame].numpy()
            image = Image.fromarray(current_frame)
            image.save(target_folder+'/'+video_name+'_'+str(frame)+'.jpg',)
        print(video_name)
    logging.info('split complete')

    images_folder = "/home/zeyu/work/deep_learning/codes_from_github/large-models/mPLUG-Owl/mPLUG-Owl2/data/virtual_car/images"
    target_file = "/home/zeyu/work/deep_learning/codes_from_github/large-models/mPLUG-Owl/mPLUG-Owl2/data/virtual_car/mmbench_type/mmbench.json"
    logging.info('generated question path:'+target_file)
    question_candidate={
    "scerario" : ["suburbs","city street","expressway","tunnel","parking-lot","gas or charging stations","unknown"],
    "weather" : ["clear","cloudy","raining","foggy","snowy","unknown"],
    "period" : ["daytime","dawn or dusk","night","unknown"],
    "road_structure" : ["normal","crossroads","T-junction","ramp","lane merging","parking lot entrance","round about","unknown"],
    "general_obstacle" : ["nothing","speed bumper","traffic cone","water horse","stone","manhole cover","nothing","unknown"],
    "abnormal_condition" : ["uneven","oil or water stain","standing water","cracked","nothing","unknown"],
    "ego_car_behavior" : ["slow down","go straight","turn right","turn left","stop","U-turn","speed up","lane change","others"],
    "closest_participants_type" : ["passenger car","bus","truck","pedestrain","policeman","nothing","others","unknown"],
    "closest_participants_behavior" : ["slow down","go straight","turn right","turn left","stop","U-turn","speed up","lane change","others"],
    }
    
    hint = {
    "scerario":f'Consider the setting or environment where the image is taking place. If there are soils or stones or mountains and no traffic signals, it is likely to be suburbs. If there are street lamps or traffic lights, it is likely to be city street',
    "weather": f'Focus on the atmospheric conditions during the scenario. Check if there is rain or snow or clouds, otherwise the weather is clear. If the car inside a parking-lot, the weather is unknown. Clear means you have better sight than foggy.',
    "period" : f'Contemplate the time of the day during the scenario. In most cases if it is not dark outside, it means daytime otherwise dark. If and only if the car is in the parking-lot, the time is unknown.',
    "road_structure" : f'Focus on the layout of the road structure. If there are straight lanes or surrounded by road shoulders, it means normal. In most cases, it is normal when driving in suburbs with stones and soils. If and only if there is a point where two or more roads intersect, providing a junction for cars to choose different directions and with traffic lights, it means crossroad. If the car is in the parking-lot, it means the road structure is parking lot entrance. If the structure is not mentioned above, choose the best option.',
    "general_obstacle" : f'Contemplate any obstacles present on the road. Check if there is traffic cone or speed bumper or other obstacles near the car and the road. If the car is driving and no obstacles mentioned above, it means nothing obstacle.',
    "abnormal_condition" : f'Focus on road abnormal conditions. Check if there is standing water or oil or some cracks on the ground or other abnormal conditions. If there is no such condition, it means that nothing is abnormal',
    "ego_car_behavior" : f'Focus on the behavior of the ego car. If the car is driving normally, it means "go straight". If the car cross two lanes or more, it means "turning" left or right. If the car move from one lane to another, it means lane change. If the car takes the action not mentioned above, choose the best option',
    "closest_participants_type" : f'Identify the types of participants closest to the ego car. Think of the differences between passenger car and other forms of car. Most of the car on the road is passengers car. If there is no car or person, it means nothing.',
    "closest_participants_behavior" : f'Focus on the directions and actions of the closest participants especially the car that closest to the ego car. The angle of the view may be outside of the car. If the car is driving normally, it means "go straight". If it is at the crossroad and turing, it means turning or U-turn. If the car change the lane from one to another by a small movement and not facing towards, it means lane change. If the braking light is on or there is a traffic cone behind the car, the car stops. If the car takes any actions not mentioned above, choose the best option'
    }

    questions = {
    "scerario" : "<scenario>.",
    "weather" : "<weather>.",
    "period" : "<period>.",
    "road_structure" :"<road structure>.",
    "general_obstacle" :"<general_obstacle> which means obstacles present on the road from following options.",
    "abnormal_condition" :"<abnormal_condition>.",
    "ego_car_behavior" :"<ego_car_behavior> which means the behavior of the ego car from following options.",
    "closest_participants_type" :"<closest_participants_type> from following options." ,
    "closest_participants_behavior" :"<closest_participants_behavior> which means the car that closest to the ego car from following options." ,
    }

    output_cache = []
    total_index = 0
    for image_path in glob.glob(images_folder+'/*'):
        image_name = image_path.split('/')[-1][:-4]
        for question_name in question_candidate.keys():
            question_dict = {} 
        
            question_dict['index'] = total_index
            question_dict['image'] = image_path
            question_dict['hint'] = hint[question_name]
            question_dict['question'] = "Based on Image and Hint, choose the best option for "+ questions[question_name]
            question_dict['choices'] = question_candidate[question_name]
            question_dict['question_name'] = question_name
        
            output_cache.append(question_dict)
            total_index += 1

    with open(target_file,mode='w') as f:
        json.dump(output_cache,f)
    logging.info('generation of hint and question complete')
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/home/zeyu/.cache/modelscope/hub/damo/mPLUG-Owl2')
    parser.add_argument('--dataset', type=str, default='car')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    os.environ['CUDA_VISIBLE_DEVICES'] = os.getenv('LOCAL_RANK', "0")

    model_path = args.checkpoint
    model_name = get_model_name_from_path(model_path)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device_map="cuda", device="cuda")
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eos_token_id

    prompt = "USER: <|image|>\nHint:{}\n{}\n{}\nAnswer only with the option’s letter from the given choices directly. ASSISTANT:"

    random.seed(args.seed)
    dataset = VQADataset(
        test=ds_collections[args.dataset]['annotation'],
        prompt=prompt,
        image_processor=image_processor,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    outputs = []
    for _, (image_tensor, input_ids, attention_mask, indices) in tqdm(enumerate(dataloader)):
        pred = model.generate(
            input_ids=input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            images=image_tensor.to(dtype=model.dtype).cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=ds_collections[args.dataset]['max_new_tokens'],
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            output_hidden_states=True,
            use_cache=True,
        )
        answers = [
            tokenizer.decode(_[input_ids.size(1):].cpu(),
                             skip_special_tokens=True).strip() for _ in pred
        ]

        for index, answer in zip(indices, answers):
            outputs.append({
                'index': index,
                'prediction': answer,
            })

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    if torch.distributed.get_rank() == 0:
        print(f"Evaluating {args.dataset} ...")
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = f'/home/zeyu/work/deep_learning/codes_from_github/large-models/mPLUG-Owl/mPLUG-Owl2/data/virtual_car/output_without_merge/raw.json'
        json.dump(merged_outputs, open(results_file, 'w'), ensure_ascii=False)

        # mapped_result = mapping_to_annotation(merged_outputs, pd.read_csv(ds_collections[args.dataset]['raw_file'], sep='\t'))

        # submission_res = generate_submission_file(merged_outputs, pd.read_csv(ds_collections[args.dataset]['raw_file'], sep='\t'))
        # res_df = pd.DataFrame(submission_res)
        # metrics_file = f'{args.dataset}_{time_prefix}_s{args.seed}_submission.xlsx'
        # res_df.to_excel(metrics_file, index=False)

    torch.distributed.barrier()
    logging.info('inference complete')
    
    input_file_path = "/home/zeyu/work/deep_learning/codes_from_github/large-models/mPLUG-Owl/mPLUG-Owl2/data/virtual_car/mmbench_type/mmbench.json"
    logging.info('output without merge path:'+input_file_path)
    output_answer_path = "/home/zeyu/work/deep_learning/codes_from_github/large-models/mPLUG-Owl/mPLUG-Owl2/data/virtual_car/output_without_merge/raw.json"
    target_path = "/home/zeyu/work/deep_learning/codes_from_github/large-models/mPLUG-Owl/mPLUG-Owl2/data/virtual_car/output_merged/submit.json"
    logging.info('output merged path:'+target_path)
    question_candidate={
    "scerario" : ["suburbs","city street","expressway","tunnel","parking-lot","gas or charging stations","unknown"],
    "weather" : ["clear","cloudy","raining","foggy","snowy","unknown"],
    "period" : ["daytime","dawn or dusk","night","unknown"],
    "road_structure" : ["normal","crossroads","T-junction","ramp","lane merging","parking lot entrance","round about","unknown"],
    "general_obstacle" : ["nothing","speed bumper","traffic cone","water horse","stone","manhole cover","nothing","unknown"],
    "abnormal_condition" : ["uneven","oil or water stain","standing water","cracked","nothing","unknown"],
    "ego_car_behavior" : ["slow down","go straight","turn right","turn left","stop","U-turn","speed up","lane change","others"],
    "closest_participants_type" : ["passenger car","bus","truck","pedestrain","policeman","nothing","others","unknown"],
    "closest_participants_behavior" : ["slow down","go straight","turn right","turn left","stop","U-turn","speed up","lane change","others"],
}

    candidate ={
    'A':0,
    'B':0,
    'C':0,
    'D':0,
    'E':0,
    'F':0,
    'G':0,
    'H':0,
    'I':0,
    'J':0,
    'K':0,
}

    output={
    "author" : "tom" ,
    "time" : "16th_Nov.",
    "model" : "base",
    "test_results" :[]}

    with open(input_file_path,mode='r') as f:
        input_file = json.load(f)
    with open(output_answer_path,mode='r') as f:
        output_answer = json.load(f)

    image_question = {}
    for idx in range(len(input_file)):
        main_image = input_file[idx]['image'].split('/')[-1].split('_')[0]
        question_name = input_file[idx]['question_name']
        chosen_option = output_answer[idx]['prediction']
        if not chosen_option in candidate.keys():
            continue
        if not main_image in image_question.keys():
            image_question[main_image] = {}
            image_question[main_image][question_name] = copy.deepcopy(candidate)
            image_question[main_image][question_name][chosen_option] += 1
        else:
            if not question_name in image_question[main_image].keys():
                image_question[main_image][question_name] = copy.deepcopy(candidate)
            else:
                image_question[main_image][question_name][chosen_option] += 1

    for image_name in image_question.keys():
        tmp = {}
        if int(image_name) >= 61:
            tmp['clip_id'] = image_name+'.mp4'
        else:
            tmp['clip_id'] = image_name+'.avi'
        for question_name in image_question[image_name].keys():
        # if question_name == "ego_car_behavior" or question_name == "closest_participants_behavior":
            if False:
                list_option = []
                tmp[question_name] = ""
                for option in image_question[image_name][question_name].keys():
                # if question_name == "road_structure":
                #     if (ord(option)-ord('A')) < 8:
                #         print(image_name,question_candidate["road_structure"][ord(option)-ord('A')],image_question[image_name][question_name][option])
                    if image_question[image_name][question_name][option] > 0:
                        list_option.append(question_candidate[question_name][ord(option)-ord('A')])
                for i,item in enumerate(list_option):
                    tmp[question_name] += item
                    if i<len(list_option)-1:
                        tmp[question_name] +=', '
            else:
                max = 0
                chosen = "A"
                for option in image_question[image_name][question_name].keys():
                # if question_name == "road_structure":
                #     if (ord(option)-ord('A')) < 8:
                #         print(image_name,question_candidate["road_structure"][ord(option)-ord('A')],image_question[image_name][question_name][option])
                    if image_question[image_name][question_name][option] >= max:
                        max = image_question[image_name][question_name][option]
                        chosen = option
                tmp[question_name] = question_candidate[question_name][ord(chosen)-ord('A')]
        # if question_name == "ego_car_behavior":
        #     if tmp['scerario'] == 'parking-lot':
        #         tmp[question_name] = "slow down"
        for question_name in image_question[image_name].keys():
            if question_name == 'closest_participants_behavior':
                if tmp['road_structure'] == 'crossroads':
                    tmp[question_name] = 'U-turn'
            
        output["test_results"].append(tmp)    

    with open(target_path,mode='w') as f:
        json.dump(output,f)
    logging.info('merge output complete')
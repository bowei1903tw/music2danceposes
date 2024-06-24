import argparse
import os
import shutil
from tqdm import tqdm
from natsort import natsorted

import torch
from torch import nn

from model.mert_adapter import HubertModel
from model.dataset import LoadDataset
from transformers import HubertConfig
from model.utils import create_video

import logging
logger = logging.getLogger(__name__)

allframe_path = './image_database'

class MERTadapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.HubertConfig = HubertConfig()
        self.HubertConfig.ratio = config.ratio
        self.HubertConfig.adapter_layer_index = config.adapter_layer_index
        self.HubertConfig.num_adapter_attention_heads = config.num_adapter_attention_heads
        self.HubertConfig.apply_spec_augment = config.apply_spec_augment        
        self.model_hubert = HubertModel(self.HubertConfig)

        self.soft = nn.Softmax(dim=-1)

    def forward(self, image_database_npz, query_audio_segment, query_audio_segment_mask, image_database_name, frame_index, top_k, audio_name, total_frames):
        
        image_database_npz = image_database_npz / image_database_npz.norm(dim=-1, keepdim=True)

        for n_segment in tqdm(range(query_audio_segment.shape[0]), total=query_audio_segment.shape[0], desc="  Retrieving...."):

            if torch.sum(query_audio_segment[n_segment]) != 0 : 

                ret = self.model_hubert(input_values=query_audio_segment[n_segment].unsqueeze(0), attention_mask=query_audio_segment_mask[n_segment].unsqueeze(0), mask_time_indices=None, output_attentions=True, output_hidden_states=True)  
                n_segment_audio_segment = ret[0]

                retrieve_logits = self.model_hubert.logit_scale * torch.matmul(n_segment_audio_segment, image_database_npz.t())
                    
                save_path = os.path.join(self.config.retrieval_result, 'segment_image' ,audio_name + f'_{n_segment+1}_{frame_index[n_segment]}')

            else:
                save_path = os.path.join(self.config.retrieval_result, 'segment_image' ,audio_name + f'_{n_segment+1}_{frame_index[n_segment]}')

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                for k in range(top_k):
                    shutil.copy2(os.path.join(os.getcwd(),'basic_pose.jpg'), os.path.join(save_path,audio_name + f'_{n_segment+1}_top{k+1}.jpg'))
                continue

            sorted_index = torch.argsort(self.soft(retrieve_logits),dim=-1, descending=True)

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for k in range(top_k):
                shutil.copy2(os.path.join(allframe_path, image_database_name[sorted_index[:,k]].split('.')[0]+'.png'), os.path.join(save_path, audio_name + f'_{n_segment+1}_top{k+1}.jpg'))

        if total_frames-1 != frame_index[-1]:
            save_path = os.path.join(self.config.retrieval_result, 'segment_image' ,audio_name + f'_{n_segment+2}_{total_frames-1}')

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for k in range(top_k):
                shutil.copy2(os.path.join(os.getcwd(),'basic_pose.jpg'), os.path.join(save_path,audio_name + f'_{n_segment+2}_top{k+1}.jpg'))

        re = os.path.join(self.config.retrieval_result, 'segment_image')
        re_list=natsorted(os.listdir(re))
        for re_seg in re_list:
            re_seg_topk_list=natsorted(os.listdir(f'{self.config.retrieval_result}/segment_image/{re_seg}'))
            for idx,re_seg_k in enumerate(re_seg_topk_list):
                if not os.path.exists(f'{self.config.retrieval_result}/top_k/segment_image_top{idx+1}'):
                    os.makedirs(f'{self.config.retrieval_result}/top_k/segment_image_top{idx+1}')

                shutil.copy2(f'{self.config.retrieval_result}/segment_image/'+re_seg+f'/{re_seg_k}', f'{self.config.retrieval_result}/top_k/segment_image_top{idx+1}/'+re_seg_k[:-4]+'_'+re_seg.split('_')[-1]+'.jpg')

def get_args():
    """parse and preprocess cmd line args"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--user_given_waveform", type=str, default='./waveform')
    parser.add_argument("--image_database_npz_dir", type=str, default='./CLIFF_output')
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--retrieval_result", type=str, default='./retrieval_result')
    parser.add_argument("--beat_resolution", type=str, required=True, choices=['4th', '8th', '16th'])

    # model config
    parser.add_argument("--num_adapter_attention_heads", type=int, default=1)
    parser.add_argument("--ratio", type=int, default=12)
    parser.add_argument("--image_feature_type", type=str, default="shape_pose_82")
    parser.add_argument("--en_de_attention_adapter", type=bool, default=True)
    parser.add_argument("--adapter_layer_index", type=str, default='1') 

    #mert model config
    parser.add_argument("--activation_dropout", type=float, default=0.1)
    parser.add_argument("--apply_spec_augment", type=bool, default=False)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--conv_bias", type=bool, default=False)
    parser.add_argument("--conv_dim", default=[512,512,512,512,512,512,512])
    parser.add_argument("--conv_kernel", default=[10,3,3,3,3,2,2])
    parser.add_argument("--conv_stride", default=[5,2,2,2,2,2,2])
    parser.add_argument("--do_stable_layer_norm", type=bool, default=False)
    parser.add_argument("--feat_extract_activation", type=str, default='gelu')
    parser.add_argument("--feat_extract_norm", type=str, default='group')
    parser.add_argument("--feat_proj_dropout", type=float, default=0.0)
    parser.add_argument("--feat_proj_layer_norm", type=bool, default=True)
    parser.add_argument("--hidden_act", type=str, default='gelu')
    parser.add_argument("--hidden_dropout", type=float, default=0.1)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--layer_norm_eps", type=float, default=1e-5)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--num_conv_pos_embedding_groups", type=int, default=16)
    parser.add_argument("--num_conv_pos_embeddings", type=int, default=128)
    parser.add_argument("--num_feat_extract_layers", type=int, default=7)
    parser.add_argument("--num_hidden_layers", type=int, default=12)

    # others
    parser.add_argument("--no_cuda", action="store_true", help="run on cpu")

    parser.add_argument("--Wav2Vec2FeatureExtractor_dir", type=str, default="./Wav2Vec2FeatureExtractor/preprocessor_config.json") 
    parser.add_argument("--pretrained_weight", type=str, default="./checkpoint/edsa_adapter_checkpoint.chkpt")
    
    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    opt.retrieval_result = opt.retrieval_result+'_'+opt.beat_resolution

    adapter_layer_index = [0]*12
    for i in opt.adapter_layer_index.split(','):
        adapter_layer_index[int(i)-1] = 1
    opt.adapter_layer_index = adapter_layer_index

    if not os.path.exists(opt.retrieval_result):
        os.makedirs(opt.retrieval_result)

    return opt



def main():
    opt = get_args()

    device = torch.device('cuda' if True else 'cpu')
    data = LoadDataset(opt)
    query_audio_segment = data.audio_segments
    query_audio_segment_mask = data.audio_segments_mask
    frame_index = data.frame_index
    total_frames = data.total_frames
    image_database_npz = data.image_database_npz
    image_database_name =  data.image_file_name
    audio_name = data.audio_file[0].split('.')[0]

    logger.info("Use model with en_de_attention_adapter")
    logger.info(f'beat resolution {opt.beat_resolution}')
    logger.info(f'query_audio_n_segment:{query_audio_segment.shape[0]}')
    logger.info(f'retrieve top_{opt.top_k} pose')

    model = MERTadapter(opt).to(device)
    model.load_state_dict(torch.load(opt.pretrained_weight)['model'])
    model.eval()
    with torch.no_grad():
        model.forward(image_database_npz.to(device), query_audio_segment.to(device), query_audio_segment_mask.to(device), image_database_name, frame_index, opt.top_k, audio_name, total_frames)
    
    create_video(opt).create()
    logger.info("Done")
    logger.info(f"The videos for retrieval results can be found in the 'retrieval_result_{opt.beat_resolution}' folder.")


if __name__ == "__main__":
    main()